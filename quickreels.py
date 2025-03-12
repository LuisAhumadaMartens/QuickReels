import cv2
import tensorflow as tf
import numpy as np
import argparse
import glob
import os
import sys
import math
import subprocess
import json
from flask import Flask, request, jsonify
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple
import base64

# -------------------------------
# GPU Configuration
# -------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# -------------------------------
# Constants
# -------------------------------
DETECTION_CONFIDENCE_THRESHOLD = 0.3
CONFIDENCE_MARGIN = 0.15
DEFAULT_CENTER = 0.5  # normalized center (50%)
ASPECT_RATIO = 9 / 16  # output crop aspect ratio (width based on full height)
MOVE_NET_INPUT_SIZE = (192, 192)

# -------------------------------
# Global variable for previous gray frame (for scene change detection)
# -------------------------------
prev_gray_frame = None

# -------------------------------
# Load MoveNet model
# -------------------------------
movenet = tf.saved_model.load('/Users/luis/Developer/Temporal/MediastreamTensor/VideoToShort/model')
movenet_func = movenet.signatures['serving_default']

# -------------------------------
# Utility: Preprocess frame for MoveNet
# -------------------------------
def prepare_input_tensor(frame):
    # Resize with OpenCV first (faster)
    frame = cv2.resize(frame, MOVE_NET_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensor and add batch dimension
    input_tensor = tf.convert_to_tensor(frame, dtype=tf.int32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    return input_tensor

# -------------------------------
# Utility: Mean Squared Error
# -------------------------------
def mse(imageA, imageB):
    """Compute Mean Squared Error between two images."""
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

# -------------------------------
# Utility: Find default video input
# -------------------------------
def find_default_input():
    video_formats = ["mp4", "avi", "mov", "mkv"]
    for ext in video_formats:
        matches = glob.glob(f"input.{ext}")
        if matches:
            return matches[0]
    return None

# -------------------------------
# Utility: Get video metadata
# -------------------------------
def get_video_metadata(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return width, height, fps, frame_count

# -------------------------------
# Utility: Cluster close detections
# -------------------------------
def cluster_people(people, threshold=0.05):
    """
    Merge detections that are very close.
    Each detection in `people` is a tuple (id, kp) where kp = (y, x, confidence, ...).
    Returns a list of clusters as tuples: (cluster_id, avg_x, avg_y, max_confidence).
    """
    clusters = []
    for person in people:
        person_id, keypoints = person
        y_pos, x_pos, confidence = keypoints[:3]
        added = False
        for cluster in clusters:
            center_x = cluster["sum_x"] / cluster["count"]
            center_y = cluster["sum_y"] / cluster["count"]
            distance = math.sqrt((x_pos - center_x)**2 + (y_pos - center_y)**2)
            if distance < threshold:
                cluster["sum_x"] += x_pos
                cluster["sum_y"] += y_pos
                cluster["count"] += 1
                if confidence > cluster["max_conf"]:
                    cluster["max_conf"] = confidence
                added = True
                break
        if not added:
            clusters.append({"sum_x": x_pos, "sum_y": y_pos, "count": 1, "max_conf": confidence})
    
    merged_people = []
    for i, cluster in enumerate(clusters):
        avg_x = cluster["sum_x"] / cluster["count"]
        avg_y = cluster["sum_y"] / cluster["count"]
        merged_people.append((i, avg_x, avg_y, cluster["max_conf"]))
    return merged_people

# -------------------------------
# Movement Planner: collects normalized x centers per frame and scene changes
# -------------------------------
class MovementPlanner:
    def __init__(self, fps):
        # List of tuples (frame_num, x_pos, is_scene_change)
        self.frame_data = []
        self.fps = fps
        self.current_scene_start = 0
        self.waiting_for_detection = False
        self.default_x = DEFAULT_CENTER
        self.smoothing_rate = 0.05
        self.max_movement_per_frame = 0.03
        self.position_history = []
        self.history_size = 3
        self.centering_weight = 0.4
        self.fast_transition_threshold = 0.1
        self.in_transition = False
        self.stable_frames = 0
        self.stable_frames_required = int(fps * 0.5)  # Half a second worth of frames
        self.is_centering = False

    def plan_movement(self, frame_num, cluster, frame_diff, scene_change_threshold):
        if frame_diff > scene_change_threshold:
            self.position_history = []
            self.in_transition = False
            self.stable_frames = 0
            self.is_centering = False
            # Reset history on scene change
            self.frame_data.append((frame_num, self.default_x, True))
            self.current_scene_start = frame_num
            self.waiting_for_detection = True
            return

        if self.waiting_for_detection and cluster:
            # Found first detection after scene change
            new_x = cluster[1]  # Normalized x position from cluster
            
            # Go back and update all frames since scene change
            for i, (f_num, _, is_scene) in enumerate(self.frame_data):
                if f_num >= self.current_scene_start:
                    if is_scene:
                        continue  # Keep scene change marker
                    self.frame_data[i] = (f_num, new_x, False)
            
            self.waiting_for_detection = False
            return

        # Get the target position
        if cluster:
            target_x = cluster[1]
        else:
            target_x = self.default_x if not self.frame_data else self.frame_data[-1][1]

        if self.frame_data:
            last_x = self.frame_data[-1][1]
            distance_to_target = abs(target_x - last_x)
            
            # Always apply normal movement first
            max_move = self.max_movement_per_frame
            movement = target_x - last_x
            if abs(movement) > max_move:
                target_x = last_x + (max_move if movement > 0 else -max_move)

            # Check if we're within the delta threshold
            if distance_to_target < self.fast_transition_threshold:
                self.stable_frames += 1
                if self.stable_frames >= self.stable_frames_required:
                    self.is_centering = True
            else:
                self.stable_frames = 0
                self.is_centering = False
                self.position_history = []

            # Only apply centering after we've been stable long enough
            if self.is_centering:
                self.position_history.append(target_x)
                if len(self.position_history) > self.history_size:
                    self.position_history.pop(0)

                if len(self.position_history) > 1:
                    avg_pos = sum(self.position_history) / len(self.position_history)
                    target_x = target_x * (1 - self.centering_weight) + avg_pos * self.centering_weight

        self.frame_data.append((frame_num, target_x, False))

    def get_scene_segments(self):
        """
        Returns a list of scene segments, where each segment is a tuple:
        (start_frame, end_frame, [positions])
        """
        segments = []
        current_segment_start = 0
        current_positions = []

        for i, (frame_num, x_pos, is_scene) in enumerate(self.frame_data):
            if is_scene and i > 0:
                # End current segment
                segments.append((
                    current_segment_start,
                    frame_num - 1,
                    current_positions
                ))
                # Start new segment
                current_segment_start = frame_num
                current_positions = []
            
            current_positions.append(x_pos)

        # Add final segment
        if current_positions:
            segments.append((
                current_segment_start,
                self.frame_data[-1][0],
                current_positions
            ))

        return segments

    def interpolate_and_smooth(self, total_frames, base_alpha=0.1, delta_threshold=0.015):
        """
        Smooth each scene segment independently with variable smoothing rates
        - Faster for large movements
        - Slower as it approaches target (deceleration)
        """
        smoothed_centers = [self.default_x] * total_frames
        segments = self.get_scene_segments()

        for start_frame, end_frame, positions in segments:
            last_x = positions[0]
            for i, frame in enumerate(range(start_frame, end_frame + 1)):
                if i >= len(positions):
                    break
                
                target_x = positions[i]
                delta = target_x - last_x
                
                if abs(delta) < delta_threshold:
                    smoothed_x = last_x
                else:
                    # Variable smoothing rate based on distance to target
                    # Slower smoothing (smaller alpha) as we get closer
                    distance_factor = min(abs(delta) * 2, 1.0)  # Scale based on distance
                    deceleration_alpha = base_alpha * distance_factor
                    
                    # Even slower for final approach
                    if abs(delta) < 0.1:  # Within 10% of target
                        deceleration_alpha *= 0.5  # Half speed for final approach
                    
                    smoothed_x = last_x + deceleration_alpha * delta
                
                smoothed_centers[frame] = smoothed_x
                last_x = smoothed_x

        return smoothed_centers

# -------------------------------
# PersonTracker (for real-time processing; unchanged)
# -------------------------------
class PersonTracker:
    def __init__(self):
        self.state = {
            'current_id': None,
            'current_confidence': None,
            'position': (DEFAULT_CENTER, DEFAULT_CENTER),
            'lost_frames': 0,
            'lock_countdown': 0,
            'new_target': {
                'id': None,
                'confidence': None,
                'frames': 0
            }
        }
        self.wait_frames = 30
        self.lock_frames = 45

    def update_position(self, x, y):
        self.state['position'] = (x, y)

    def get_position(self):
        return self.state['position']

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def update(self, cluster, frame_diff, scene_change_threshold, movement_threshold):
        if frame_diff > scene_change_threshold:
            self.reset()
            return "Scene change detected. Resetting to center."
        
        self.state['lost_frames'] += 1

        if self.state['current_id'] is None:
            self.state['current_id'] = cluster[0]
            self.state['current_confidence'] = cluster[3]
            self.update_position(cluster[1], cluster[2])
            return f"Initializing on person {cluster[0]}."
        
        if self.distance(self.get_position()[0], self.get_position()[1], cluster[1], cluster[2]) > movement_threshold:
            self.state['current_id'] = cluster[0]
            self.state['current_confidence'] = cluster[3]
            self.update_position(cluster[1], cluster[2])
            return f"Person moved too far. Switching to person {cluster[0]}."

        if cluster[0] == self.state['current_id']:
            self.update_position(cluster[1], cluster[2])
            self.state['current_confidence'] = cluster[3]
            self.state['new_target']['id'] = None
            self.state['new_target']['frames'] = 0
            return f"Following person {self.state['current_id']}."

        if cluster[3] < (self.state['current_confidence'] + CONFIDENCE_MARGIN):
            self.update_position(cluster[1], cluster[2])
            self.state['new_target']['id'] = None
            self.state['new_target']['frames'] = 0
            return f"Candidate not strong enough; continuing on person {self.state['current_id']}."

        if self.state['lock_countdown'] > 0:
            self.update_position(cluster[1], cluster[2])
            return f"In lock; continuing on person {self.state['current_id']}."

        if (self.state['new_target']['id'] is None) or (self.state['new_target']['id'] != cluster[0]):
            self.state['new_target']['id'] = cluster[0]
            self.state['new_target']['confidence'] = cluster[3]
            self.state['new_target']['frames'] = 1
            return f"Candidate switch to person {self.state['new_target']['id']} started. (1/{self.wait_frames})"

        new_x = (self.state['new_target']['confidence'] * cluster[1] + self.state['current_confidence'] * self.state['position'][0]) / (self.state['new_target']['confidence'] + self.state['current_confidence'])
        new_y = (self.state['new_target']['confidence'] * cluster[2] + self.state['current_confidence'] * self.state['position'][1]) / (self.state['new_target']['confidence'] + self.state['current_confidence'])
        self.update_position(new_x, new_y)
        self.state['new_target']['frames'] += 1

        if self.state['new_target']['frames'] >= self.wait_frames:
            self.state['current_id'] = self.state['new_target']['id']
            self.state['current_confidence'] = self.state['new_target']['confidence']
            blended_x = self.state['new_target']['confidence'] * cluster[1] + (1 - self.state['new_target']['confidence']) * self.state['position'][0]
            blended_y = self.state['new_target']['confidence'] * cluster[2] + (1 - self.state['new_target']['confidence']) * self.state['position'][1]
            self.update_position(blended_x, blended_y)
            self.state['lock_countdown'] = self.lock_frames
            self.state['new_target']['id'] = None
            self.state['new_target']['frames'] = 0
            return f"Switched to person {self.state['current_id']}. Lock initiated."

        return f"Buffering candidate switch to person {self.state['new_target']['id']}... ({self.state['new_target']['frames']}/{self.wait_frames})"

    def handle_no_detection(self, frame_diff, scene_change_threshold):
        if frame_diff > scene_change_threshold:
            self.reset()
            return "Scene change detected. Resetting to center."
        self.state['lost_frames'] += 1
        if self.state['lost_frames'] < self.wait_frames:
            return f"No detection; holding last position ({self.state['lost_frames']}/{self.wait_frames})."
        else:
            self.reset()
            return "No detection for buffer duration. Resetting to center."

    def reset(self):
        self.state['current_id'] = None
        self.state['current_confidence'] = None
        self.state['position'] = (DEFAULT_CENTER, DEFAULT_CENTER)
        self.state['lost_frames'] = 0
        self.state['lock_countdown'] = 0
        self.state['new_target'] = {'id': None, 'confidence': None, 'frames': 0}

# -------------------------------
# API Handling
# -------------------------------
@dataclass
class VideoSegment:
    output_path: str
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None

class VideoProcessor:
    def __init__(self, input_path: str, segments: List[VideoSegment]):
        self.input_path = input_path
        self.segments = segments
        self.total_frames = None
        self._initialize_video_metadata()

    def _initialize_video_metadata(self):
        width, height, fps, total_frames = get_video_metadata(self.input_path)
        self.total_frames = total_frames
        
        # Fill in missing frame ranges with full video duration
        for segment in self.segments:
            if segment.start_frame is None:
                segment.start_frame = 0
            if segment.end_frame is None:
                segment.end_frame = total_frames - 1
                
            # Validate frame ranges
            if segment.start_frame < 0 or segment.end_frame >= total_frames:
                raise ValueError(f"Invalid frame range for {segment.output_path}: [{segment.start_frame}-{segment.end_frame}]")
            if segment.start_frame >= segment.end_frame:
                raise ValueError(f"Start frame must be less than end frame for {segment.output_path}")

    def process_all(self):
        # First pass: analyze the entire video once
        planner = MovementPlanner(fps)
        video = cv2.VideoCapture(self.input_path)
        frame_count = 0
        
        print("Analyzing video...")
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            # Your existing frame analysis code here
            # This populates the planner with all frame data
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_diff = mse(prev_gray_frame, gray) if prev_gray_frame is not None else 0
            prev_gray_frame = gray

            input_tensor = prepare_input_tensor(frame)
            outputs = movenet_func(input_tensor)
            keypoints = outputs['output_0'].numpy()[0]

            people = [(i, kp) for i, kp in enumerate(keypoints) if kp[2] > DETECTION_CONFIDENCE_THRESHOLD]
            merged_people = cluster_people(people, threshold=0.05)

            if merged_people:
                best_cluster = max(merged_people, key=lambda c: c[3])
                planner.plan_movement(frame_count, best_cluster, frame_diff, scene_change_threshold=3000)
            else:
                planner.plan_movement(frame_count, None, frame_diff, scene_change_threshold=3000)
            
            frame_count += 1
            print(f"Analyzing: {(frame_count / self.total_frames) * 100:.2f}%", end='\r')

        video.release()

        # Get smoothed centers for the entire video
        smoothed_centers = planner.interpolate_and_smooth(self.total_frames)

        # Second pass: Process each segment
        for segment in self.segments:
            print(f"\nProcessing segment: {segment.output_path} [{segment.start_frame}-{segment.end_frame}]")
            self._process_segment(segment, smoothed_centers)

    def _process_segment(self, segment: VideoSegment, smoothed_centers: List[float]):
        video = cv2.VideoCapture(self.input_path)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        
        crop_width = int(height * ASPECT_RATIO)
        output_height = height
        output_width = crop_width
        
        writer = cv2.VideoWriter(
            segment.output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (output_width, output_height)
        )

        video.set(cv2.CAP_PROP_POS_FRAMES, segment.start_frame)
        frame_count = segment.start_frame

        while frame_count <= segment.end_frame:
            success, frame = video.read()
            if not success:
                break

            # Use the smoothed center for this frame
            norm_center = smoothed_centers[frame_count]
            
            # Your existing frame cropping code here
            x_center = int(norm_center * width)
            x_start = x_center - (crop_width // 2)
            if x_start < 0:
                x_start = 0
            elif x_start + crop_width > width:
                x_start = width - crop_width
            x_end = x_start + crop_width

            cropped_frame = frame[:, x_start:x_end].copy()
            writer.write(cropped_frame)
            
            frame_count += 1
            print(f"Processing {segment.output_path}: {((frame_count - segment.start_frame) / (segment.end_frame - segment.start_frame + 1)) * 100:.2f}%", end='\r')

        video.release()
        writer.release()

@app.route('/process-video', methods=['POST'])
def process_video_api():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        segments_data = request.form.get('segments', '[]')
        segments_data = json.loads(segments_data)
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input:
            video_file.save(temp_input.name)
            input_path = temp_input.name

        # Create segments
        segments = []
        output_files = []
        for segment in segments_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
                output_files.append(temp_output.name)
                segments.append(VideoSegment(
                    output_path=temp_output.name,
                    start_frame=segment.get('start_frame'),
                    end_frame=segment.get('end_frame')
                ))

        # Process video
        processor = VideoProcessor(input_path, segments)
        processor.process_all()

        # Read all processed videos
        results = []
        for output_path in output_files:
            with open(output_path, 'rb') as f:
                results.append(f.read())

        # Clean up
        os.unlink(input_path)
        for output_path in output_files:
            os.unlink(output_path)

        # Return results
        return jsonify({
            'videos': [base64.b64encode(video).decode('utf-8') for video in results]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------------------
# Main Entry Point
# -------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process video for Instagram Reels/TikTok format')
    parser.add_argument('-i', '--input', required=True, help='Input video file')
    parser.add_argument('-o', '--output', action='append', nargs='+', 
                        help='Output file with optional frame range [start-end]. Example: -o output1.mp4 [123-834]')
    
    args = parser.parse_args()
    
    # Process output arguments into segments
    segments = []
    if args.output:
        for output_group in args.output:
            output_file = output_group[0]
            frame_range = output_group[1] if len(output_group) > 1 else None
            
            start_frame = None
            end_frame = None
            
            if frame_range:
                if not frame_range.startswith('[') or not frame_range.endswith(']'):
                    raise ValueError(f"Frame range must be in format [start-end], got: {frame_range}")
                try:
                    start_frame, end_frame = map(int, frame_range[1:-1].split('-'))
                except ValueError:
                    raise ValueError(f"Invalid frame range format: {frame_range}. Must be [start-end]")
            
            segments.append(VideoSegment(
                output_path=output_file,
                start_frame=start_frame,
                end_frame=end_frame
            ))
    
    return args.input, segments

def main():
    # Check if running as CLI or API
    if len(sys.argv) > 1:
        # CLI mode
        input_path, segments = parse_arguments()
        processor = VideoProcessor(input_path, segments)
        processor.process_all()
    else:
        # API mode
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

if __name__ == "__main__":
    main()