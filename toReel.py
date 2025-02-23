import tensorflow as tf
import numpy as np
import cv2
import argparse
import glob
import os
import sys

# -------------------------------
# Constants
# -------------------------------
DETECTION_CONFIDENCE_THRESHOLD = 0.3
MOVE_NET_INPUT_SIZE = (192, 192)
DEFAULT_CENTER = 0.5
ASPECT_RATIO = 9 / 16
SCENE_CHANGE_THRESHOLD = 3000  # Added from second file

# -------------------------------
# Global variable for scene change detection
# -------------------------------
prev_gray_frame = None

# -------------------------------
# Load MoveNet model
# -------------------------------
movenet = tf.saved_model.load('model')
movenet_func = movenet.signatures['serving_default']

# -------------------------------
# PersonTracker class
# -------------------------------
class PersonTracker:
    def __init__(self, smoothing_rate=0.05, confidence_margin=0.15, wait_frames=30, lock_frames=45, min_confidence=50):
        """
        Simple and reliable tracking with:
        - Smooth camera movement
        - Stable target switching
        - Scene change handling
        """
        self.smoothing_rate = smoothing_rate
        self.confidence_margin = confidence_margin
        self.wait_frames = wait_frames
        self.lock_frames = lock_frames
        self.min_confidence = min_confidence
        self.reset()

    def reset(self):
        self.current_id = None
        self.current_confidence = None
        self.pos_x = 0.5  # normalized center
        self.pos_y = 0.5
        self.new_target_id = None
        self.new_target_x = None
        self.new_target_y = None
        self.new_target_confidence = None
        self.new_target_frames = 0
        self.lock_countdown = 0
        self.lost_frames = 0
        self.first_frame_of_scene = False

    # ... copy rest of PersonTracker class methods from second file ...

# -------------------------------
# Utility: Preprocess frame for MoveNet
# -------------------------------
def prepare_input_tensor(frame):
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = tf.image.resize(input_tensor, MOVE_NET_INPUT_SIZE)
    input_tensor = tf.cast(input_tensor, dtype=tf.int32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    return input_tensor

# -------------------------------
# Utility: Mean Squared Error for scene change detection
# -------------------------------
def mse(imageA, imageB):
    """Compute Mean Squared Error between two images."""
    if imageA is None or imageB is None:
        return 0
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

def draw_keypoints(frame, keypoints, x_offset=0):
    """Draw keypoints on frame with confidence values."""
    y, x, c = frame.shape
    shaped_keypoints = np.reshape(keypoints, (17, 3))
    
    for kp in shaped_keypoints:
        ky, kx, kp_conf = kp
        if kp_conf > DETECTION_CONFIDENCE_THRESHOLD:
            cx, cy = int(kx * x) - x_offset, int(ky * y)
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
            conf_text = f"{int(kp_conf*100)}%"
            cv2.putText(frame, conf_text, (cx+10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

# -------------------------------
# Utility functions
# -------------------------------
def cluster_people(people, threshold=0.05):
    """
    Merge detections that are very close.
    Each detection in `people` is a tuple (id, kp) where kp = (y, x, confidence).
    Returns a list of clusters as tuples: (cluster_id, avg_x, avg_y, max_confidence).
    """
    # ... copy cluster_people implementation from second file ...

def process_video(input_source=None, output_path="output_video.mp4", debug=False):
    global prev_gray_frame
    
    if input_source is None:
        input_source = find_default_input()
        if not input_source:
            print("Error: No input file specified and no 'input.*' file found.")
            return
    
    width, height, fps, total_frames = get_video_metadata(input_source)
    crop_width = int(height * ASPECT_RATIO)
    
    # Initialize tracker
    tracker = PersonTracker(
        smoothing_rate=0.05,
        confidence_margin=0.15,
        wait_frames=int(fps * 1.0),  # 1 second buffer
        lock_frames=int(fps * 1.5)    # 1.5 second lock
    )
    
    cap = cv2.VideoCapture(input_source)
    writer = cv2.VideoWriter(output_path, 
                           cv2.VideoWriter_fourcc(*'avc1'),
                           fps, 
                           (crop_width, height))
    
    frame_count = 0
    print("Processing video...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame using the new tracker
        processed_frame, state_text = process_frame(
            frame, width, height, crop_width, height, 
            tracker, merge_distance=0.05, frame_count=frame_count, debug=debug
        )
        
        writer.write(processed_frame)
        frame_count += 1
        print(f"Progress: {(frame_count/total_frames)*100:.2f}%", end='\r')
    
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("\nProcessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video using MoveNet pose detection")
    parser.add_argument("-i", "--input", help="Path to input video file")
    parser.add_argument("-o", "--output", default="output_video.mp4", help="Path to output video file")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug visualization")
    args = parser.parse_args()
    
    process_video(args.input, args.output, args.debug) 
