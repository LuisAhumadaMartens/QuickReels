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

def process_video(input_source=None, output_path="output_video.mp4", debug=False):
    global prev_gray_frame
    
    if input_source is None:
        input_source = find_default_input()
        if not input_source:
            print("Error: No input file specified and no 'input.*' file found.")
            return
    
    width, height, fps, total_frames = get_video_metadata(input_source)
    crop_width = int(height * ASPECT_RATIO)
    
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
            
        if debug:
            cv2.imwrite(f"debug_frame_{frame_count}.jpg", frame)
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = mse(prev_gray_frame, gray)
        prev_gray_frame = gray.copy()
        
        input_tensor = prepare_input_tensor(frame)
        outputs = movenet_func(input_tensor)
        keypoints = outputs['output_0'].numpy()[0]
        
        confidences = keypoints[:, 2]
        weights = confidences / np.sum(confidences)
        center_x = np.sum(keypoints[:, 1] * weights)
        
        x_center = int(center_x * width)
        x_start = x_center - (crop_width // 2)
        x_start = max(0, min(width - crop_width, x_start))
        x_end = x_start + crop_width
        
        cropped_frame = frame[:, x_start:x_end].copy()
        
        if debug:
            cropped_frame = draw_keypoints(cropped_frame, keypoints, x_start)
            if frame_diff > 3000: 
                cv2.putText(cropped_frame, "Scene Change", (crop_width//2 - 50, height//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        writer.write(cropped_frame)
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
