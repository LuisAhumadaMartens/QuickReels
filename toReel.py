import tensorflow as tf
import numpy as np
import cv2

movenet = tf.saved_model.load('model')

def detect_pose(image):
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.cast(input_image, dtype=tf.int32)
    
    model = movenet.signatures['serving_default']
    outputs = model(input_image)
    
    keypoints = outputs['output_0'].numpy()
    return keypoints[0]

def draw_keypoints(frame, keypoints, confidence_threshold=0.4):
    y, x, c = frame.shape
    shaped_keypoints = np.reshape(keypoints, (17, 3))
    
    for kp in shaped_keypoints:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cx, cy = int(kx * x), int(ky * y)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
    
    return frame

def process_video(input_source=0):  
    cap = cv2.VideoCapture(input_source)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        img = tf.image.resize_with_pad(frame, 192, 192)
        
        keypoints = detect_pose(img)
        
        output_frame = draw_keypoints(frame, keypoints)
        
        cv2.imshow('MoveNet Pose Detection', output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video() 
