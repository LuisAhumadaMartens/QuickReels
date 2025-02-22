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
