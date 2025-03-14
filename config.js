// Direct configuration values from the original implementation
const PORT = 3000;
const DETECTION_THRESHOLD = 0.3;
const PERSON_CLASS_ID = 0;
const SCENE_CHANGE_THRESHOLD = 3000;  // Using the original threshold value
const DEFAULT_CENTER = 0.5;
const moveNetInputSize = 192;

// Video processing constants
const ASPECT_RATIO = 9 / 16;       // Output crop aspect ratio (portrait)
const BATCH_SIZE = 10;             // Batch size for frame processing

// Video encoding settings
const ENCODING = {
  PRESET: 'medium',
  CRF: 18,
  PIXEL_FORMAT: 'yuv420p'
};


// File paths
const TEMP_DIR_NAME = 'temp';
const FRAMES_DIR_NAME = 'frames';
const PROCESSING_DIR_NAME = 'processing';
const TEMP_INPUT_NAME = 'input.mp4';
const TEMP_OUTPUT_NAME = 'output.mp4';

const MOVE_NET_INPUT_SIZE = [moveNetInputSize, moveNetInputSize];

module.exports = {
  PORT,
  DETECTION_THRESHOLD,
  PERSON_CLASS_ID,
  SCENE_CHANGE_THRESHOLD,
  DEFAULT_CENTER,
  MOVE_NET_INPUT_SIZE,
  
  // Video processing
  ASPECT_RATIO,
  BATCH_SIZE,
  
  // Encoding
  ENCODING,
  
  // File paths
  TEMP_DIR_NAME,
  FRAMES_DIR_NAME,
  PROCESSING_DIR_NAME,
  TEMP_INPUT_NAME,
  TEMP_OUTPUT_NAME
}; 