// Direct configuration values instead of reading from package.json
const PORT = 3000;
const DETECTION_THRESHOLD = 0.3;
const PERSON_CLASS_ID = 0;
const SCENE_CHANGE_THRESHOLD = 3000;
const DEFAULT_CENTER = 0.5;
const moveNetInputSize = 192;

// Video processing constants
const ASPECT_RATIO = 9 / 16;       // Output crop aspect ratio (portrait)
const BATCH_SIZE = 10;             // Batch size for frame processing

// Video encoding settings
const ENCODING = {
  PRESET: 'medium',         // FFmpeg preset (balance between speed and quality)
  CRF: 18,                  // Constant Rate Factor (lower = higher quality, 18 is "visually lossless")
  PIXEL_FORMAT: 'yuv420p'   // Standard pixel format for maximum compatibility
};

// Progress tracking settings
const PROGRESS = {
  UPDATE_INTERVAL_MS: 1000, // How often to update progress (milliseconds)
  PROCESSING_PHASES: {      // Processing phases with weights 
    // Processing now only tracks frame cropping
    FRAME_CROPPING: { weight: 1.0, description: "Processing frames" },
    
    // Encoding is tracked separately in the UI
    
    // TODO: These phases are deprecated and will be addressed later
    PREPARATION: { weight: 0, description: "Preparing for processing", deprecated: true },
    AUDIO_MERGING: { weight: 0, description: "Adding audio", deprecated: true }
  }
};

// ID generation
const DEFAULT_ID_LENGTH = 10;      // Default length for random job IDs

// File paths
const PROGRESS_FILE = 'progress.json';
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
  
  // Progress tracking
  PROGRESS,
  
  // ID generation
  DEFAULT_ID_LENGTH,
  
  // File paths
  PROGRESS_FILE,
  TEMP_DIR_NAME,
  FRAMES_DIR_NAME,
  PROCESSING_DIR_NAME,
  TEMP_INPUT_NAME,
  TEMP_OUTPUT_NAME
}; 