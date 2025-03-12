#!/usr/bin/env node
// movenet-test.js - Test script specifically for Google's MoveNet model
// Based on https://www.kaggle.com/google/movenet

const path = require('path');
const fs = require('fs');
const os = require('os');
const tf = require('@tensorflow/tfjs-node');
const ffmpeg = require('fluent-ffmpeg');
const ffmpegPath = require('@ffmpeg-installer/ffmpeg').path;

// Configure ffmpeg
ffmpeg.setFfmpegPath(ffmpegPath);

// Global model variable
let model = null;
const MODEL_PATH = path.join(__dirname, 'model');

// Configuration options
const CONFIG = {
  inputSize: 192,            // MoveNet Lightning uses 192x192 input
  keypointThreshold: 0.3,    // Minimum confidence threshold for keypoints
  frameRate: 5,              // Number of frames per second to analyze
  debug: true,               // Enable debug visualization
  visualizationColors: {
    keypoint: 'green@0.8',   // Color for keypoint visualization
    connection: 'red@0.5'    // Color for skeleton connection lines
  }
};

// MoveNet keypoints mapping
const KEYPOINTS = [
  "nose", "left_eye", "right_eye", "left_ear", "right_ear",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle"
];

// Pairs of keypoints that form skeleton connections
const SKELETON_CONNECTIONS = [
  [0, 1], [0, 2], [1, 3], [2, 4],       // Face
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], // Upper body
  [5, 11], [6, 12], [11, 12],           // Torso
  [11, 13], [13, 15], [12, 14], [14, 16] // Lower body
];

/**
 * Initialize TensorFlow and load the MoveNet model
 */
async function initializeTensorFlow() {
  console.log('Initializing TensorFlow.js and loading MoveNet model...');
  
  try {
    // Check if model directory exists
    if (!fs.existsSync(MODEL_PATH)) {
      throw new Error(`Model directory not found: ${MODEL_PATH}`);
    }
    
    // Look for model files
    const modelJsonPath = path.join(MODEL_PATH, 'model.json');
    const savedModelPath = path.join(MODEL_PATH, 'saved_model.pb');
    
    // Load the model - use appropriate method based on available files
    console.log(`Loading model from: ${MODEL_PATH}`);
    
    if (fs.existsSync(modelJsonPath)) {
      // Load using loadGraphModel if model.json exists (typical for converted models)
      model = await tf.loadGraphModel(`file://${modelJsonPath}`);
      console.log('Loaded MoveNet model from model.json');
    } else if (fs.existsSync(savedModelPath)) {
      // If saved_model.pb exists (typical for TensorFlow Hub models)
      console.log('Found saved_model.pb format');
      model = await tf.node.loadSavedModel(MODEL_PATH);
      console.log('Loaded MoveNet from SavedModel directory');
    } else {
      throw new Error('No compatible MoveNet model files found in the model directory');
    }
    
    console.log('MoveNet model loaded successfully');
    return true;
  } catch (error) {
    console.error('Error initializing TensorFlow or loading MoveNet:', error);
    return false;
  }
}

/**
 * Prepare input tensor from image for MoveNet model
 */
async function prepareInputForMoveNet(imageBuffer) {
  try {
    // Decode image to tensor (3 channels for RGB)
    const imageTensor = tf.node.decodeImage(imageBuffer, 3);
    
    // Create properly formatted input tensor (tidy for auto cleanup)
    const inputTensor = tf.tidy(() => {
      // Resize to MoveNet input size
      const resized = tf.image.resizeBilinear(imageTensor, [CONFIG.inputSize, CONFIG.inputSize]);
      
      // Normalize to [-1, 1]
      const normalized = tf.div(tf.sub(tf.cast(resized, 'float32'), 127.5), 127.5);
      
      // Add batch dimension
      return tf.expandDims(normalized, 0);
    });
    
    // Clean up original tensor
    imageTensor.dispose();
    
    return inputTensor;
  } catch (error) {
    console.error('Error preparing input for MoveNet:', error);
    throw error;
  }
}

/**
 * Process a frame with MoveNet model
 */
async function detectPoseInFrame(frameBuffer) {
  try {
    // Prepare input tensor
    const inputTensor = await prepareInputForMoveNet(frameBuffer);
    
    // Run inference
    let result;
    try {
      result = await model.predict(inputTensor);
    } catch (error) {
      console.error('MoveNet inference failed:', error);
      inputTensor.dispose();
      return null;
    }
    
    // Get keypoints from the output tensor
    let keypoints = [];
    try {
      // Process output format depending on model
      if (result.output_0) {
        // For SavedModel format
        keypoints = result.output_0.arraySync()[0];
      } else if (result.shape && result.shape.length === 3) {
        // For GraphModel format
        keypoints = result.arraySync()[0];
      } else {
        console.warn('Unknown MoveNet output format:', result);
      }
    } catch (error) {
      console.error('Error extracting keypoints from MoveNet output:', error);
    }
    
    // Clean up tensors
    inputTensor.dispose();
    if (result) {
      if (typeof result === 'object' && result.output_0) {
        result.output_0.dispose();
      } else if (result.dispose) {
        result.dispose();
      }
    }
    
    // Filter keypoints by confidence threshold
    const validKeypoints = keypoints.filter((kp, i) => 
      kp && kp.length >= 3 && kp[2] > CONFIG.keypointThreshold
    ).map((kp, i) => ({
      position: { y: kp[0], x: kp[1] },  // y, x as in MoveNet output
      score: kp[2],                       // Confidence score
      name: KEYPOINTS[i] || `keypoint_${i}`  // Named keypoint
    }));
    
    // Calculate pose center from valid keypoints
    let poseCenter = { x: 0.5, y: 0.5 };
    if (validKeypoints.length > 0) {
      const sumX = validKeypoints.reduce((sum, kp) => sum + kp.position.x, 0);
      const sumY = validKeypoints.reduce((sum, kp) => sum + kp.position.y, 0);
      poseCenter = {
        x: sumX / validKeypoints.length,
        y: sumY / validKeypoints.length
      };
    }
    
    return {
      center: poseCenter,
      keypoints: validKeypoints,
      score: validKeypoints.length > 0 
        ? validKeypoints.reduce((sum, kp) => sum + kp.score, 0) / validKeypoints.length
        : 0
    };
  } catch (error) {
    console.error('Error in pose detection:', error);
    return null;
  }
}

/**
 * Generate FFmpeg visualization commands for pose keypoints
 */
function generateVisualizationCommands(pose, timestamp, videoWidth, videoHeight) {
  const commands = [];
  
  if (!pose || !pose.keypoints || pose.keypoints.length === 0) {
    return commands;
  }
  
  // Draw each keypoint
  pose.keypoints.forEach(keypoint => {
    const x = Math.floor(keypoint.position.x * videoWidth);
    const y = Math.floor(keypoint.position.y * videoHeight);
    
    // Draw keypoint dot
    commands.push(`${timestamp} drawbox x ${x-5} y ${y-5} w 10 h 10 color ${CONFIG.visualizationColors.keypoint} t fill`);
    
    // Draw confidence score
    const scoreText = `${Math.floor(keypoint.score * 100)}%`;
    commands.push(`${timestamp} drawbox x ${x+8} y ${y-15} w 40 h 15 color black@0.5 t fill`);
  });
  
  // Draw skeleton connections
  SKELETON_CONNECTIONS.forEach(([i, j]) => {
    const kp1 = pose.keypoints.find(kp => kp.name === KEYPOINTS[i]);
    const kp2 = pose.keypoints.find(kp => kp.name === KEYPOINTS[j]);
    
    if (kp1 && kp2) {
      const x1 = Math.floor(kp1.position.x * videoWidth);
      const y1 = Math.floor(kp1.position.y * videoHeight);
      const x2 = Math.floor(kp2.position.x * videoWidth);
      const y2 = Math.floor(kp2.position.y * videoHeight);
      
      // Draw line between keypoints (FFmpeg line syntax)
      commands.push(`${timestamp} drawbox x ${Math.min(x1, x2)} y ${Math.min(y1, y2)} w ${Math.abs(x2-x1) || 1} h ${Math.abs(y2-y1) || 1} color ${CONFIG.visualizationColors.connection} t fill`);
    }
  });
  
  // Draw center of pose
  const centerX = Math.floor(pose.center.x * videoWidth);
  const centerY = Math.floor(pose.center.y * videoHeight);
  commands.push(`${timestamp} drawbox x ${centerX-8} y ${centerY-8} w 16 h 16 color yellow@0.8 t fill`);
  
  return commands;
}

/**
 * Process video with MoveNet pose detection
 */
async function processVideoWithMoveNet(inputPath, outputPath) {
  console.log(`Processing video with MoveNet: ${inputPath} -> ${outputPath}`);
  
  // Create temporary directory for frames
  const tempDir = path.join(os.tmpdir(), `movenet-${Date.now()}`);
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }
  
  try {
    // Get video metadata
    const metadata = await new Promise((resolve, reject) => {
      ffmpeg.ffprobe(inputPath, (err, metadata) => {
        if (err) return reject(err);
        
        const videoStream = metadata.streams.find(s => s.codec_type === 'video');
        if (!videoStream) return reject(new Error('No video stream found'));
        
        // Extract relevant metadata
        const fps = eval(videoStream.r_frame_rate || '30');
        const duration = parseFloat(metadata.format.duration);
        const totalFrames = Math.round(duration * fps);
        
        resolve({
          width: videoStream.width,
          height: videoStream.height,
          fps,
          duration,
          totalFrames
        });
      });
    });
    
    console.log('Video metadata:', metadata);
    
    // Extract frames at specified frame rate
    const frameInterval = Math.max(1, Math.floor(metadata.fps / CONFIG.frameRate));
    console.log(`Extracting frames at ${CONFIG.frameRate} fps (interval: ${frameInterval})`);
    
    await new Promise((resolve, reject) => {
      ffmpeg(inputPath)
        .outputOptions([
          '-vf', `fps=${metadata.fps/frameInterval}`,
          '-q:v', '2'
        ])
        .on('start', (cmd) => {
          console.log('Extracting frames with command:', cmd);
        })
        .on('end', () => {
          console.log('Frame extraction complete');
          resolve();
        })
        .on('error', (err) => {
          console.error('Frame extraction error:', err);
          reject(err);
        })
        .output(path.join(tempDir, 'frame-%04d.jpg'))
        .run();
    });
    
    // Find all extracted frames
    const frameFiles = fs.readdirSync(tempDir)
      .filter(file => file.endsWith('.jpg'))
      .sort();
    
    console.log(`Found ${frameFiles.length} frames to process`);
    
    // Process each frame with MoveNet
    const visualizationCommands = [];
    const poseData = [];
    
    // Process frames
    for (let i = 0; i < frameFiles.length; i++) {
      const frameFile = frameFiles[i];
      const framePath = path.join(tempDir, frameFile);
      const frameIndex = i;
      
      console.log(`Processing frame ${i+1}/${frameFiles.length}: ${frameFile}`);
      
      try {
        // Read the frame
        const frameBuffer = fs.readFileSync(framePath);
        
        // Detect pose in frame
        const pose = await detectPoseInFrame(frameBuffer);
        
        // Store pose data
        poseData.push({
          frame: frameIndex,
          timestamp: frameIndex / (metadata.fps / frameInterval),
          pose: pose
        });
        
        // Generate visualization if debug mode is enabled
        if (CONFIG.debug && pose) {
          const timestamp = frameIndex / (metadata.fps / frameInterval);
          const commands = generateVisualizationCommands(
            pose, 
            timestamp, 
            metadata.width, 
            metadata.height
          );
          visualizationCommands.push(...commands);
          
          console.log(`  Found ${pose.keypoints.length} keypoints in frame ${i+1}`);
        }
      } catch (error) {
        console.error(`Error processing frame ${frameFile}:`, error);
      }
    }
    
    // Create filter script for FFmpeg
    let filterScript = '';
    if (visualizationCommands.length > 0) {
      console.log(`Generated ${visualizationCommands.length} visualization commands`);
      filterScript = `sendcmd=c='${visualizationCommands.join('\n').replace(/'/g, "\''")}',`;
      filterScript += `drawbox=x=0:y=0:w=10:h=10:color=red@0:t=fill`; // Initial invisible box
    } else {
      console.warn('No visualization commands generated');
    }
    
    // Process the video with visualization
    console.log('Creating final video with pose visualization...');
    await new Promise((resolve, reject) => {
      const ffmpegCommand = ffmpeg(inputPath);
      
      if (filterScript) {
        ffmpegCommand.videoFilter(filterScript);
      }
      
      ffmpegCommand
        .outputOptions([
          '-c:v', 'libx264',
          '-pix_fmt', 'yuv420p',
          '-preset', 'fast',
          '-crf', '23',
          '-c:a', 'copy'  // Copy audio stream
        ])
        .output(outputPath)
        .on('start', (cmd) => {
          console.log('FFmpeg command:', cmd);
        })
        .on('progress', (progress) => {
          console.log(`Processing video: ${Math.floor(progress.percent || 0)}% done`);
        })
        .on('end', () => {
          console.log('Video processing complete');
          resolve();
        })
        .on('error', (err) => {
          console.error('FFmpeg error:', err);
          reject(err);
        })
        .run();
    });
    
    // Save pose data to JSON file (for reference)
    const jsonOutputPath = outputPath.replace(/\.[^.]+$/, '.json');
    fs.writeFileSync(jsonOutputPath, JSON.stringify({
      metadata: {
        inputVideo: inputPath,
        width: metadata.width,
        height: metadata.height,
        fps: metadata.fps,
        duration: metadata.duration,
        processedFrames: frameFiles.length
      },
      poseData: poseData
    }, null, 2));
    
    console.log(`Saved pose data to ${jsonOutputPath}`);
    
    // Clean up temp directory
    try {
      fs.rmSync(tempDir, { recursive: true, force: true });
    } catch (error) {
      console.warn('Failed to remove temporary directory:', error);
    }
    
    console.log(`✅ Successfully processed video with MoveNet: ${outputPath}`);
    return {
      status: 'success',
      inputPath,
      outputPath,
      jsonOutputPath
    };
  } catch (error) {
    console.error('Error processing video with MoveNet:', error);
    
    // Clean up temp directory on error
    try {
      fs.rmSync(tempDir, { recursive: true, force: true });
    } catch (cleanupError) {
      console.warn('Failed to remove temporary directory:', cleanupError);
    }
    
    throw error;
  }
}

/**
 * Print usage information
 */
function printUsage() {
  console.log('MoveNet Video Pose Detection Tool');
  console.log('Usage: node movenet-test.js <input-video> <output-video>');
  console.log('');
  console.log('This tool will process a video with Google\'s MoveNet pose detection model');
  console.log('and visualize the detected poses with keypoints and skeleton connections.');
  console.log('');
  console.log('Example:');
  console.log('  node movenet-test.js input.mp4 output-with-poses.mp4');
}

/**
 * Main function
 */
async function main() {
  // Parse command line arguments
  const args = process.argv.slice(2);
  
  if (args.length < 2) {
    printUsage();
    process.exit(1);
  }
  
  const inputVideo = args[0];
  const outputVideo = args[1];
  
  // Check if input file exists
  if (!fs.existsSync(inputVideo)) {
    console.error(`Error: Input file does not exist: ${inputVideo}`);
    process.exit(1);
  }
  
  console.log(`Input: ${inputVideo}`);
  console.log(`Output: ${outputVideo}`);
  
  try {
    // Initialize TensorFlow and load MoveNet model
    const modelLoaded = await initializeTensorFlow();
    if (!modelLoaded) {
      console.error('Error: Failed to load MoveNet model');
      process.exit(1);
    }
    
    // Process the video with MoveNet
    await processVideoWithMoveNet(inputVideo, outputVideo);
    
    console.log('✅ Processing complete!');
    console.log(`Output saved to: ${outputVideo}`);
  } catch (error) {
    console.error('❌ Error processing video:', error);
    process.exit(1);
  }
}

// Run the main function
if (require.main === module) {
  main().catch(error => {
    console.error('Unhandled error:', error);
    process.exit(1);
  });
}

module.exports = {
  processVideoWithMoveNet,
  initializeTensorFlow,
  detectPoseInFrame
}; 