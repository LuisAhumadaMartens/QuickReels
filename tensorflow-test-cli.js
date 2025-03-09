#!/usr/bin/env node
// tensorflow-test-cli.js - Command-line tool to visualize TensorFlow detections

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

// Detection thresholds
const DETECTION_THRESHOLD = 0.3;
const PERSON_CLASS_ID = 0; // In COCO dataset, person is class 0

/**
 * Initialize TensorFlow and load the MoveNet model
 */
async function initializeTensorFlow() {
  console.log('Initializing TensorFlow.js...');
  
  try {
    // Check if model directory exists
    if (!fs.existsSync(MODEL_PATH)) {
      throw new Error(`Model directory not found: ${MODEL_PATH}`);
    }
    
    // Look for model files
    const modelJsonPath = path.join(MODEL_PATH, 'model.json');
    const savedModelPath = path.join(MODEL_PATH, 'saved_model.pb');
    
    console.log(`Loading model from: ${MODEL_PATH}`);
    
    // First try loading as GraphModel (model.json)
    if (fs.existsSync(modelJsonPath)) {
      console.log('Found model.json, loading as GraphModel');
      try {
        model = await tf.loadGraphModel(`file://${modelJsonPath}`);
        console.log('Successfully loaded model from model.json');
        return true;
      } catch (err) {
        console.warn('Failed to load as GraphModel:', err.message);
      }
    }
    
    // Then try loading as SavedModel
    if (fs.existsSync(savedModelPath)) {
      console.log('Found saved_model.pb, loading as SavedModel');
      try {
        model = await tf.node.loadSavedModel(MODEL_PATH);
        
        // For SavedModel, we need to warm up the model
        console.log('Warming up the model with a test tensor...');
        const dummyTensor = tf.ones([1, 192, 192, 3]); // Create a dummy tensor of the right shape
        const result = await model.predict(dummyTensor);
        
        // Log the model output structure to understand it better
        if (result && result.output_0) {
          console.log('Model output shape:', result.output_0.shape);
        } else if (result && result.shape) {
          console.log('Model output shape:', result.shape);
        }
        
        // Clean up
        tf.dispose([dummyTensor, result]);
        
        console.log('Successfully loaded model from SavedModel');
        return true;
      } catch (err) {
        console.error('Failed to load as SavedModel:', err);
        throw err;
      }
    }
    
    throw new Error('No compatible model files found in the specified directory');
  } catch (error) {
    console.error('Error initializing TensorFlow:', error);
    return false;
  }
}

/**
 * Prepare input tensor from image buffer for MoveNet model
 * This matches the Python implementation in toReel.py
 */
async function prepareInputTensor(frameBuffer) {
  try {
    // Decode image to tensor (3 channels for RGB)
    const imageTensor = tf.node.decodeImage(frameBuffer, 3);
    
    // Log original shape
    console.log(`Original image shape: ${imageTensor.shape}`);
    
    // Resize to MoveNet input size (192x192) - matches Python's cv2.resize
    const inputSize = 192;
    const resized = tf.image.resizeBilinear(imageTensor, [inputSize, inputSize]);
    
    // Convert to int32 (matches Python's tf.convert_to_tensor(..., dtype=tf.int32))
    const convertedTensor = tf.cast(resized, 'int32');
    
    // Add batch dimension (matches Python's tf.expand_dims(input_tensor, axis=0))
    const batchedTensor = tf.expandDims(convertedTensor, 0);
    
    // Log final tensor shape
    console.log(`Prepared tensor shape: ${batchedTensor.shape}`);
    
    // Clean up intermediate tensors
    tf.dispose([imageTensor, resized, convertedTensor]);
    
    return batchedTensor;
  } catch (error) {
    console.error('Error preparing input tensor:', error);
    throw error;
  }
}

/**
 * Run inference on a frame tensor using MoveNet model
 * Adapted for TensorFlow.js limitations
 */
async function runInference(tensor) {
  if (!model) {
    console.warn('Model not loaded, using default detection');
    return [{ 
      x: 0.5,
      y: 0.5,
      width: 0.2,
      height: 0.5,
      confidence: 0.9,
      class: PERSON_CLASS_ID,
      keypoints: Array(17).fill().map(() => [0.5, 0.5, 0.1]) // Create empty keypoints
    }];
  }

  try {
    // Process the frame through the model
    console.log('Running MoveNet inference...');
    
    // In TensorFlow.js, we need to use predict() instead of execute()
    // This is different from Python, which uses signatures['serving_default']
    let result = await model.predict(tensor);
    
    // Extract keypoints from the result
    let keypoints = [];
    if (result.output_0) {
      // SavedModel format
      console.log('Found output_0 tensor:', result.output_0.shape);
      keypoints = result.output_0.arraySync()[0];
    } else if (result.arraySync) {
      // GraphModel format
      console.log('Found tensor with shape:', result.shape);
      keypoints = result.arraySync()[0];
    } else {
      console.warn('Unknown result format:', result);
      return [];
    }
    
    // Clean up the tensors
    if (result.output_0 && result.output_0.dispose) {
      result.output_0.dispose();
    } else if (result.dispose) {
      result.dispose();
    }
    
    // Filter keypoints by confidence
    const detections = [];
    
    // Process keypoints
    if (keypoints && keypoints.length > 0) {
      // Create a detection object from all keypoints
      const validKeypoints = keypoints.filter(kp => kp && kp.length >= 3 && kp[2] > DETECTION_THRESHOLD);
      
      if (validKeypoints.length > 0) {
        // Calculate average position from valid keypoints
        const sumX = validKeypoints.reduce((sum, kp) => sum + kp[1], 0); // x coordinate
        const sumY = validKeypoints.reduce((sum, kp) => sum + kp[0], 0); // y coordinate
        const avgX = sumX / validKeypoints.length;
        const avgY = sumY / validKeypoints.length;
        
        // Calculate average confidence
        const avgConfidence = validKeypoints.reduce((sum, kp) => sum + kp[2], 0) / validKeypoints.length;
        
        // Create a detection object with the keypoints
        const detection = {
          x: avgX,
          y: avgY,
          width: 0.3,
          height: 0.7,
          confidence: avgConfidence,
          class: PERSON_CLASS_ID,
          keypoints: keypoints
        };
        
        console.log(`Found person with ${validKeypoints.length} valid keypoints, confidence ${avgConfidence.toFixed(2)}`);
        detections.push(detection);
      }
    }
    
    return detections;
  } catch (error) {
    console.error('Error in runInference:', error);
    return [];
  }
}

/**
 * Process a video to show TensorFlow detections without cropping
 */
async function processVideoWithTensorFlow(inputPath, outputPath, debug = true) {
  console.log(`Processing video with TensorFlow: ${inputPath} -> ${outputPath}`);
  
  // Create temporary directory for frames
  const tempDir = path.join(os.tmpdir(), `tf-test-${Date.now()}`);
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
    
    // Extract frames at a lower frame rate for analysis
    // Reduce frames-per-second to avoid generating too many commands
    const targetFps = 2; // Only process 2 frames per second (instead of 5)
    const frameInterval = Math.max(1, Math.floor(metadata.fps / targetFps));
    console.log(`Extracting frames at interval ${frameInterval} (${metadata.fps/frameInterval} fps)`);
    
    await new Promise((resolve, reject) => {
      ffmpeg(inputPath)
        .outputOptions([
          '-vf', `fps=${metadata.fps/frameInterval}`,
          '-q:v', '2'
        ])
        .on('start', (cmd) => {
          console.log('FFmpeg frame extraction command:', cmd);
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
    
    console.log(`Found ${frameFiles.length} frames`);
    
    // If we have too many frames, sample them to stay within FFmpeg's command limit
    // FFmpeg has limits on command buffer size
    const MAX_FRAMES_TO_PROCESS = 300; // Limit the number of frames to process
    
    let processFrames = frameFiles;
    if (frameFiles.length > MAX_FRAMES_TO_PROCESS) {
      console.log(`Too many frames (${frameFiles.length}), sampling to ${MAX_FRAMES_TO_PROCESS}...`);
      const samplingInterval = Math.ceil(frameFiles.length / MAX_FRAMES_TO_PROCESS);
      processFrames = frameFiles.filter((_, index) => index % samplingInterval === 0);
      console.log(`Sampled to ${processFrames.length} frames with interval ${samplingInterval}`);
    }
    
    // Process each frame with TensorFlow and generate visualization commands
    const drawCommands = [];
    
    // Variables for frame difference calculation (for scene change detection)
    let prevGrayFrame = null;
    
    for (let i = 0; i < processFrames.length; i++) {
      const frameFile = processFrames[i];
      const framePath = path.join(tempDir, frameFile);
      
      console.log(`Processing frame ${i+1}/${processFrames.length}: ${frameFile}`);
      
      try {
        // Read the frame
        const frameBuffer = fs.readFileSync(framePath);
        const frame = tf.node.decodeImage(frameBuffer, 3);
        
        // Calculate frame difference for scene change detection
        // In Python: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        // We'll compute a simplified grayscale and MSE here
        let frameDiff = 0;
        const grayscale = tf.tidy(() => {
          // Convert to grayscale: (r + g + b) / 3
          return frame.mean(2);
        });
        
        if (prevGrayFrame !== null) {
          // Calculate MSE (mean squared error)
          frameDiff = tf.tidy(() => {
            const diff = tf.sub(grayscale, prevGrayFrame);
            const squaredDiff = tf.square(diff);
            return squaredDiff.mean().arraySync();
          });
          console.log(`Frame difference: ${frameDiff}`);
        }
        
        // Store current frame for next comparison
        if (prevGrayFrame) prevGrayFrame.dispose();
        prevGrayFrame = grayscale;
        
        // Prepare input and run inference (like Python's process_frame function)
        const inputTensor = await prepareInputTensor(frameBuffer);
        const detections = await runInference(inputTensor);
        
        // Clean up tensors
        tf.dispose([frame, inputTensor]);
        
        console.log(`Found ${detections ? detections.length : 0} detections in frame`);
        
        // Generate visualization commands for this frame
        const frameIndex = i;
        const timestamp = frameIndex / (metadata.fps / frameInterval);
        
        // Add visualization for each detection - SIMPLIFIED TO JUST ONE DOT
        if (detections && detections.length > 0) {
          detections.forEach((detection, idx) => {
            console.log(`  Detection ${idx+1}: confidence=${detection.confidence || 'unknown'}, keypoints=${detection.keypoints ? detection.keypoints.length : 0}`);
            
            // Just draw one center point for the detected person
            // This is much simpler and avoids overwhelming FFmpeg
            const xCenter = Math.floor(detection.x * metadata.width);
            const yCenter = Math.floor(detection.y * metadata.height);
            
            // Draw a single dot at the center - make it larger and more visible
            drawCommands.push(`${timestamp} drawbox x ${xCenter-10} y ${yCenter-10} w 20 h 20 color yellow@0.8 t fill`);
            
            // Add a confidence text if desired
            const confPercent = Math.floor((detection.confidence || 0.5) * 100);
            // drawCommands.push(`${timestamp} drawbox x ${xCenter+15} y ${yCenter-15} w 50 h 20 color black@0.5 t fill`);
          });
        }
      } catch (error) {
        console.error(`Error processing frame ${frameFile}:`, error);
      }
    }
    
    // Clean up
    if (prevGrayFrame) prevGrayFrame.dispose();
    
    // Create filter script for ffmpeg
    let filterScript = '';
    if (drawCommands.length > 0) {
      console.log(`Generated ${drawCommands.length} visualization commands`);
      
      // Split commands into chunks if we have too many
      const MAX_COMMANDS_PER_FILE = 1000;
      
      if (drawCommands.length > MAX_COMMANDS_PER_FILE) {
        console.log(`Too many commands (${drawCommands.length}), using simpler approach...`);
        
        // Use a simpler approach for very long videos
        // Just track position using a sliding effect
        let lastCommand = null;
        const simplifiedCommands = [];
        
        // Take one command every few seconds
        for (let i = 0; i < drawCommands.length; i += 20) {
          if (drawCommands[i]) {
            simplifiedCommands.push(drawCommands[i]);
            lastCommand = drawCommands[i];
          }
        }
        
        console.log(`Simplified to ${simplifiedCommands.length} commands`);
        filterScript = `sendcmd=c='${simplifiedCommands.join('\n').replace(/'/g, "\''")}',` +
                       `drawbox=x=0:y=0:w=10:h=10:color=red@0:t=fill`; // Initial invisible box
      } else {
        // Normal approach for shorter videos
        filterScript = `sendcmd=c='${drawCommands.join('\n').replace(/'/g, "\''")}',` +
                       `drawbox=x=0:y=0:w=10:h=10:color=red@0:t=fill`; // Initial invisible box
      }
    } else {
      console.warn('No visualization commands generated');
    }
    
    // Apply the visualizations to the video
    console.log('Creating final video with visualization...');
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
          console.log('FFmpeg command:', cmd.substring(0, 1000) + '... [truncated]');
        })
        .on('progress', (progress) => {
          console.log(`Processing: ${Math.floor(progress.percent || 0)}% done`);
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
    
    // Clean up temp directory
    try {
      fs.rmSync(tempDir, { recursive: true, force: true });
    } catch (error) {
      console.warn('Failed to remove temporary directory:', error);
    }
    
    console.log(`Successfully processed video: ${outputPath}`);
    return {
      status: 'success',
      inputPath,
      outputPath
    };
  } catch (error) {
    console.error('Error processing video:', error);
    
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
  console.log('TensorFlow Video Visualization Tool');
  console.log('Usage: node tensorflow-test-cli.js <input-video> <output-video>');
  console.log('');
  console.log('This tool will process a video with TensorFlow and visualize the detections');
  console.log('It will NOT crop the video, but will show bounding boxes and keypoints');
  console.log('');
  console.log('Example:');
  console.log('  node tensorflow-test-cli.js input.mp4 output-with-tf.mp4');
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
    // Initialize TensorFlow
    const modelLoaded = await initializeTensorFlow();
    if (!modelLoaded) {
      console.warn('Warning: TensorFlow model could not be loaded, using mock data');
    }
    
    // Process the video
    await processVideoWithTensorFlow(inputVideo, outputVideo);
    
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
  processVideoWithTensorFlow,
  initializeTensorFlow
}; 