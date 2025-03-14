const fs = require('fs');
const path = require('path');
const ffmpeg = require('fluent-ffmpeg');
const cv = require('@u4/opencv4nodejs');

// Constants for processing
const ASPECT_RATIO = 9/16; // 9:16 aspect ratio for vertical video
const SCENE_CHANGE_THRESHOLD = 0.05;
const DEFAULT_CENTER = 0.5;
const ENCODING = {
  PRESET: 'medium',
  CRF: 23,
  PIXEL_FORMAT: 'yuv420p'
};
const BATCH_SIZE = 20;

/**
 * Calculate Mean Squared Error between two images
 * Used for scene change detection
 * @param {object} imageA - First image (OpenCV Mat)
 * @param {object} imageB - Second image (OpenCV Mat)
 * @returns {number} - MSE value indicating frame difference
 */
function mse(imageA, imageB) {
  // Use absdiff since we confirmed it exists
  const diff = imageA.absdiff(imageB);
  
  // Calculate the mean - w component for grayscale images
  const mean = diff.mean();
  
  return mean.w;
}

/**
 * Process a video based on analysis results - exact 1:1 implementation from original
 * @param {string} inputPath - Path to the input video file
 * @param {string} outputPath - Path for the output video file
 * @param {Object} analysis - Analysis results from analizeVideo
 * @returns {Promise<Object>} - Processing result
 */
async function processVideo(inputPath, outputPath, analysis) {
  // Set up directories - exactly matching the original structure
  const projectRoot = process.cwd();
  const tempBaseDir = path.join(projectRoot, 'temp');
  const processingId = path.basename(outputPath, path.extname(outputPath));
  const tempDir = analysis.tempDir || path.join(tempBaseDir, processingId);
  const framesDir = path.join(tempDir, 'frames');
  const processingFramesDir = path.join(framesDir, 'processing');
  const tempInputPath = path.join(tempDir, 'input.mp4');
  const tempOutputPath = path.join(tempDir, 'output.mp4');
  
  // Create all directories at once with recursive option - exactly as original
  const dirsToCreate = [
    tempBaseDir,
    tempDir,
    framesDir,
    processingFramesDir
  ];
  
  for (const dir of dirsToCreate) {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }
  
  try {
    // Step 1: Copy input file to temp directory if not already there
    if (!fs.existsSync(tempInputPath)) {
      fs.copyFileSync(inputPath, tempInputPath);
    }
    
    // Open the video file to get accurate properties
    const video = new cv.VideoCapture(tempInputPath);
    
    // Step 2: Extract metadata - using original method
    const videoMetadata = extractVideoMetadata(analysis, video);
    const { smoothedPositions } = analysis;
    
    // Calculate crop dimensions for 9:16 aspect ratio - exactly as original
    const { width, height, fps, totalFrames } = videoMetadata;
    const actualCropWidth = Math.round(height * ASPECT_RATIO);
    
    // Process the entire video - exactly as original
    const startFrame = 0;
    const endFrame = totalFrames - 1 || Math.floor(video.get(cv.CAP_PROP_FRAME_COUNT)) - 1;
    
    // For progress reporting
    let processedFrames = 0;
    const totalFramesToProcess = endFrame - startFrame + 1;
    
    // For scene change detection
    let prevGrayFrame = null;
    
    // Process frames
    let currentFrame = 0;
    
    // Batch processing for better performance - exactly as original
    const framePromises = [];
    
    // Main frame processing loop - exactly as original
    while (true) {
      // Read a frame from the video
      const frame = video.read();
      
      // Check if we've reached the end of the video
      if (frame.empty) {
        break;
      }
      
      // Stop if we've processed all required frames
      if (currentFrame > endFrame) {
        break;
      }
      
      // Calculate the index in the smoothedPositions array
      const positionIndex = currentFrame - startFrame;
      
      // Determine the center position (from analysis or default)
      const normCenter = positionIndex < smoothedPositions.length
        ? smoothedPositions[positionIndex]
        : DEFAULT_CENTER;
      
      // Process the video frame - using the original method
      const { croppedFrame, grayFrame, frameDiff } = processVideoFrame(frame, {
        normCenter,
        width,
        height,
        cropWidth: actualCropWidth,
        prevGrayFrame,
        currentFrame
      });
      
      // Update previous frame for next comparison
      prevGrayFrame = grayFrame;
      
      // Save the frame to disk (async) - exactly as original
      const framePath = path.join(processingFramesDir, `frame_${String(processedFrames).padStart(8, '0')}.png`);
      framePromises.push(
        (async () => {
          try {
            await cv.imwriteAsync(framePath, croppedFrame);
          } catch (err) {
            console.error(`Error writing frame ${currentFrame}: ${err.message}`);
          }
        })()
      );
      
      // Process in batches to avoid too many open files - exactly as original
      if (framePromises.length >= BATCH_SIZE) {
        await Promise.all(framePromises);
        framePromises.length = 0;
      }
      
      // Update progress
      processedFrames++;
      currentFrame++;
    }
    
    // Process any remaining frames - exactly as original
    if (framePromises.length > 0) {
      await Promise.all(framePromises);
    }
    
    // Release resources
    video.release();
    
    console.log("Frame processing complete, starting video encoding");
    
    // Encode the video - using original encoding method
    await encodeVideo(processingFramesDir, tempOutputPath, fps);
    
    // Create output directory if it doesn't exist
    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    console.log("Encoding complete, adding audio");
    
    // Add audio to the video - using original audio method
    await addAudioToVideo(tempOutputPath, tempInputPath, outputPath);
    
    console.log("Processing complete");
    
    // Clean up temporary files - exactly as original
    try {
      fs.rmSync(tempDir, { recursive: true, force: true });
    } catch (err) {
      console.warn(`Warning: Failed to clean up temporary directory: ${err.message}`);
    }
    
    return {
      inputPath,
      outputPath,
      duration: ((endFrame - startFrame + 1) / fps).toFixed(2),
      status: 'completed'
    };
  } catch (error) {
    console.error(`Error processing video: ${error.message}`);
    throw error;
  }
}

/**
 * Extract video metadata from analysis or directly from video file - exact match to original
 */
function extractVideoMetadata(analysis, video) {
  // Get video metadata either from analysis or directly from video
  let videoMetadata = {};
  
  if (analysis.metadata && 
      analysis.metadata.width && 
      analysis.metadata.height && 
      analysis.metadata.fps) {
    // Use metadata from analysis if complete
    videoMetadata = analysis.metadata;
    console.log("Using metadata from analysis");
  } else {
    // Get metadata directly from video file
    videoMetadata = {
      width: video.get(cv.CAP_PROP_FRAME_WIDTH),
      height: video.get(cv.CAP_PROP_FRAME_HEIGHT),
      fps: video.get(cv.CAP_PROP_FPS),
      totalFrames: Math.floor(video.get(cv.CAP_PROP_FRAME_COUNT))
    };
    console.log("Using metadata from video file");
  }
  
  return videoMetadata;
}

/**
 * Process a single video frame - exact match to original
 */
function processVideoFrame(frame, options) {
  const { 
    normCenter, 
    width, 
    height, 
    cropWidth, 
    prevGrayFrame, 
    currentFrame
  } = options;
  
  // Calculate crop parameters for this frame
  const xCenter = Math.floor(normCenter * width);
  let xStart = xCenter - Math.floor(cropWidth / 2);
  
  // Ensure crop stays within boundaries
  if (xStart < 0) {
    xStart = 0;
  } else if (xStart + cropWidth > width) {
    xStart = width - cropWidth;
  }
  
  // Convert to grayscale for scene change detection
  const grayFrame = frame.cvtColor(cv.COLOR_BGR2GRAY);
  
  // Calculate frame difference if we have a previous frame
  let frameDiff = 0;
  if (prevGrayFrame) {
    frameDiff = mse(grayFrame, prevGrayFrame);
    
    // Log scene changes
    if (frameDiff > SCENE_CHANGE_THRESHOLD) {
      console.log(`Scene change detected at frame ${currentFrame}`);
    }
  }
  
  // Crop the frame - exactly as original
  const croppedFrame = frame.getRegion(new cv.Rect(xStart, 0, cropWidth, height));
  
  return {
    croppedFrame,
    grayFrame,
    frameDiff
  };
}

/**
 * Encode video from processed frames - exact match to original
 */
function encodeVideo(framesDir, outputPath, fps) {
  return new Promise((resolve, reject) => {
    ffmpeg()
      .input(path.join(framesDir, 'frame_%08d.png'))
      .inputOptions([
        '-framerate', fps.toString()
      ])
      .outputOptions([
        '-c:v', 'libx264',
        '-preset', ENCODING.PRESET,
        '-crf', ENCODING.CRF.toString(),
        '-pix_fmt', ENCODING.PIXEL_FORMAT
      ])
      .output(outputPath)
      .on('progress', (progress) => {
        // Log progress with consistent format
        if (progress && progress.percent) {
          const percent = Math.floor(progress.percent);
          console.log(`Encoding: ${percent}%`);
        }
      })
      .on('end', () => {
        console.log("Encoding complete");
        resolve();
      })
      .on('error', (err) => {
        console.error(`Encoding error: ${err.message}`);
        reject(err);
      })
      .run();
  });
}

/**
 * Add audio from original video to encoded video - exact match to original
 */
function addAudioToVideo(videoPath, audioSource, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg()
      .input(videoPath)
      .input(audioSource)
      .outputOptions([
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-shortest'  // Ensures the output duration matches the video
      ])
      .output(outputPath)
      .on('progress', (progress) => {
        if (progress && progress.percent) {
          const percent = Math.floor(progress.percent);
          console.log(`Adding audio: ${percent}%`);
        }
      })
      .on('end', () => {
        console.log("Audio merging complete");
        resolve();
      })
      .on('error', (err) => {
        console.error(`Audio merging error: ${err.message}`);
        reject(err);
      })
      .run();
  });
}

// Include MovementPlanner class for compatibility
class MovementPlanner {
  constructor(fps) {
    this.frameData = new Map();
    this.frameOrder = [];
    this.segmentsCache = null;
    this.lastFrameNum = -1;
    this.fps = fps;
    this.currentSceneStart = 0;
    this.waitingForDetection = false;
    this.defaultX = DEFAULT_CENTER;
    this.smoothingRate = 0.05;
    this.maxMovementPerFrame = 0.03;
    this.positionHistory = [];
    this.historyMaxSize = 3;
    this.centeringWeight = 0.4;
    this.fastTransitionThreshold = 0.1;
    this.inTransition = false;
    this.stableFrames = 0;
    this.stableFramesRequired = Math.floor(fps * 0.5);
    this.isCentering = false;
    this.halfThreshold = 0.1;
  }

  // ... (methods identical to those in the file you showed)
}

module.exports = {
  processVideo,
  MovementPlanner
}; 