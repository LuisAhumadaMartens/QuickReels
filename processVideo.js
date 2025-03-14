const fs = require('fs');
const path = require('path');
const ffmpeg = require('fluent-ffmpeg');
const cv = require('@u4/opencv4nodejs');
const config = require('./config');

const ASPECT_RATIO = config.ASPECT_RATIO;
const SCENE_CHANGE_THRESHOLD = config.SCENE_CHANGE_THRESHOLD;
const DEFAULT_CENTER = config.DEFAULT_CENTER;
const ENCODING = config.ENCODING;
const BATCH_SIZE = config.BATCH_SIZE;

// Calculate Mean Squared Error between two images
function mse(imageA, imageB) {
  const diff = imageA.absdiff(imageB);
  const mean = diff.mean();
  return mean.w;
}

async function processVideo(inputPath, outputPath, analysis) {
  const { smoothedPositions, tempDir: analysisTemp, metadata } = analysis;
  
  const projectRoot = process.cwd();
  const tempBaseDir = path.join(projectRoot, 'temp');
  const processingId = path.basename(outputPath, path.extname(outputPath));
  const tempDir = analysisTemp || path.join(tempBaseDir, processingId);
  const framesDir = path.join(tempDir, 'frames');
  const processingFramesDir = path.join(framesDir, 'processing');
  const tempInputPath = path.join(tempDir, 'input.mp4');
  const tempOutputPath = path.join(tempDir, 'output.mp4');
  
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
    // Copy input file to temp directory if not already there
    if (!fs.existsSync(tempInputPath)) {
      fs.copyFileSync(inputPath, tempInputPath);
    }
    const video = new cv.VideoCapture(tempInputPath);
    
    // Extract metadata - using original method but with optimization
    const videoMetadata = extractVideoMetadata(metadata, video);
    
    // Calculate crop dimensions for 9:16 aspect ratio
    const { width, height, fps, totalFrames } = videoMetadata;
    const actualCropWidth = Math.round(height * ASPECT_RATIO);
    
    const startFrame = 0;
    const endFrame = totalFrames - 1 || Math.floor(video.get(cv.CAP_PROP_FRAME_COUNT)) - 1;
    
    // For progress reporting
    let processedFrames = 0;
    const totalFramesToProcess = endFrame - startFrame + 1;
    
    let prevGrayFrame = null;
    let currentFrame = 0;
    const framePromises = [];
    
    // Main frame processing loop
    while (true) {
      const frame = video.read();
      if (frame.empty) {
        break;
      }
      if (currentFrame > endFrame) {
        break;
      }
      
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
    
      // Log progress percentage when it changes (every 1%)
      if (processedFrames % Math.max(1, Math.floor(totalFramesToProcess / 100)) === 0 || 
          processedFrames === totalFramesToProcess) {
        const percent = Math.floor((processedFrames / totalFramesToProcess) * 100);
        console.log(`Cropping: ${percent}%`);
      }
    }
    
    if (framePromises.length > 0) {
      await Promise.all(framePromises);
    }
    
    video.release();
    console.log("Frame processing complete, starting video encoding");
    
    await encodeVideo(processingFramesDir, tempOutputPath, fps);
    
    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    console.log("Encoding complete, adding audio");
    
    // Add audio to the video
    await addAudioToVideo(tempOutputPath, tempInputPath, outputPath);
    console.log("Processing complete");
    
    // Clean up temporary files
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

// Extract video metadata
function extractVideoMetadata(metadata, video) {
  let videoMetadata = {};
  
  if (metadata && 
      metadata.width && 
      metadata.height && 
      metadata.fps) {
    videoMetadata = metadata;
    console.log("Using metadata from analysis");
  } else {
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

// Process a video frame (cropping included)
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
  
  const croppedFrame = frame.getRegion(new cv.Rect(xStart, 0, cropWidth, height));
  
  return {
    croppedFrame,
    grayFrame,
    frameDiff
  };
}

// Encode video 
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

// Add audio from original video to encoded video
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

module.exports = {
  processVideo
}; 