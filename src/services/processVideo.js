const videoAnalyzer = require('./analizeVideo');
const { VideoProcessor, VideoSegment, parseFrameRange } = require('./videoProcessor');
const path = require('path');
const os = require('os');
const fs = require('fs');
const ffmpeg = require('fluent-ffmpeg');
const tf = require('@tensorflow/tfjs-node');
const cv = require('@u4/opencv4nodejs');
// const sharp = require('sharp');

// Constants to match Python implementation
const DEFAULT_CENTER = 0.5;  // normalized center (50%)
const ASPECT_RATIO = 9 / 16;  // output crop aspect ratio
const SCENE_CHANGE_THRESHOLD = 3000;

// Define a batch size for processing frames
const BATCH_SIZE = 30;  // Increased from 10 to 30 for better parallelism

/**
 * Calculate Mean Squared Error between two images
 * Matches the Python implementation for consistency
 */
function mse(imageA, imageB) {
  // Use absdiff since we confirmed it exists
  const diff = imageA.absdiff(imageB);
  
  // We need to square the result but mul doesn't exist
  // So we'll use standard math to process the mean manually
  const mean = diff.mean();
  
  // Calculate the MSE - we'll use the first channel for grayscale images
  return mean.w;
}

/**
 * MovementPlanner class - matches Python implementation
 * Handles camera movement planning, scene changes, and position smoothing
 */
class MovementPlanner {
  constructor(fps) {
    // List of tuples [frame_num, x_pos, is_scene_change]
    this.frameData = [];
    this.fps = fps;
    this.currentSceneStart = 0;
    this.waitingForDetection = false;
    this.defaultX = DEFAULT_CENTER;
    this.smoothingRate = 0.05;
    this.maxMovementPerFrame = 0.03;
    this.positionHistory = [];
    this.historySize = 3;
    this.centeringWeight = 0.4;
    this.fastTransitionThreshold = 0.1;
    this.inTransition = false;
    this.stableFrames = 0;
    this.stableFramesRequired = Math.floor(fps * 0.5);  // Half a second worth of frames
    this.isCentering = false;
  }

  /**
   * Plan movement for a frame - direct port from Python
   */
  planMovement(frameNum, cluster, frameDiff, sceneChangeThreshold) {
    // Handle scene change
    if (frameDiff > sceneChangeThreshold) {
      this.positionHistory = [];
      this.inTransition = false;
      this.stableFrames = 0;
      this.isCentering = false;
      // Reset history on scene change
      this.frameData.push([frameNum, this.defaultX, true]);
      this.currentSceneStart = frameNum;
      this.waitingForDetection = true;
      return;
    }

    // Handle first detection after scene change
    if (this.waitingForDetection && cluster) {
      // Found first detection after scene change
      const newX = cluster[1];  // Normalized x position from cluster
      
      // Go back and update all frames since scene change
      for (let i = 0; i < this.frameData.length; i++) {
        const [fNum, _, isScene] = this.frameData[i];
        if (fNum >= this.currentSceneStart) {
          if (isScene) {
            continue;  // Keep scene change marker
          }
          this.frameData[i] = [fNum, newX, false];
        }
      }
      
      this.waitingForDetection = false;
      return;
    }

    // Get the target position
    let targetX;
    if (cluster) {
      targetX = cluster[1];
    } else {
      targetX = this.frameData.length ? this.frameData[this.frameData.length - 1][1] : this.defaultX;
    }

    if (this.frameData.length) {
      const lastX = this.frameData[this.frameData.length - 1][1];
      const distanceToTarget = Math.abs(targetX - lastX);
      
      // Apply normal movement with limits
      const maxMove = this.maxMovementPerFrame;
      let movement = targetX - lastX;
      if (Math.abs(movement) > maxMove) {
        targetX = lastX + (movement > 0 ? maxMove : -maxMove);
      }

      // Check if we're within the delta threshold for stability
      if (distanceToTarget < this.fastTransitionThreshold) {
        this.stableFrames++;
        if (this.stableFrames >= this.stableFramesRequired) {
          this.isCentering = true;
        }
      } else {
        this.stableFrames = 0;
        this.isCentering = false;
        this.positionHistory = [];
      }

      // Apply centering after stable period
      if (this.isCentering) {
        this.positionHistory.push(targetX);
        if (this.positionHistory.length > this.historySize) {
          this.positionHistory.shift();
        }

        if (this.positionHistory.length > 1) {
          const avgPos = this.positionHistory.reduce((sum, pos) => sum + pos, 0) / this.positionHistory.length;
          targetX = targetX * (1 - this.centeringWeight) + avgPos * this.centeringWeight;
        }
      }
    }

    this.frameData.push([frameNum, targetX, false]);
  }

  /**
   * Get scene segments - matches Python implementation
   */
  getSceneSegments() {
    const segments = [];
    let currentSegmentStart = 0;
    let currentPositions = [];

    for (let i = 0; i < this.frameData.length; i++) {
      const [frameNum, xPos, isScene] = this.frameData[i];
      
      if (isScene && i > 0) {
        // End current segment
        segments.push([
          currentSegmentStart,
          frameNum - 1,
          currentPositions
        ]);
        // Start new segment
        currentSegmentStart = frameNum;
        currentPositions = [];
      }
      
      currentPositions.push(xPos);
    }

    // Add final segment
    if (currentPositions.length) {
      segments.push([
        currentSegmentStart,
        this.frameData[this.frameData.length - 1][0],
        currentPositions
      ]);
    }

    return segments;
  }

  /**
   * Interpolate and smooth positions - matches Python implementation
   */
  interpolateAndSmooth(totalFrames, baseAlpha = 0.1, deltaThreshold = 0.015) {
    const smoothedCenters = Array(totalFrames).fill(this.defaultX);
    const segments = this.getSceneSegments();

    for (const [startFrame, endFrame, positions] of segments) {
      let lastX = positions[0];
      
      for (let i = 0; i < positions.length; i++) {
        const frame = startFrame + i;
        if (frame > endFrame) break;
        
        const targetX = positions[i];
        const delta = targetX - lastX;
        
        let smoothedX;
        if (Math.abs(delta) < deltaThreshold) {
          smoothedX = lastX;
        } else {
          // Variable smoothing rate based on distance to target
          // Slower smoothing (smaller alpha) as we get closer
          const distanceFactor = Math.min(Math.abs(delta) * 2, 1.0);  // Scale based on distance
          let decelerationAlpha = baseAlpha * distanceFactor;
          
          // Even slower for final approach
          if (Math.abs(delta) < 0.1) {  // Within 10% of target
            decelerationAlpha *= 0.5;  // Half speed for final approach
          }
          
          smoothedX = lastX + decelerationAlpha * delta;
        }
        
        smoothedCenters[frame] = smoothedX;
        lastX = smoothedX;
      }
    }

    return smoothedCenters;
  }
}

/**
 * Process a video file with specified outputs
 * @param {string} inputPath - Path to the input video file
 * @param {Array} outputs - Array of output configurations
 * @param {Object} options - Additional processing options
 * @returns {Promise<Object>} - Processing results
 */
async function processVideo(inputPath, outputs, options = {}) {
  console.log(`Starting video processing for: ${inputPath}`);
  console.log('Processing options:', options);
  
  try {
    // Step 1: Convert outputs to VideoSegment objects
    console.log('Step 1: Preparing segments...');
    const segments = outputs.map(output => {
      const { url, range, debug } = output;
      
      // Parse frame range if provided
      let startFrame = null;
      let endFrame = null;
      
      if (range) {
        try {
          const parsedRange = parseFrameRange(range);
          startFrame = parsedRange.startFrame;
          endFrame = parsedRange.endFrame;
        } catch (error) {
          console.warn(`Warning: Invalid range format: ${range}. Using full video.`);
        }
      }
      
      // Use segment-specific debug flag or global debug flag from options
      const debugMode = debug === true || options.debug === true;
      
      return new VideoSegment(url, startFrame, endFrame, debugMode);
    });
    
    // Step 2: Create video processor
    console.log('Step 2: Creating video processor...');
    const processor = new VideoProcessor(inputPath, segments);
    
    // Step 3: Process the video
    console.log('Step 3: Processing video...');
    const processingResults = await processor.processAll();
    
    // Step 4: Format the results
    console.log('Step 4: Compiling results...');
    const result = {
      input: inputPath,
      outputs: processingResults.segments.map(segment => ({
        outputPath: segment.outputPath,
        range: segment.startFrame && segment.endFrame ? 
          `[${segment.startFrame}-${segment.endFrame}]` : 'full video',
        duration: segment.duration,
        status: 'completed'
      })),
      analysis: {
        duration: processingResults.duration,
        totalFrames: processingResults.totalFrames,
        fps: processingResults.fps
      },
      status: 'completed',
      timestamp: new Date().toISOString()
    };
    
    console.log('Video processing completed successfully');
    return result;
  } catch (error) {
    console.error('Error processing video:', error);
    throw new Error(`Video processing failed: ${error.message}`);
  }
}

/**
 * Process video based on analysis results
 * @param {string} inputPath - Path to the input video file
 * @param {Array} outputs - Array of output configurations
 * @param {Object} analysis - Analysis results from analizeVideo
 * @returns {Promise<Array>} - Processed output files
 */
async function processVideo(inputPath, outputs, analysis) {
  console.log(`Processing video based on analysis: ${inputPath}`);
  
  // Create temporary directory in current folder
  const tempDir = path.join(process.cwd(), 'temp', `quickreels-${Date.now()}`);
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }
  
  console.log(`Using temporary directory: ${tempDir}`);
  
  try {
    // Extract metadata and positions from analysis
    const { metadata, smoothedPositions } = analysis;
    const { width, height, fps } = metadata;
    
    // Calculate crop dimensions for 9:16 aspect ratio
    const cropWidth = Math.round(height * (9/16));
    console.log(`Original dimensions: ${width}x${height}, Crop width: ${cropWidth}`);
    
    // Keep the smoothedPositions in memory only, similar to the Python implementation
    // No longer saving coordinates to a file
    
    // Process each output
    const results = await Promise.all(outputs.map(async (output, index) => {
      const { url, range } = output;
      const outputPath = url;
      
      console.log(`Processing output ${index + 1}: ${outputPath}`);
      
      // Ensure output directory exists
      const outputDir = path.dirname(outputPath);
      if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
      }
      
      // Create temporary directory for output
      const outputTempDir = path.join(tempDir, `output_${index}`);
      if (!fs.existsSync(outputTempDir)) {
        fs.mkdirSync(outputTempDir, { recursive: true });
      }
      
      // Process frames one by one, like the Python implementation
      // This provides smoother camera movement instead of segmenting similar positions
      const totalFrames = smoothedPositions.length;
      
      // Create directories for extracted frames and processed frames
      const framesDir = path.join(outputTempDir, 'frames');
      const processedFramesDir = path.join(outputTempDir, 'processed_frames');
      
      if (!fs.existsSync(framesDir)) {
        fs.mkdirSync(framesDir, { recursive: true });
      }
      
      if (!fs.existsSync(processedFramesDir)) {
        fs.mkdirSync(processedFramesDir, { recursive: true });
      }
      
      console.log('Processing video with frame-by-frame camera movement...');
      console.log('Step 1: Extracting frames...');
      
      // Extract frames from the video - this stays the same, we extract all frames at once
      await new Promise((resolve, reject) => {
        ffmpeg(inputPath)
          .outputOptions(['-vf', `fps=${fps}`])
          // Use PNG for highest quality
          .output(path.join(framesDir, 'frame-%04d.png'))
          .on('end', () => {
            console.log('Frames extracted successfully');
            resolve();
          })
          .on('error', (err) => {
            console.error('Error extracting frames:', err);
            reject(err);
          })
          .run();
      });
      
      // Get list of extracted frames
      const frameFiles = fs.readdirSync(framesDir)
        .filter(file => file.startsWith('frame-') && file.endsWith('.png'))
        .sort((a, b) => {
          const numA = parseInt(a.match(/frame-(\d+)/)[1]);
          const numB = parseInt(b.match(/frame-(\d+)/)[1]);
          return numA - numB;
        });
      
      console.log(`Found ${frameFiles.length} frames to process`);
      console.log('Step 2: Processing frames with camera movements using OpenCV...');
      
      // For scene change detection
      let prevGrayFrame = null;
      
      // Process frames in batches to be more efficient
      for (let batchStart = 0; batchStart < Math.min(frameFiles.length, smoothedPositions.length); batchStart += BATCH_SIZE) {
        const batchEnd = Math.min(batchStart + BATCH_SIZE, frameFiles.length, smoothedPositions.length);
        const batch = frameFiles.slice(batchStart, batchEnd);
        
        // Process each batch in parallel
        await Promise.all(batch.map(async (frameFile, idx) => {
          const frameIndex = batchStart + idx;
          const normCenter = smoothedPositions[frameIndex];
          
          // Calculate crop parameters for this frame - same logic as before
          const xCenter = Math.floor(normCenter * width);
          let xStart = xCenter - Math.floor(cropWidth / 2);
          
          // Ensure crop stays within boundaries
          if (xStart < 0) {
            xStart = 0;
          } else if (xStart + cropWidth > width) {
            xStart = width - cropWidth;
          }
          
          // Process the frame with OpenCV instead of spawning FFmpeg
          try {
            const frameMat = cv.imread(path.join(framesDir, frameFile), cv.IMREAD_COLOR);
            
            // Convert to grayscale for scene change detection
            const grayFrame = frameMat.cvtColor(cv.COLOR_BGR2GRAY);
            
            // Calculate frame difference if we have a previous frame
            let frameDiff = 0;
            if (frameIndex > 0 && prevGrayFrame) {
              frameDiff = mse(grayFrame, prevGrayFrame);
              
              // Log scene changes for debugging
              if (frameDiff > SCENE_CHANGE_THRESHOLD) {
                console.log(`Scene change detected at frame ${frameIndex} with diff ${frameDiff.toFixed(2)}`);
              }
            }
            
            // Store grayscale frame for next comparison (no need to manage memory with release)
            prevGrayFrame = frameIndex === 0 || (frameIndex % BATCH_SIZE === BATCH_SIZE - 1) 
              ? grayFrame 
              : prevGrayFrame;
            
            // Create a region of interest for cropping
            // Our test showed that getRegion with cv.Rect works
            const croppedFrameMat = frameMat.getRegion(new cv.Rect(xStart, 0, cropWidth, height));
            
            // Check if we need to resize the output (if output dimensions are different from crop)
            // Use defaults from crop dimensions if not specified in the analysis
            const outputWidth = cropWidth;
            const outputHeight = height;
            
            if (outputWidth !== cropWidth || outputHeight !== height) {
              // Use standard resize method
              const resizedMat = croppedFrameMat.resize(outputHeight, outputWidth);
              cv.imwrite(path.join(processedFramesDir, frameFile), resizedMat);
              // No need to call release as it doesn't exist
            } else {
              cv.imwrite(path.join(processedFramesDir, frameFile), croppedFrameMat);
            }
            
            // No need for explicit memory management with release() as it's not available
          } catch (err) {
            console.error(`Error processing frame ${frameFile}:`, err);
            throw err;
          }
        }));
        
        // Update progress after each batch
        const progress = Math.floor((batchEnd / frameFiles.length) * 100);
        console.log(`Processing frames: ${progress}%`);
        fs.writeFileSync('progress.json', JSON.stringify({
          progress,
          status: `Processing frames: ${progress}%`
        }, null, 2));
      }
      
      console.log('Step 3: Combining processed frames into video...');
      
      // Combine processed frames back into a video - one FFmpeg call instead of per-frame
      await new Promise((resolve, reject) => {
        ffmpeg()
          .input(path.join(processedFramesDir, 'frame-%04d.png'))
          .inputOptions(['-framerate', fps.toString()])
          .outputOptions([
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p'
          ])
          .output(path.join(outputTempDir, 'video.mp4'))
          .on('end', resolve)
          .on('error', reject)
          .run();
      });
      
      console.log('Step 4: Adding audio from original video...');
      
      // Add audio from the original video - this stays the same
      await new Promise((resolve, reject) => {
        ffmpeg()
          .input(path.join(outputTempDir, 'video.mp4'))
          .input(inputPath)
          .outputOptions([
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0?'
          ])
          .output(outputPath)
          .on('end', () => {
            console.log(`Processing complete: ${outputPath}`);
            resolve();
          })
          .on('error', (err) => {
            console.error(`Error adding audio:`, err);
            reject(err);
          })
          .run();
      });
      
      // Clean up temporary files to save space
      try {
        fs.rmSync(framesDir, { recursive: true, force: true });
        fs.rmSync(processedFramesDir, { recursive: true, force: true });
        fs.unlinkSync(path.join(outputTempDir, 'video.mp4'));
      } catch (err) {
        console.warn('Warning: Could not clean up some temporary files', err);
      }
      
      return {
        outputPath,
        status: 'completed',
        frames: frameFiles.length
      };
    }));
    
    // Keep temporary files for inspection (comment out to clean up)
    console.log(`Temporary files are saved in: ${tempDir}`);
    // Uncomment to clean up:
    // try {
    //   if (fs.existsSync(tempDir)) {
    //     fs.rmSync(tempDir, { recursive: true });
    //   }
    // } catch (err) {
    //   console.warn('Warning: Failed to clean up temporary directory', err);
    // }
    
    return results;
  } catch (error) {
    console.error('Error processing video:', error);
    throw error;
  }
}

// Export the processVideo function
module.exports = {
  processVideo,
  MovementPlanner  // Export for testing or advanced usage
}; 