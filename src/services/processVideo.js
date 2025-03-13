const videoAnalyzer = require('./analizeVideo');
const path = require('path');
const fs = require('fs');
const ffmpeg = require('fluent-ffmpeg');
const cv = require('@u4/opencv4nodejs');
const crypto = require('crypto');

// Import configuration from central config.js
const config = require('../config/config');

/**
 * Generate a random alphanumeric ID
 * @param {number} [length=config.DEFAULT_ID_LENGTH] - Length of the ID
 * @returns {string} - Random alphanumeric ID
 */
function generateRandomId(length = config.DEFAULT_ID_LENGTH) {
  return crypto.randomBytes(Math.ceil(length / 2))
    .toString('hex')
    .slice(0, length);
}

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
  
  // We need to square the result but mul doesn't exist
  // So we'll use standard math to process the mean manually
  const mean = diff.mean();
  
  // Calculate the MSE - we'll use the first channel for grayscale images
  return mean.w;
}

/**
 * MovementPlanner class - handles camera movement planning, scene changes, and position smoothing
 * This provides intelligent camera tracking based on subject position and scene changes
 */
class MovementPlanner {
  /**
   * Create a new MovementPlanner
   * @param {number} fps - Frames per second of the video
   */
  constructor(fps) {
    // Simplified data structure: single Map for all frame data
    // frameData: Map<frameNum, {position: number, isSceneChange: boolean}>
    this.frameData = new Map();
    this.frameOrder = []; // Keep this for ordered iteration
    
    // Cache for optimized segment retrieval
    this.segmentsCache = null;
    this.lastFrameNum = -1;
    
    // Constants
    this.fps = fps;
    this.currentSceneStart = 0;
    this.waitingForDetection = false;
    this.defaultX = config.DEFAULT_CENTER;
    this.smoothingRate = 0.05;
    this.maxMovementPerFrame = 0.03;
    
    // Simplified position history: just use a fixed-size array
    this.positionHistory = [];
    this.historyMaxSize = 3;
    
    this.centeringWeight = 0.4;
    this.fastTransitionThreshold = 0.1;
    this.inTransition = false;
    this.stableFrames = 0;
    this.stableFramesRequired = Math.floor(fps * 0.5);  // Half a second worth of frames
    this.isCentering = false;
    
    // Pre-compute frequently used values
    this.halfThreshold = 0.1; // Within 10% of target
  }

  /**
   * Add position to history, maintaining the maximum size
   * @private
   */
  _updatePositionHistory(position) {
    // Add to history and keep only the most recent entries
    this.positionHistory.push(position);
    if (this.positionHistory.length > this.historyMaxSize) {
      this.positionHistory.shift(); // Remove oldest entry
    }
  }

  /**
   * Calculate average of position history
   * @private
   */
  _getPositionHistoryAverage() {
    if (this.positionHistory.length === 0) return this.defaultX;
    
    const sum = this.positionHistory.reduce((acc, pos) => acc + pos, 0);
    return sum / this.positionHistory.length;
  }

  /**
   * Reset position history
   * @private
   */
  _resetPositionHistory() {
    this.positionHistory = [];
  }

  /**
   * Get the last frame position
   * @private
   */
  _getLastPosition() {
    if (this.frameOrder.length === 0) return this.defaultX;
    const lastFrame = this.frameOrder[this.frameOrder.length - 1];
    return this.frameData.get(lastFrame).position;
  }

  /**
   * Plan movement for a frame - optimized version
   */
  planMovement(frameNum, cluster, frameDiff, sceneChangeThreshold) {
    // Invalidate segments cache when adding new data
    this.segmentsCache = null;
    this.lastFrameNum = Math.max(this.lastFrameNum, frameNum);
    
    // Handle scene change (fast path)
    if (frameDiff > sceneChangeThreshold) {
      this._resetPositionHistory();
      this.inTransition = false;
      this.stableFrames = 0;
      this.isCentering = false;
      
      // Store scene change data
      this.frameData.set(frameNum, { position: this.defaultX, isSceneChange: true });
      this.frameOrder.push(frameNum);
      
      this.currentSceneStart = frameNum;
      this.waitingForDetection = true;
      return;
    }

    // Handle first detection after scene change
    if (this.waitingForDetection && cluster) {
      // Found first detection after scene change
      const newX = cluster[1];  // Normalized x position from cluster
      
      // Update all frames since scene change
      for (const fNum of this.frameOrder) {
        if (fNum >= this.currentSceneStart) {
          const frameData = this.frameData.get(fNum);
          // Skip scene change frames
          if (frameData.isSceneChange) continue;  
          
          // Update position
          this.frameData.set(fNum, { 
            position: newX, 
            isSceneChange: false 
          });
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
      targetX = this._getLastPosition();
    }

    if (this.frameOrder.length > 0) {
      const lastX = this._getLastPosition();
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
        this._resetPositionHistory();
      }

      // Apply centering after stable period
      if (this.isCentering) {
        this._updatePositionHistory(targetX);

        if (this.positionHistory.length > 1) {
          const avgPos = this._getPositionHistoryAverage();
          targetX = targetX * (1 - this.centeringWeight) + avgPos * this.centeringWeight;
        }
      }
    }

    // Store frame data
    this.frameData.set(frameNum, { position: targetX, isSceneChange: false });
    this.frameOrder.push(frameNum);
  }

  /**
   * Get scene segments - optimized implementation with caching
   */
  getSceneSegments() {
    // Return cached result if available and no new frames added
    if (this.segmentsCache !== null) {
      return this.segmentsCache;
    }
    
    const segments = [];
    let currentSegmentStart = 0;
    let currentPositions = [];
    let lastFrameNum = -1;

    // Iterate through frames in order
    for (const frameNum of this.frameOrder) {
      const { position, isSceneChange } = this.frameData.get(frameNum);
      
      // Check if this is the first frame
      if (lastFrameNum === -1) {
        currentSegmentStart = frameNum;
      }
      // Check for scene change
      else if (isSceneChange) {
        // End current segment
        segments.push([
          currentSegmentStart,
          lastFrameNum,
          currentPositions
        ]);
        // Start new segment
        currentSegmentStart = frameNum;
        currentPositions = [];
      }
      
      currentPositions.push(position);
      lastFrameNum = frameNum;
    }

    // Add final segment
    if (currentPositions.length) {
      segments.push([
        currentSegmentStart,
        lastFrameNum,
        currentPositions
      ]);
    }

    // Cache the result
    this.segmentsCache = segments;
    return segments;
  }

  /**
   * Interpolate and smooth positions - optimized implementation
   */
  interpolateAndSmooth(totalFrames, baseAlpha = 0.1, deltaThreshold = 0.015) {
    // Pre-allocate result array
    const smoothedCenters = new Array(totalFrames);
    smoothedCenters.fill(this.defaultX);
    
    // Get segments (using cached result if available)
    const segments = this.getSceneSegments();

    // Pre-calculate common threshold values
    const halfThresholdValue = 0.1;

    for (const [startFrame, endFrame, positions] of segments) {
      if (positions.length === 0) continue;
      
      let lastX = positions[0];
      
      for (let i = 0; i < positions.length; i++) {
        const frame = startFrame + i;
        if (frame > endFrame || frame >= totalFrames) break;
        
        const targetX = positions[i];
        const delta = targetX - lastX;
        const absDelta = Math.abs(delta);
        
        let smoothedX;
        // Fast path for small movements
        if (absDelta < deltaThreshold) {
          smoothedX = lastX;
        } else {
          // Variable smoothing rate with optimized calculations
          const distanceFactor = Math.min(absDelta * 2, 1.0);
          let decelerationAlpha = baseAlpha * distanceFactor;
          
          // Check for final approach
          if (absDelta < halfThresholdValue) {
            decelerationAlpha *= 0.5;
          }
          
          smoothedX = lastX + decelerationAlpha * delta;
        }
        
        smoothedCenters[frame] = smoothedX;
        lastX = smoothedX;
      }
    }

    return smoothedCenters;
  }
  
  /**
   * Convert to the original data format for compatibility
   * This ensures output is identical to the original implementation
   */
  toOriginalFormat() {
    const originalFrameData = [];
    
    for (const frameNum of this.frameOrder) {
      const { position, isSceneChange } = this.frameData.get(frameNum);
      originalFrameData.push([frameNum, position, isSceneChange]);
    }
    
    return originalFrameData;
  }
}

// Track completed jobs to prevent duplicate updates
const completedJobs = new Set();

/**
 * Update the progress tracking file with job status
 * @param {string} jobId - The job ID
 * @param {Object} statusUpdate - The status update object containing progress and status information
 * @param {Object} [statusUpdate.analysis] - Analysis phase status
 * @param {Object} [statusUpdate.processing] - Processing phase status
 * @param {Object} [statusUpdate.encoding] - Encoding phase status
 * @param {Object} [statusUpdate.audio] - Audio merging phase status
 */
function updateProgress(jobId, statusUpdate) {
  try {
    // If the job has been marked as completed, ignore further updates
    if (completedJobs.has(jobId)) {
      return;
    }

    // Read existing progress data once
    let progressData = {};
    const progressFilePath = config.PROGRESS_FILE;
    
    if (fs.existsSync(progressFilePath)) {
      progressData = JSON.parse(fs.readFileSync(progressFilePath, 'utf-8'));
    }
    
    // Initialize job entry if it doesn't exist
    if (!progressData[jobId]) {
      progressData[jobId] = {
        status: "Initializing...",
        analysis: 0,
        processing: 0,
        encoding: 0,
        audio: 0
      };
    }
    
    // Generalized function to update a specific phase
    const updatePhase = (phase, phaseUpdate) => {
      if (!phaseUpdate) return false;
      
      let wasUpdated = false;
      
      // Update progress if provided and higher than current (or error state)
      if (phaseUpdate.progress !== undefined) {
        const newProgress = phaseUpdate.progress;
        if (newProgress === -1 || newProgress >= progressData[jobId][phase]) {
          progressData[jobId][phase] = newProgress;
          wasUpdated = true;
        }
      }
      
      // Update status message if provided
      if (phaseUpdate.status) {
        progressData[jobId].status = phaseUpdate.status;
        console.log(`Job ID [${jobId}]: ${phaseUpdate.status}`);
        wasUpdated = true;
      }
      
      return wasUpdated;
    };
    
    // Update each phase if needed
    const analysisUpdated = updatePhase('analysis', statusUpdate.analysis);
    const processingUpdated = updatePhase('processing', statusUpdate.processing);
    const encodingUpdated = updatePhase('encoding', statusUpdate.encoding);
    const audioUpdated = updatePhase('audio', statusUpdate.audio);
    
    // Only write to file if anything changed
    if (analysisUpdated || processingUpdated || encodingUpdated || audioUpdated) {
      // Check for completion conditions
      const isComplete = statusUpdate.processing && 
                         statusUpdate.processing.status === "Processing complete";
      
      const hasError = statusUpdate.processing && 
                       statusUpdate.processing.status && 
                       statusUpdate.processing.status.startsWith("Error");
      
      // Handle job completion - regular completion or error
      if (isComplete || hasError) {
        if (isComplete) {
          // Set final status for regular completion
          progressData[jobId].status = "Video generation complete";
          console.log(`Job ID [${jobId}]: Video generation complete`);
        }
        
        // Write the progress file with the completion status
        fs.writeFileSync(progressFilePath, JSON.stringify(progressData, null, 2));
        
        // Mark job as completed
        completedJobs.add(jobId);
        
        // Remove the job from progress tracking
        delete progressData[jobId];
        console.log(`Job ID [${jobId}]: Job ${hasError ? 'completed with error' : 'completed'}, removed from tracking`);
        
        // Write the updated file without the job
        fs.writeFileSync(progressFilePath, JSON.stringify(progressData, null, 2));
      } else {
        // Regular update - write once at the end
        fs.writeFileSync(progressFilePath, JSON.stringify(progressData, null, 2));
      }
    }
  } catch (err) {
    console.warn(`Job ID [${jobId}]: Warning: Could not update progress: ${err.message}`);
  }
}

/**
 * Process a video based on analysis results
 * @param {string} inputPath - Path to the input video file
 * @param {string} outputPath - Path for the output video file
 * @param {Object} analysis - Analysis results from analizeVideo
 * @param {string} [jobId] - Optional job ID for progress tracking
 * @returns {Promise<Object>} - Processing result
 */
async function processVideo(inputPath, outputPath, analysis, jobId = null) {
  // Use the job ID from analysis if available, otherwise generate a new one
  const processingId = analysis.jobId || jobId || generateRandomId();
  console.log(`Job ID [${processingId}]: Processing video`);
  
  // Use processing phases from config
  const processingPhases = config.PROGRESS.PROCESSING_PHASES;
  
  // Simplified progress tracking function
  function updatePhaseProgress(phase, phaseProgress = 1) {
    // Only track frame cropping progress
    if (phase !== 'frameCropping') {
      return 0;
    }
    
    // Convert phase name to config key format
    const phaseKey = phase.replace(/([A-Z])/g, '_$1').toUpperCase();
    const phaseInfo = processingPhases[phaseKey];
    
    if (!phaseInfo) return 0;
    
    // Calculate percentage for frame cropping phase
    const phasePercentage = Math.floor(phaseProgress * 100);
    
    // Create the status message for frame cropping
    const statusMessage = `${phaseInfo.description}: ${phasePercentage}%`;
    
    // Update processing progress for frame cropping
    const progressUpdate = {
      processing: { 
        progress: phasePercentage,
        status: statusMessage
      }
    };
    
    // Update progress
    updateProgress(processingId, progressUpdate);
    
    return phasePercentage;
  }
  
  // Set up directories
  const projectRoot = process.cwd();
  const tempBaseDir = path.join(projectRoot, config.TEMP_DIR_NAME);
  const tempDir = analysis.tempDir || path.join(tempBaseDir, processingId);
  const framesDir = path.join(tempDir, config.FRAMES_DIR_NAME);
  const processingFramesDir = path.join(framesDir, config.PROCESSING_DIR_NAME);
  const tempInputPath = path.join(tempDir, config.TEMP_INPUT_NAME);
  const tempOutputPath = path.join(tempDir, config.TEMP_OUTPUT_NAME);
  
  // Create all directories at once with recursive option
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
  
  // Update progress.json with job ID - analysis is complete, starting preparation
  updateProgress(processingId, {
    analysis: { progress: 100, status: "Analysis complete" }
  });
  updatePhaseProgress('preparation', 0);
  
  try {
    // Step 1: Copy input file to temp directory if not already there
    if (!fs.existsSync(tempInputPath)) {
      fs.copyFileSync(inputPath, tempInputPath);
    }
    
    // Open the video file to get accurate properties
    const video = new cv.VideoCapture(tempInputPath);
    
    // Step 2: Extract metadata
    const videoMetadata = extractVideoMetadata(analysis, video, processingId);
    const { smoothedPositions } = analysis;
    
    // Update progress - preparation phase halfway complete
    updatePhaseProgress('preparation', 0.5);
    
    // Calculate crop dimensions for 9:16 aspect ratio
    const { width, height, fps, totalFrames } = videoMetadata;
    const actualCropWidth = Math.round(height * config.ASPECT_RATIO);
    
    // Process the entire video
    const startFrame = 0;
    const endFrame = totalFrames - 1 || Math.floor(video.get(cv.CAP_PROP_FRAME_COUNT)) - 1;
    
    // For progress reporting
    let processedFrames = 0;
    const totalFramesToProcess = endFrame - startFrame + 1;
    let lastProgressReport = Date.now();
    
    // For scene change detection
    let prevGrayFrame = null;
    
    // Process frames
    let currentFrame = 0;
    
    // Preparation phase complete, starting frame processing
    updatePhaseProgress('preparation', 1);
    updatePhaseProgress('frameCropping', 0);
    
    // Batch processing for better performance
    const framePromises = [];
    
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
        : config.DEFAULT_CENTER;
      
      // Process the video frame
      const { croppedFrame, grayFrame, frameDiff } = processVideoFrame(frame, {
        normCenter,
        width,
        height,
        cropWidth: actualCropWidth,
        prevGrayFrame,
        currentFrame,
        jobId: processingId
      });
      
      // Update previous frame for next comparison
      prevGrayFrame = grayFrame;
      
      // Save the frame to disk (async)
      const framePath = path.join(processingFramesDir, `frame_${String(processedFrames).padStart(8, '0')}.png`);
      framePromises.push(
        (async () => {
          try {
            await cv.imwriteAsync(framePath, croppedFrame);
          } catch (err) {
            console.error(`Job ID [${processingId}]: Error writing frame ${currentFrame}: ${err.message}`);
          }
        })()
      );
      
      // Process in batches to avoid too many open files
      if (framePromises.length >= config.BATCH_SIZE) {
        await Promise.all(framePromises);
        framePromises.length = 0;
      }
      
      // Update progress
      processedFrames++;
      currentFrame++;
      
      // Report progress periodically
      const now = Date.now();
      if (now - lastProgressReport > config.PROGRESS.UPDATE_INTERVAL_MS) {
        // Calculate frame cropping phase progress (0-1)
        const framesProgress = processedFrames / totalFramesToProcess;
        
        // Update progress using the phase-based calculation
        updatePhaseProgress('frameCropping', framesProgress);
        lastProgressReport = now;
      }
    }
    
    // Process any remaining frames
    if (framePromises.length > 0) {
      await Promise.all(framePromises);
    }
    
    // Release resources
    video.release();
    
    // Frame cropping complete
    updatePhaseProgress('frameCropping', 1);
    
    // Encode the video
    await encodeVideo(processingFramesDir, tempOutputPath, fps, processingId);
    
    // Create output directory if it doesn't exist
    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    // Add audio to the video
    await addAudioToVideo(tempOutputPath, tempInputPath, outputPath, processingId);
    
    // Update progress to 100% when processing is complete
    updateProgress(processingId, {
      processing: {
        progress: 100,
        status: "Processing complete"
      },
      encoding: {
        progress: 100,
        status: "Encoding complete"
      },
      audio: {
        progress: 100,
        status: "Audio merging complete"
      },
      videoGenerated: true
    });
    
    // Clean up temporary files
    try {
      fs.rmSync(tempDir, { recursive: true, force: true });
    } catch (err) {
      console.warn(`Job ID [${processingId}]: Warning: Failed to clean up temporary directory: ${err.message}`);
    }
    
    return {
      jobId: processingId,
      outputPath,
      duration: ((endFrame - startFrame + 1) / fps).toFixed(2),
      status: 'completed'
    };
  } catch (error) {
    console.error(`Job ID [${processingId}]: Error processing video: ${error.message}`);
    
    // Update progress to show error
    updateProgress(processingId, {
      processing: { progress: -1, status: `Error: ${error.message}` }
    });
    
    throw error;
  }
}

// Helper functions for processVideo
/**
 * Extract video metadata from analysis or directly from video file
 * @param {Object} analysis - Analysis results 
 * @param {Object} video - CV VideoCapture object
 * @param {string} jobId - Processing job ID
 * @returns {Object} Video metadata
 */
function extractVideoMetadata(analysis, video, jobId) {
  // Get video metadata either from analysis or directly from video
  let videoMetadata = {};
  
  if (analysis.metadata && 
      analysis.metadata.width && 
      analysis.metadata.height && 
      analysis.metadata.fps) {
    // Use metadata from analysis if complete
    videoMetadata = analysis.metadata;
    console.log(`Job ID [${jobId}]: Using metadata from analysis`);
  } else {
    // Get metadata directly from video file
    videoMetadata = {
      width: video.get(cv.CAP_PROP_FRAME_WIDTH),
      height: video.get(cv.CAP_PROP_FRAME_HEIGHT),
      fps: video.get(cv.CAP_PROP_FPS),
      totalFrames: Math.floor(video.get(cv.CAP_PROP_FRAME_COUNT))
    };
    console.log(`Job ID [${jobId}]: Using metadata from video file`);
  }
  
  return videoMetadata;
}

/**
 * Process a single video frame
 * @param {Object} frame - CV frame object
 * @param {Object} options - Frame processing options
 * @param {number} options.normCenter - Normalized center position
 * @param {number} options.width - Video width
 * @param {number} options.height - Video height
 * @param {number} options.cropWidth - Width to crop
 * @param {Object} options.prevGrayFrame - Previous grayscale frame for scene detection
 * @param {number} options.currentFrame - Current frame number
 * @param {string} options.jobId - Processing job ID
 * @returns {Object} Processed frame data
 */
function processVideoFrame(frame, options) {
  const { 
    normCenter, 
    width, 
    height, 
    cropWidth, 
    prevGrayFrame, 
    currentFrame,
    jobId
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
    if (frameDiff > config.SCENE_CHANGE_THRESHOLD) {
      console.log(`Job ID [${jobId}]: Scene change detected at frame ${currentFrame}`);
    }
  }
  
  // Crop the frame
  const croppedFrame = frame.getRegion(new cv.Rect(xStart, 0, cropWidth, height));
  
  return {
    croppedFrame,
    grayFrame,
    frameDiff
  };
}

/**
 * Encode video from processed frames
 * @param {string} framesDir - Directory containing processed frames
 * @param {string} outputPath - Path for output video
 * @param {number} fps - Frames per second
 * @param {string} jobId - Processing job ID for logging
 * @returns {Promise<void>}
 */
async function encodeVideo(framesDir, outputPath, fps, jobId) {
  return new Promise((resolve, reject) => {
    ffmpeg()
      .input(path.join(framesDir, 'frame_%08d.png'))
      .inputOptions([
        '-framerate', fps.toString()
      ])
      .outputOptions([
        '-c:v', 'libx264',
        '-preset', config.ENCODING.PRESET,
        '-crf', config.ENCODING.CRF.toString(),
        '-pix_fmt', config.ENCODING.PIXEL_FORMAT
      ])
      .output(outputPath)
      .on('progress', (progress) => {
        // Update progress.json and log progress with consistent format
        if (progress && progress.percent) {
          const percent = Math.floor(progress.percent);
          updateProgress(jobId, {
            encoding: {
              progress: percent,
              status: `Encoding: ${percent}%`
            }
          });
          console.log(`Job ID [${jobId}]: Encoding: ${percent}%`);
        }
      })
      .on('end', () => {
        // Always set to 100% when encoding is complete, regardless of last reported progress
        updateProgress(jobId, {
          encoding: { 
            progress: 100,
            status: "Encoding complete"
          }
        });
        console.log(`Job ID [${jobId}]: Encoding complete`);
        resolve();
      })
      .on('error', (err) => {
        // Log error to progress.json
        updateProgress(jobId, {
          encoding: { 
            progress: -1,
            status: `Encoding error: ${err.message}`
          }
        });
        console.error(`Job ID [${jobId}]: Encoding error: ${err.message}`);
        reject(err);
      })
      .run();
  });
}

/**
 * Add audio from original video to encoded video
 * @param {string} videoPath - Path to encoded video without audio
 * @param {string} audioSource - Path to audio source
 * @param {string} outputPath - Path for final output with audio
 * @param {string} jobId - Processing job ID for logging
 * @returns {Promise<void>}
 */
async function addAudioToVideo(videoPath, audioSource, outputPath, jobId) {
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
        // Update progress.json and log progress with consistent format
        if (progress && progress.percent) {
          const percent = Math.floor(progress.percent);
          updateProgress(jobId, {
            audio: {
              progress: percent,
              status: `Adding audio: ${percent}%`
            }
          });
          console.log(`Job ID [${jobId}]: Adding audio: ${percent}%`);
        }
      })
      .on('end', () => {
        // Mark audio merging as complete in progress.json
        updateProgress(jobId, {
          audio: { 
            progress: 100,
            status: "Audio merging complete"
          }
        });
        console.log(`Job ID [${jobId}]: Audio merging complete`);
        resolve();
      })
      .on('error', (err) => {
        // Log error to progress.json
        updateProgress(jobId, {
          audio: { 
            progress: -1,
            status: `Audio merging error: ${err.message}`
          }
        });
        console.error(`Job ID [${jobId}]: Audio merging error: ${err.message}`);
        reject(err);
      })
      .run();
  });
}

// Export the processVideo function
module.exports = {
  processVideo,
  MovementPlanner,  // Export for testing or advanced usage
  generateRandomId,  // Export for reuse in other modules
  updateProgress     // Export for external use
}; 