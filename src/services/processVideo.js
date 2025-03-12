const videoAnalyzer = require('./analizeVideo');
const { VideoProcessor, VideoSegment, parseFrameRange } = require('./videoProcessor');
const path = require('path');
const os = require('os');
const fs = require('fs');
const ffmpeg = require('fluent-ffmpeg');
const tf = require('@tensorflow/tfjs-node');
const cv = require('@u4/opencv4nodejs');
const crypto = require('crypto');
// const sharp = require('sharp');

// Constants to match Python implementation
const DEFAULT_CENTER = 0.5;  // normalized center (50%)
const ASPECT_RATIO = 9 / 16;  // output crop aspect ratio
const SCENE_CHANGE_THRESHOLD = 3000;

// Define a batch size for processing frames
const BATCH_SIZE = 10;

/**
 * Generate a random alphanumeric ID
 * @param {number} length - Length of the ID
 * @returns {string} - Random alphanumeric ID
 */
function generateRandomId(length = 10) {
  return crypto.randomBytes(Math.ceil(length / 2))
    .toString('hex')
    .slice(0, length);
}

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
 * Optimized version with improved data structures and algorithms
 */
class MovementPlanner {
  constructor(fps) {
    // Optimized data structures
    this.framePositions = new Map(); // frameNum -> [x_pos, is_scene_change]
    this.sceneChanges = new Set(); // Set of frameNums that are scene changes
    this.frameOrder = []; // Ordered list of processed frames for faster iteration
    
    // Cache for optimized segment retrieval
    this.segmentsCache = null;
    this.lastFrameNum = -1;
    
    // Constants
    this.fps = fps;
    this.currentSceneStart = 0;
    this.waitingForDetection = false;
    this.defaultX = DEFAULT_CENTER;
    this.smoothingRate = 0.05;
    this.maxMovementPerFrame = 0.03;
    
    // Use circular buffer for position history
    this.positionHistoryBuffer = new Array(3).fill(0);
    this.positionHistoryIndex = 0;
    this.positionHistoryCount = 0;
    this.historySize = 3;
    
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
   * Add position to circular buffer
   * @private
   */
  _addToPositionHistory(position) {
    this.positionHistoryBuffer[this.positionHistoryIndex] = position;
    this.positionHistoryIndex = (this.positionHistoryIndex + 1) % this.historySize;
    if (this.positionHistoryCount < this.historySize) {
      this.positionHistoryCount++;
    }
  }

  /**
   * Calculate average of position history
   * @private
   */
  _getPositionHistoryAverage() {
    if (this.positionHistoryCount === 0) return this.defaultX;
    
    let sum = 0;
    for (let i = 0; i < this.positionHistoryCount; i++) {
      sum += this.positionHistoryBuffer[i];
    }
    return sum / this.positionHistoryCount;
  }

  /**
   * Reset position history buffer
   * @private
   */
  _resetPositionHistory() {
    this.positionHistoryBuffer.fill(0);
    this.positionHistoryIndex = 0;
    this.positionHistoryCount = 0;
  }

  /**
   * Get the last frame position
   * @private
   */
  _getLastPosition() {
    if (this.frameOrder.length === 0) return this.defaultX;
    const lastFrame = this.frameOrder[this.frameOrder.length - 1];
    return this.framePositions.get(lastFrame)[0];
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
      this.framePositions.set(frameNum, [this.defaultX, true]);
      this.sceneChanges.add(frameNum);
      this.frameOrder.push(frameNum);
      
      this.currentSceneStart = frameNum;
      this.waitingForDetection = true;
      return;
    }

    // Handle first detection after scene change
    if (this.waitingForDetection && cluster) {
      // Found first detection after scene change
      const newX = cluster[1];  // Normalized x position from cluster
      
      // Update all frames since scene change (more efficient with Map)
      for (const fNum of this.frameOrder) {
        if (fNum >= this.currentSceneStart) {
          const [_, isScene] = this.framePositions.get(fNum);
          if (isScene) continue;  // Keep scene change marker
          
          this.framePositions.set(fNum, [newX, false]);
        }
      }
      
      this.waitingForDetection = false;
      return;
    }

    // Get the target position (optimized lookups)
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

      // Check if we're within the delta threshold for stability (early exit possible)
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

      // Apply centering after stable period (optimized buffer operations)
      if (this.isCentering) {
        this._addToPositionHistory(targetX);

        if (this.positionHistoryCount > 1) {
          const avgPos = this._getPositionHistoryAverage();
          targetX = targetX * (1 - this.centeringWeight) + avgPos * this.centeringWeight;
        }
      }
    }

    // Store frame data
    this.framePositions.set(frameNum, [targetX, false]);
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

    // Optimize iteration by using ordered frameOrder array
    for (const frameNum of this.frameOrder) {
      const [xPos, isScene] = this.framePositions.get(frameNum);
      
      // Check if this is the first frame
      if (lastFrameNum === -1) {
        currentSegmentStart = frameNum;
      }
      // Check for scene change
      else if (isScene) {
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
      
      currentPositions.push(xPos);
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
      const [xPos, isScene] = this.framePositions.get(frameNum);
      originalFrameData.push([frameNum, xPos, isScene]);
    }
    
    return originalFrameData;
  }
}

/**
 * Update the progress tracking file with job status
 * @param {string} jobId - The job ID
 * @param {Object} statusUpdate - The status update object
 * @param {Object} [statusUpdate.analysis] - Analysis status update
 * @param {number} [statusUpdate.analysis.progress] - Analysis progress (0-100)
 * @param {string} [statusUpdate.analysis.status] - Analysis status message
 * @param {Object} [statusUpdate.processing] - Processing status update
 * @param {number} [statusUpdate.processing.progress] - Processing progress (0-100)
 * @param {string} [statusUpdate.processing.status] - Processing status message
 * @param {boolean} [statusUpdate.videoGenerated] - Whether the video has been generated
 */
function updateProgress(jobId, statusUpdate) {
  try {
    // Read existing progress data
    let progressData = {};
    if (fs.existsSync('progress.json')) {
      progressData = JSON.parse(fs.readFileSync('progress.json', 'utf-8'));
    }
    
    // Initialize job entry if it doesn't exist
    if (!progressData[jobId]) {
      progressData[jobId] = {
        status: "Initializing...",
        analysis: 0,
        processing: 0,
        videoGenerated: false
      };
    }
    
    // Update job data with the provided updates, but ensure progress never regresses
    if (statusUpdate.analysis) {
      // Get the progress value
      const newProgress = statusUpdate.analysis.progress;
      
      // Only update progress if the new value is higher than the existing one
      // (unless it's an error state with progress = -1)
      if (newProgress === -1 || newProgress >= progressData[jobId].analysis) {
        progressData[jobId].analysis = newProgress;
      }
      
      // Update the general status from analysis status
      if (statusUpdate.analysis.status) {
        progressData[jobId].status = statusUpdate.analysis.status;
        // Log the status update to console with the job ID
        console.log(`Job ID [${jobId}]: ${statusUpdate.analysis.status}`);
      }
    }
    
    if (statusUpdate.processing) {
      // Get the progress value
      const newProgress = statusUpdate.processing.progress;
      
      // Only update progress if the new value is higher than the existing one
      // (unless it's an error state with progress = -1)
      if (newProgress === -1 || newProgress >= progressData[jobId].processing) {
        progressData[jobId].processing = newProgress;
      }
      
      // Update the general status from processing status
      if (statusUpdate.processing.status) {
        progressData[jobId].status = statusUpdate.processing.status;
        // Log the status update to console with the job ID
        console.log(`Job ID [${jobId}]: ${statusUpdate.processing.status}`);
      }
    }
    
    if (statusUpdate.videoGenerated !== undefined) {
      progressData[jobId].videoGenerated = statusUpdate.videoGenerated;
      
      // If video is generated, update the general status
      if (statusUpdate.videoGenerated === true) {
        progressData[jobId].status = "Video generation complete";
        console.log(`Job ID [${jobId}]: Video generation complete`);
      }
    }
    
    // If video is generated or there's an error, schedule removal
    if (statusUpdate.videoGenerated === true || 
        (statusUpdate.processing && statusUpdate.processing.status && statusUpdate.processing.status.startsWith("Error"))) {
      // Immediately remove the job from progress.json instead of waiting
      try {
        // Write updated progress data first
        fs.writeFileSync('progress.json', JSON.stringify(progressData, null, 2));
        
        console.log(`Job ID [${jobId}]: Job completed, removed from tracking`);
        
        // Immediately delete the entry from progress.json
        delete progressData[jobId];
        fs.writeFileSync('progress.json', JSON.stringify(progressData, null, 2));
        
        return; // Skip the regular write since we've already written the file
      } catch (err) {
        console.warn(`Job ID [${jobId}]: Warning: Could not remove job from progress tracking: ${err.message}`);
      }
    }
    
    // Write updated progress data
    fs.writeFileSync('progress.json', JSON.stringify(progressData, null, 2));
  } catch (err) {
    console.warn(`Job ID [${jobId}]: Warning: Could not update progress: ${err.message}`);
  }
}

/**
 * Process a video based on analysis results
 * @param {string} inputPath - Path to the input video file
 * @param {string} outputPath - Path for the output video file
 * @param {Object} analysis - Analysis results from analizeVideo
 * @param {string} [jobId] - Optional job ID for progress tracking (generated if not provided)
 * @returns {Promise<Object>} - Processing result
 */
async function processVideo(inputPath, outputPath, analysis, jobId = null) {
  // Use the job ID from analysis if available, otherwise generate a new one
  const processingId = analysis.jobId || jobId || generateRandomId();
  console.log(`Job ID [${processingId}]: Processing video`);
  
  // Define processing phases with weighted percentages
  const processingPhases = {
    preparation: { weight: 0.05, description: "Preparing for processing" },
    frameCropping: { weight: 0.70, description: "Processing frames" },
    encoding: { weight: 0.20, description: "Encoding video" },
    audioMerging: { weight: 0.05, description: "Adding audio" }
  };
  
  // Function to update progress based on current phase and completion percentage
  function updatePhaseProgress(phase, phaseProgress = 1) {
    // Calculate overall progress based on weights and current phase
    let totalProgress = 0;
    let phaseFound = false;
    
    for (const [phaseName, phaseInfo] of Object.entries(processingPhases)) {
      if (phaseName === phase) {
        totalProgress += phaseInfo.weight * phaseProgress;
        phaseFound = true;
      } else if (!phaseFound) {
        totalProgress += phaseInfo.weight; // Add 100% for completed phases
      }
    }
    
    // Calculate the overall percentage (0-100)
    const overallPercentage = Math.floor(totalProgress * 100);
    
    // Get the phase description and percent for status message
    const phaseInfo = processingPhases[phase];
    const phasePercentage = Math.floor(phaseProgress * 100);
    let statusMessage = `${phaseInfo.description}: ${phasePercentage}%`;
    
    // Special case for 100% completion
    if (phase === 'audioMerging' && phaseProgress === 1) {
      statusMessage = "Processing complete";
    }
    
    // Update progress with calculated percentage
    updateProgress(processingId, {
      processing: { 
        progress: overallPercentage,
        status: statusMessage
      }
    });
    
    return overallPercentage;
  }
  
  // Create project-relative temp directory structure
  const projectRoot = process.cwd();
  const tempBaseDir = path.join(projectRoot, 'temp');
  const tempDir = analysis.tempDir || path.join(tempBaseDir, processingId);
  const framesDir = path.join(tempDir, 'frames');
  const processingFramesDir = path.join(framesDir, 'processing');
  const tempInputPath = path.join(tempDir, 'input.mp4');
  const tempOutputPath = path.join(tempDir, 'output.mp4');
  
  // Create directories if they don't exist yet
  if (!fs.existsSync(tempBaseDir)) {
    fs.mkdirSync(tempBaseDir, { recursive: true });
  }
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }
  if (!fs.existsSync(framesDir)) {
    fs.mkdirSync(framesDir, { recursive: true });
  }
  if (!fs.existsSync(processingFramesDir)) {
    fs.mkdirSync(processingFramesDir, { recursive: true });
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
    
    // Step 2: Extract metadata and positions from analysis
    const { metadata, smoothedPositions } = analysis;
    const { width, height, fps } = metadata || {};
    
    // Update progress - preparation phase halfway complete
    updatePhaseProgress('preparation', 0.5);
    
    // Calculate crop dimensions for 9:16 aspect ratio
    const initialCropWidth = height ? Math.round(height * (9/16)) : 0;
    
    // Open the video file to get accurate properties
    const video = new cv.VideoCapture(tempInputPath);
    
    // Get video properties directly from the video
    const videoWidth = video.get(cv.CAP_PROP_FRAME_WIDTH);
    const videoHeight = video.get(cv.CAP_PROP_FRAME_HEIGHT);
    const videoFps = video.get(cv.CAP_PROP_FPS);
    const videoTotalFrames = video.get(cv.CAP_PROP_FRAME_COUNT);
    
    // Use video properties if metadata is missing
    const actualWidth = width || videoWidth;
    const actualHeight = height || videoHeight;
    const actualFps = fps || videoFps;
    const actualTotalFrames = Math.floor(videoTotalFrames);
    const actualCropWidth = Math.round(actualHeight * (9/16));
    
    // Process the entire video
    const startFrame = 0;
    const endFrame = actualTotalFrames - 1;
    
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
        : DEFAULT_CENTER;
      
      // Calculate crop parameters for this frame
      const xCenter = Math.floor(normCenter * actualWidth);
      let xStart = xCenter - Math.floor(actualCropWidth / 2);
      
      // Ensure crop stays within boundaries
      if (xStart < 0) {
        xStart = 0;
      } else if (xStart + actualCropWidth > actualWidth) {
        xStart = actualWidth - actualCropWidth;
      }
      
      // Convert to grayscale for scene change detection
      const grayFrame = frame.cvtColor(cv.COLOR_BGR2GRAY);
      
      // Calculate frame difference if we have a previous frame
      let frameDiff = 0;
      if (prevGrayFrame) {
        frameDiff = mse(grayFrame, prevGrayFrame);
        
        // Log scene changes
        if (frameDiff > SCENE_CHANGE_THRESHOLD) {
          console.log(`Job ID [${processingId}]: Scene change detected at frame ${currentFrame}`);
        }
      }
      
      // Store grayscale frame for next comparison
      prevGrayFrame = grayFrame;
      
      // Crop the frame
      const croppedFrame = frame.getRegion(new cv.Rect(xStart, 0, actualCropWidth, actualHeight));
      
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
      if (framePromises.length >= BATCH_SIZE) {
        await Promise.all(framePromises);
        framePromises.length = 0;
      }
      
      // Update progress
      processedFrames++;
      currentFrame++;
      
      // Report progress periodically
      const now = Date.now();
      if (now - lastProgressReport > 1000) {
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
    
    // Frame cropping complete, starting encoding
    updatePhaseProgress('frameCropping', 1);
    updatePhaseProgress('encoding', 0);
    
    // Use FFmpeg to combine frames into a video
    await new Promise((resolve, reject) => {
      let encodingProgress = 0;
      
      ffmpeg()
        .input(path.join(processingFramesDir, 'frame_%08d.png'))
        .inputOptions([
          '-framerate', actualFps.toString()
        ])
        .outputOptions([
          '-c:v', 'libx264',
          '-preset', 'medium', // Balance between speed and compression
          '-crf', '18',        // High quality (lower value = higher quality)
          '-pix_fmt', 'yuv420p'
        ])
        .output(tempOutputPath)
        .on('progress', (progress) => {
          // FFmpeg reports progress as a percentage
          if (progress.percent) {
            encodingProgress = progress.percent / 100;
            // Update encoding phase progress
            updatePhaseProgress('encoding', Math.min(1, encodingProgress));
          }
        })
        .on('end', resolve)
        .on('error', reject)
        .run();
    });
    
    // Encoding complete, starting audio merging
    updatePhaseProgress('encoding', 1);
    updatePhaseProgress('audioMerging', 0);
    
    // Create output directory if it doesn't exist
    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    // Add audio from the original video using FFmpeg
    await new Promise((resolve, reject) => {
      ffmpeg()
        .input(tempOutputPath)
        .input(tempInputPath)
        .outputOptions([
          '-c:v', 'copy',
          '-c:a', 'aac',
          '-map', '0:v:0',
          '-map', '1:a:0?',
          '-shortest'  // Ensures the output duration matches the video
        ])
        .output(outputPath)
        .on('progress', (progress) => {
          if (progress.percent) {
            const audioProgress = progress.percent / 100;
            // Update audio merging phase progress
            updatePhaseProgress('audioMerging', Math.min(1, audioProgress));
          }
        })
        .on('end', () => {
          // Update progress to 100% when audio is added and file is saved
          updatePhaseProgress('audioMerging', 1);
          updateProgress(processingId, {
            videoGenerated: true
          });
          
          resolve();
        })
        .on('error', (err) => {
          console.error(`Job ID [${processingId}]: Error adding audio: ${err.message}`);
          reject(err);
        })
        .run();
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
      duration: ((endFrame - startFrame + 1) / actualFps).toFixed(2),
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

// Export the processVideo function
module.exports = {
  processVideo,
  MovementPlanner,  // Export for testing or advanced usage
  generateRandomId,  // Export for reuse in other modules
  updateProgress     // Export for external use
}; 