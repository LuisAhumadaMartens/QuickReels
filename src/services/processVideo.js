const videoAnalyzer = require('./analizeVideo');
const { VideoProcessor, VideoSegment, parseFrameRange } = require('./videoProcessor');
const path = require('path');
const os = require('os');
const fs = require('fs');
const ffmpeg = require('fluent-ffmpeg');
const tf = require('@tensorflow/tfjs-node');

// Constants to match Python implementation
const DEFAULT_CENTER = 0.5;  // normalized center (50%)
const ASPECT_RATIO = 9 / 16;  // output crop aspect ratio
const SCENE_CHANGE_THRESHOLD = 3000;

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
  
  // Create temporary directory in current folder rather than system temp
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
      
      // Group similar positions to reduce the number of segments
      const segments = [];
      let currentSegmentStart = 0;
      let currentPosition = smoothedPositions[0];
      
      // Create segments with similar positions (round to nearest 0.05)
      for (let i = 1; i < smoothedPositions.length; i++) {
        const roundedCurrent = Math.round(currentPosition * 20) / 20;
        const roundedNew = Math.round(smoothedPositions[i] * 20) / 20;
        
        // If position changed significantly, start a new segment
        if (roundedCurrent !== roundedNew || i === smoothedPositions.length - 1) {
          segments.push({
            startFrame: currentSegmentStart,
            endFrame: i - 1,
            position: currentPosition
          });
          
          currentSegmentStart = i;
          currentPosition = smoothedPositions[i];
        }
      }
      
      // Add the last segment if needed
      if (currentSegmentStart < smoothedPositions.length - 1) {
        segments.push({
          startFrame: currentSegmentStart,
          endFrame: smoothedPositions.length - 1,
          position: currentPosition
        });
      }
      
      console.log(`Created ${segments.length} segments with different crop positions`);
      
      // Process each segment and create temporary segment files
      const segmentFiles = [];
      
      for (let i = 0; i < segments.length; i++) {
        const segment = segments[i];
        const segmentFile = path.join(tempDir, `segment_${index}_${i}.mp4`);
        segmentFiles.push(segmentFile);
        
        // Calculate crop parameters for this segment
        const xCenter = Math.round(segment.position * width);
        let xStart = xCenter - Math.floor(cropWidth / 2);
        
        // Ensure crop stays within boundaries
        if (xStart < 0) xStart = 0;
        if (xStart + cropWidth > width) xStart = width - cropWidth;
        
        // Calculate start and end time in seconds
        const startTime = segment.startFrame / fps;
        const duration = (segment.endFrame - segment.startFrame + 1) / fps;
        
        console.log(`Segment ${i}: frames ${segment.startFrame}-${segment.endFrame}, position ${segment.position.toFixed(3)}, crop at x=${xStart}`);
        
        // Process this segment
        await new Promise((resolve, reject) => {
          ffmpeg(inputPath)
            .setStartTime(startTime)
            .setDuration(duration)
            .outputOptions([
              '-filter:v', `crop=${cropWidth}:${height}:${xStart}:0`,
              '-c:v', 'libx264',
              '-preset', 'medium',
              '-crf', '23',
              '-c:a', 'aac'
            ])
            .output(segmentFile)
            .on('end', resolve)
            .on('error', reject)
            .run();
        });
      }
      
      // Now concatenate all segments
      if (segmentFiles.length === 1) {
        // Just rename the file if there's only one segment
        fs.copyFileSync(segmentFiles[0], outputPath);
      } else {
        // Create a concat file
        const concatFile = path.join(tempDir, `concat_${index}.txt`);
        const concatContent = segmentFiles.map(file => `file '${file}'`).join('\n');
        fs.writeFileSync(concatFile, concatContent);
        
        // Concatenate all segments
        await new Promise((resolve, reject) => {
          ffmpeg()
            .input(concatFile)
            .inputOptions(['-f', 'concat', '-safe', '0'])
            .outputOptions([
              '-c', 'copy'
            ])
            .output(outputPath)
            .on('end', resolve)
            .on('error', reject)
            .run();
        });
      }
      
      return {
        outputPath: url,
        range: range,
        status: 'completed'
      };
    }));
    
    // Clean up temporary directory
    try {
      if (fs.existsSync(tempDir)) {
        fs.rmSync(tempDir, { recursive: true });
      }
    } catch (err) {
      console.warn('Warning: Failed to clean up temporary directory', err);
    }
    
    return results;
  } catch (error) {
    console.error('Error processing video:', error);
    
    // Clean up on error
    try {
      if (fs.existsSync(tempDir)) {
        fs.rmSync(tempDir, { recursive: true });
      }
    } catch (cleanupErr) {
      console.warn('Warning: Failed to clean up on error', cleanupErr);
    }
    
    throw error;
  }
}

// Export the processVideo function
module.exports = {
  processVideo,
  MovementPlanner  // Export for testing or advanced usage
}; 