const ffmpeg = require('fluent-ffmpeg');
const ffmpegPath = require('@ffmpeg-installer/ffmpeg').path;
const fs = require('fs');
const path = require('path');
const os = require('os');
const videoAnalyzer = require('./analizeVideo');

// Set the ffmpeg path
ffmpeg.setFfmpegPath(ffmpegPath);

// VideoSegment class similar to Python's dataclass
class VideoSegment {
  constructor(outputPath, startFrame = null, endFrame = null) {
    this.outputPath = outputPath;
    this.startFrame = startFrame;
    this.endFrame = endFrame;
  }
}

/**
 * Parse frame range from string, similar to Python implementation
 * @param {string} rangeStr - Frame range string in format "[start-end]"
 * @returns {Object} - {startFrame, endFrame} or null if invalid
 */
function parseFrameRange(rangeStr) {
  if (!rangeStr) return null;
  
  // Check format
  if (!rangeStr.startsWith('[') || !rangeStr.endsWith(']')) {
    throw new Error(`Frame range must be in format [start-end], got: ${rangeStr}`);
  }
  
  try {
    // Extract and parse numbers
    const [startFrame, endFrame] = rangeStr
      .substring(1, rangeStr.length - 1)
      .split('-')
      .map(s => parseInt(s.trim(), 10));
    
    return { startFrame, endFrame };
  } catch (error) {
    throw new Error(`Invalid frame range format: ${rangeStr}`);
  }
}

/**
 * Get video metadata using ffmpeg
 * @param {string} videoPath - Path to video file
 * @returns {Promise<Object>} - Video metadata
 */
function getVideoMetadata(videoPath) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(videoPath, (err, metadata) => {
      if (err) {
        reject(err);
        return;
      }
      
      const videoStream = metadata.streams.find(s => s.codec_type === 'video');
      if (!videoStream) {
        reject(new Error('No video stream found'));
        return;
      }
      
      // Calculate duration in seconds
      const duration = parseFloat(metadata.format.duration);
      
      // Handle variable frame rate by using the average
      let fps = eval(videoStream.r_frame_rate || videoStream.avg_frame_rate || '30/1');
      
      // Calculate total frames
      const totalFrames = Math.round(duration * fps);
      
      resolve({
        width: videoStream.width,
        height: videoStream.height,
        fps,
        duration,
        totalFrames,
        format: metadata.format.format_name
      });
    });
  });
}

/**
 * Extract frames from a video
 * @param {string} videoPath - Path to video file
 * @param {number} interval - Frame interval to extract
 * @param {number} maxFrames - Maximum number of frames to extract
 * @returns {Promise<string>} - Path to directory containing extracted frames
 */
async function extractFrames(videoPath, interval = 1, maxFrames = null) {
  const frameDir = path.join(os.tmpdir(), `quickreels_frames_${Date.now()}`);
  fs.mkdirSync(frameDir, { recursive: true });
  
  const metadata = await getVideoMetadata(videoPath);
  const frameCount = metadata.totalFrames;
  
  // Calculate how many frames to extract
  const framesToExtract = maxFrames ? 
    Math.min(Math.ceil(frameCount / interval), maxFrames) : 
    Math.ceil(frameCount / interval);
  
  return new Promise((resolve, reject) => {
    let extractedCount = 0;
    
    const command = ffmpeg(videoPath)
      .outputOptions([
        `-vf select='not(mod(n,${interval}))'`,
        '-vsync 0',
        '-q:v 2'
      ])
      .output(path.join(frameDir, 'frame-%04d.jpg'))
      .on('end', () => {
        console.log(`Extracted ${extractedCount} frames to ${frameDir}`);
        resolve(frameDir);
      })
      .on('error', (err) => {
        reject(err);
      })
      .on('progress', (progress) => {
        console.log(`Extracting frames: ${Math.round(progress.percent || 0)}%`);
      });
    
    if (maxFrames) {
      command.frames(maxFrames);
    }
    
    command.run();
  });
}

/**
 * Class for processing videos similar to the Python implementation
 */
class VideoProcessor {
  constructor(inputPath, segments) {
    this.inputPath = inputPath;
    this.segments = segments;
    this.metadata = null;
    this.personTracker = null;
    this.movementPlanner = null;
    this.jobId = null; // Adding jobId to match original implementation
  }
  
  async _initializeVideoMetadata() {
    this.metadata = await getVideoMetadata(this.inputPath);
    if (this.jobId) {
      console.log(`Job ID [${this.jobId}]: Video metadata initialized`);
    } else {
      console.log(`Video metadata initialized`);
    }
    this.personTracker = new videoAnalyzer.PersonTracker();
    this.movementPlanner = new videoAnalyzer.MovementPlanner(this.metadata.fps);
  }
  
  async processAll(jobId = null, analysis = null) {
    this.jobId = jobId;
    try {
      // Initialize video metadata first
      await this._initializeVideoMetadata();
      
      // If analysis is provided, store it for later use
      if (analysis) {
        this._analysisResults = analysis;
        console.log(`Using provided analysis data`);
      } else {
        // First pass: analyze the entire video once
        console.log(`Starting video analysis`);
        
        // Get the analysis with frame interval
        const newAnalysis = await videoAnalyzer.analizeVideo(this.inputPath, null);
        
        // Store the analysis results for later use
        this._analysisResults = newAnalysis;
        
        console.log(`Video analysis complete`);
      }
      
      // Use the smoothedPositions from analysis
      const { smoothedPositions } = this._analysisResults;
      
      if (!smoothedPositions || smoothedPositions.length === 0) {
        if (this.jobId) {
          console.log(`Job ID [${this.jobId}]: WARNING: Analysis did not produce smoothed positions, using center position`);
        } else {
          console.log(`WARNING: Analysis did not produce smoothed positions, using center position`);
        }
        // Create default positions (center)
        const centerPositions = Array(this.metadata.totalFrames).fill(0.5);
        smoothedPositions = centerPositions;
      }
      
      // Second pass: Process each segment
      if (this.jobId) {
        console.log(`Job ID [${this.jobId}]: Processing ${this.segments.length} segments with ${smoothedPositions.length} position keyframes`);
      } else {
        console.log(`Processing ${this.segments.length} segments with ${smoothedPositions.length} position keyframes`);
      }
      
      // Add timeout protection
      const MAX_PROCESSING_TIME = 300000; // 5 minutes in milliseconds
      const startTime = Date.now();
      
      // Process each segment with a time limit
      for (const segment of this.segments) {
        if (this.jobId) {
          console.log(`Job ID [${this.jobId}]: Processing segment: ${segment.outputPath} [${segment.startFrame || 0}-${segment.endFrame || 'end'}]`);
        } else {
          console.log(`Processing segment: ${segment.outputPath} [${segment.startFrame || 0}-${segment.endFrame || 'end'}]`);
        }
        
        try {
          // Check if we've been processing too long
          if (Date.now() - startTime > MAX_PROCESSING_TIME) {
            if (this.jobId) {
              console.log(`Job ID [${this.jobId}]: WARNING: Processing time limit exceeded, stopping further processing`);
            } else {
              console.log(`WARNING: Processing time limit exceeded, stopping further processing`);
            }
            break;
          }
          
          await this._processSegment(segment, smoothedPositions, analysis);
        } catch (segmentError) {
          if (this.jobId) {
            console.error(`Job ID [${this.jobId}]: Error processing segment ${segment.outputPath}: ${segmentError.message}`);
          } else {
            console.error(`Error processing segment ${segment.outputPath}: ${segmentError.message}`);
          }
          // Continue with next segment
        }
      }
      
      return {
        inputPath: this.inputPath,
        segments: this.segments,
        duration: this.metadata.duration
      };
    } catch (error) {
      if (this.jobId) {
        console.error(`Job ID [${this.jobId}]: Error processing video: ${error.message}`);
      } else {
        console.error("Error processing video:", error);
      }
      throw error;
    }
  }
  
  async _processSegment(segment, smoothedPositions, analysis = null) {
    console.log(`Processing segment: ${segment.outputPath}`);
    
    // Ensure metadata is initialized
    if (!this.metadata) {
      console.error('Error: Video metadata not initialized');
      throw new Error('Video metadata not initialized. Call _initializeVideoMetadata first.');
    }
    
    // Use analysis data if provided, otherwise use stored analysis results
    const analysisToUse = analysis || this._analysisResults;
    
    // If analysis data is available, use its smoothedPositions
    const positionsToUse = analysisToUse && analysisToUse.smoothedPositions 
      ? analysisToUse.smoothedPositions 
      : smoothedPositions;
    
    // Determine frame range with safeguards
    const totalFrames = this.metadata.totalFrames || 0;
    const startFrame = segment.startFrame || 0;
    let endFrame = segment.endFrame || totalFrames - 1;
    
    if (endFrame <= startFrame) {
      if (this.jobId) {
        console.log(`Job ID [${this.jobId}]: WARNING: Invalid frame range ${startFrame}-${endFrame}, using default`);
      } else {
        console.log(`WARNING: Invalid frame range ${startFrame}-${endFrame}, using default`);
      }
      // Use a reasonable default - 5 second clip
      endFrame = Math.min(startFrame + (5 * this.metadata.fps), totalFrames - 1);
    }
    
    // Calculate time values with safeguards
    const fps = this.metadata.fps || 30; // Default to 30fps if not available
    const startTime = startFrame / fps;
    const duration = (endFrame - startFrame) / fps;
    
    if (this.jobId) {
      console.log(`Job ID [${this.jobId}]: Segment timing: ${startTime}s to ${startTime + duration}s (duration: ${duration}s)`);
    } else {
      console.log(`Segment timing: ${startTime}s to ${startTime + duration}s (duration: ${duration}s)`);
    }
    
    // Get relevant smoothed positions for this segment (with bounds checking)
    let segmentPositions = [];
    if (Array.isArray(positionsToUse) && positionsToUse.length > 0) {
      const startIdx = Math.min(startFrame, positionsToUse.length - 1);
      const endIdx = Math.min(endFrame + 1, positionsToUse.length);
      segmentPositions = positionsToUse.slice(startIdx, endIdx);
      
      // If we couldn't get enough positions, pad with the last position
      if (segmentPositions.length < (endFrame - startFrame + 1) && segmentPositions.length > 0) {
        const lastPos = segmentPositions[segmentPositions.length - 1];
        const needed = (endFrame - startFrame + 1) - segmentPositions.length;
        segmentPositions = segmentPositions.concat(Array(needed).fill(lastPos));
      }
    }
    
    if (segmentPositions.length === 0) {
      if (this.jobId) {
        console.log(`Job ID [${this.jobId}]: No position data available, using center position`);
      } else {
        console.log(`No position data available, using center position`);
      }
      segmentPositions = Array(endFrame - startFrame + 1).fill(0.5);
    }
    
    // Generate the filter string for ffmpeg
    const filterScript = this._generateFilterScript(segmentPositions);
    
    return new Promise((resolve, reject) => {
      ffmpeg(this.inputPath)
        .setStartTime(startTime)
        .duration(duration)
        .videoFilter(filterScript) // Use videoFilter instead of complexFilter
        .outputOptions([
          '-c:v libx264',   // Use H.264 codec
          '-pix_fmt yuv420p', // Compatible pixel format for most players
          '-preset fast',   // Encoding preset
          '-crf 23'         // Quality level
        ])
        .output(segment.outputPath)
        .on('start', (commandLine) => {
          if (this.jobId) {
            console.log(`Job ID [${this.jobId}]: FFmpeg started`);
          } else {
            console.log(`FFmpeg started`);
          }
        })
        .on('progress', (progress) => {
          if (this.jobId) {
            console.log(`Job ID [${this.jobId}]: Processing ${segment.outputPath}: ${Math.floor(progress.percent)}% done`);
          } else {
            console.log(`Processing ${segment.outputPath}: ${Math.floor(progress.percent)}% done`);
          }
        })
        .on('end', () => {
          if (this.jobId) {
            console.log(`Job ID [${this.jobId}]: Successfully processed segment: ${segment.outputPath}`);
          } else {
            console.log(`Successfully processed segment: ${segment.outputPath}`);
          }
          resolve();
        })
        .on('error', (err) => {
          if (this.jobId) {
            console.error(`Job ID [${this.jobId}]: Error processing segment: ${err.message}`);
          } else {
            console.error(`Error processing segment: ${err.message}`);
          }
          reject(err);
        })
        .run();
    });
  }
  
  /**
   * Calculate crop dimensions for 9:16 aspect ratio
   * @returns {Object} Object containing cropWidth, videoWidth, videoHeight
   * @private
   */
  _calculateCropDimensions() {
    const targetAspect = 9 / 16;  // vertical video aspect ratio
    const videoWidth = this.metadata.width;
    const videoHeight = this.metadata.height;
    const cropWidth = Math.floor(videoHeight * targetAspect);
    
    return { targetAspect, cropWidth, videoWidth, videoHeight };
  }
  
  /**
   * Ensure position is within video bounds
   * @param {number} xPos - X position
   * @param {number} cropWidth - Width of crop window
   * @param {number} videoWidth - Width of video
   * @returns {number} Adjusted position
   * @private
   */
  _ensurePositionInBounds(xPos, cropWidth, videoWidth) {
    if (xPos < 0) xPos = 0;
    if (xPos + cropWidth > videoWidth) xPos = videoWidth - cropWidth;
    return Math.floor(xPos);
  }
  
  /**
   * Calculate average position from an array of positions
   * @param {Array<number>} positions - Array of normalized positions (0-1)
   * @returns {number} Average position
   * @private
   */
  _calculateAveragePosition(positions) {
    return positions.reduce((sum, pos) => sum + pos, 0) / positions.length;
  }
  
  /**
   * Generate fallback filter for center position
   * @param {number} cropWidth - Width of crop window
   * @param {number} videoWidth - Width of video
   * @param {number} videoHeight - Height of video
   * @returns {string} FFmpeg filter string
   * @private
   */
  _generateCenterPositionFilter(cropWidth, videoWidth, videoHeight) {
    const centerX = Math.floor((videoWidth - cropWidth) / 2);
    return `crop=${cropWidth}:${videoHeight}:${centerX}:0,scale=1080:1920`;
  }
  
  /**
   * Generate average position filter 
   * @param {Array<number>} positions - Array of normalized positions (0-1)
   * @param {number} cropWidth - Width of crop window
   * @param {number} videoWidth - Width of video
   * @param {number} videoHeight - Height of video
   * @returns {string} FFmpeg filter string
   * @private
   */
  _generateAveragePositionFilter(positions, cropWidth, videoWidth, videoHeight) {
    const avgPos = this._calculateAveragePosition(positions);
    const xCenter = Math.floor(avgPos * videoWidth);
    let xPos = xCenter - (cropWidth / 2);
    
    xPos = this._ensurePositionInBounds(xPos, cropWidth, videoWidth);
    
    return `crop=${cropWidth}:${videoHeight}:${xPos}:0,scale=1080:1920`;
  }
  
  _generateFilterScript(positions) {
    // Use the utility method to calculate dimensions
    const { cropWidth, videoWidth, videoHeight } = this._calculateCropDimensions();
    
    // Check if we have valid positions or use center as fallback
    if (!positions || positions.length === 0) {
      return this._generateCenterPositionFilter(cropWidth, videoWidth, videoHeight);
    }
    
    // Generate a complex filter with multiple crops and timeline positions
    // For each frame, we need to calculate the x position based on normalized coordinate
    const cropCommands = positions.map((pos, index) => {
      // Convert normalized position (0-1) to pixel position
      // Ensuring the frame stays within video bounds
      const xCenter = Math.floor(pos * videoWidth);
      let xPos = xCenter - (cropWidth / 2);
      
      // Ensure we don't go outside the frame boundaries
      xPos = this._ensurePositionInBounds(xPos, cropWidth, videoWidth);
      
      return `crop=${cropWidth}:${videoHeight}:${xPos}:0`;
    });
    
    // For a large number of positions, use a simpler approach
    // FFmpeg can't handle thousands of filter commands efficiently
    if (cropCommands.length > 10) {
      return this._generateAveragePositionFilter(positions, cropWidth, videoWidth, videoHeight);
    }
    
    // For small number of positions (e.g. during testing), 
    // we can use a dynamic x position based on frame index
    // We'll create a complex filter that uses 'sendcmd' to update crop position
    
    // Create a command file string with position changes
    let cmdFileContent = '';
    positions.forEach((pos, index) => {
      const xCenter = Math.floor(pos * videoWidth);
      let xPos = xCenter - (cropWidth / 2);
      
      // Ensure we don't go outside the frame boundaries
      xPos = this._ensurePositionInBounds(xPos, cropWidth, videoWidth);
      
      // Convert frame index to timestamp (assuming constant frame rate)
      const timestamp = index / this.metadata.fps;
      cmdFileContent += `${timestamp} crop x ${xPos}\n`;
    });
    
    // Use sendcmd filter with crop
    const filter = `sendcmd=c='${cmdFileContent.replace(/'/g, "\''")}',` +
                   `crop=${cropWidth}:${videoHeight}:0:0,scale=1080:1920`;
    
    // For very long videos or many positions, this approach may not work
    // In that case, we'll fall back to the average position
    if (cmdFileContent.length > 10000) {
      return this._generateAveragePositionFilter(positions, cropWidth, videoWidth, videoHeight);
    }
    
    return filter;
  }
}

module.exports = {
  VideoSegment,
  VideoProcessor,
  parseFrameRange,
  getVideoMetadata,
  extractFrames
}; 