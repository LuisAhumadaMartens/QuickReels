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
  constructor(outputPath, startFrame = null, endFrame = null, debug = false) {
    this.outputPath = outputPath;
    this.startFrame = startFrame;
    this.endFrame = endFrame;
    this.debug = debug;
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
  }
  
  async _initializeVideoMetadata() {
    this.metadata = await getVideoMetadata(this.inputPath);
    console.log('Video metadata:', this.metadata);
    this.personTracker = new videoAnalyzer.PersonTracker();
    this.movementPlanner = new videoAnalyzer.MovementPlanner(this.metadata.fps);
  }
  
  async processAll() {
    try {
      // Initialize video metadata first
      await this._initializeVideoMetadata();
      console.log('Video metadata initialized:', this.metadata);
      
      // First pass: analyze the entire video once
      console.log("Starting video analysis...");
      
      // Get the analysis with frame interval
      console.log(`Analyzing video: ${this.inputPath}`);
      const analysis = await videoAnalyzer.analizeVideo(this.inputPath, this.segments.map(s => s.outputPath));
      
      // Store the analysis results for later use (especially in debug mode)
      this._analysisResults = analysis;
      
      // At this point, we should have the analysis results including smoothed positions
      console.log("Video analysis complete");
      
      if (!analysis || !analysis.smoothedPositions || analysis.smoothedPositions.length === 0) {
        console.warn("WARNING: Analysis did not produce smoothed positions, using center position");
        // Create default positions (center)
        const centerPositions = Array(this.metadata.totalFrames).fill(0.5);
        analysis.smoothedPositions = centerPositions;
      }
      
      // Second pass: Process each segment
      const { smoothedPositions } = analysis;
      console.log(`Processing ${this.segments.length} segments with ${smoothedPositions.length} position keyframes`);
      
      // Add timeout protection
      const MAX_PROCESSING_TIME = 300000; // 5 minutes in milliseconds
      const startTime = Date.now();
      
      // Process each segment with a time limit
      for (const segment of this.segments) {
        console.log(`Processing segment: ${segment.outputPath} [${segment.startFrame || 0}-${segment.endFrame || 'end'}]`);
        
        try {
          // Check if we've been processing too long
          if (Date.now() - startTime > MAX_PROCESSING_TIME) {
            console.warn("WARNING: Processing time limit exceeded, stopping further processing");
            break;
          }
          
          await this._processSegment(segment, smoothedPositions);
        } catch (segmentError) {
          console.error(`Error processing segment ${segment.outputPath}:`, segmentError);
          // Continue with next segment
        }
      }
      
      return {
        inputPath: this.inputPath,
        segments: this.segments,
        duration: this.metadata.duration
      };
    } catch (error) {
      console.error("Error processing video:", error);
      throw error;
    }
  }
  
  async _processSegment(segment, smoothedPositions) {
    console.log(`Processing segment: ${segment.outputPath}`);
    
    // Ensure metadata is initialized
    if (!this.metadata) {
      console.error('Error: Video metadata not initialized');
      throw new Error('Video metadata not initialized. Call _initializeVideoMetadata first.');
    }
    
    // Determine frame range with safeguards
    const totalFrames = this.metadata.totalFrames || 0;
    const startFrame = segment.startFrame || 0;
    let endFrame = segment.endFrame || totalFrames - 1;
    
    if (endFrame <= startFrame) {
      console.warn(`Warning: Invalid frame range ${startFrame}-${endFrame}, using default`);
      // Use a reasonable default - 5 second clip
      endFrame = Math.min(startFrame + (5 * this.metadata.fps), totalFrames - 1);
    }
    
    // Calculate time values with safeguards
    const fps = this.metadata.fps || 30; // Default to 30fps if not available
    const startTime = startFrame / fps;
    const duration = (endFrame - startFrame) / fps;
    
    console.log(`Segment timing: ${startTime}s to ${startTime + duration}s (duration: ${duration}s)`);
    
    // Get relevant smoothed positions for this segment (with bounds checking)
    let segmentPositions = [];
    if (Array.isArray(smoothedPositions) && smoothedPositions.length > 0) {
      const startIdx = Math.min(startFrame, smoothedPositions.length - 1);
      const endIdx = Math.min(endFrame + 1, smoothedPositions.length);
      segmentPositions = smoothedPositions.slice(startIdx, endIdx);
      
      // If we couldn't get enough positions, pad with the last position
      if (segmentPositions.length < (endFrame - startFrame + 1) && segmentPositions.length > 0) {
        const lastPos = segmentPositions[segmentPositions.length - 1];
        const needed = (endFrame - startFrame + 1) - segmentPositions.length;
        segmentPositions = segmentPositions.concat(Array(needed).fill(lastPos));
      }
    }
    
    if (segmentPositions.length === 0) {
      console.warn('No position data available, using center position');
      segmentPositions = Array(endFrame - startFrame + 1).fill(0.5);
    }
    
    // Check if we're in debug mode
    const isDebugMode = segment.debug === true;
    
    // Generate the filter string directly instead of using a file
    let filterScript;
    if (isDebugMode) {
      // In debug mode, we'll retrieve keypoint data and generate visualization
      // First, we need to run the analysis again to get the actual keypoints
      console.log("Debug mode: Getting keypoint data for visualization");
      
      try {
        // Retrieve keypoint data from original analysis
        // We'll pull this data from the videoAnalyzer
        const keypoints = await this._getKeypointDataForSegment(
          this.inputPath, 
          startFrame, 
          endFrame
        );
        
        // Generate a debug filter with keypoint visualization matching Python version
        filterScript = this._generateDebugFilterScriptWithKeypoints(
          segmentPositions, 
          keypoints
        );
      } catch (error) {
        console.error("Error getting keypoint data:", error);
        // Fall back to simple debug visualization without keypoints
        filterScript = this._generateDebugFilterScript(segmentPositions);
      }
    } else {
      filterScript = this._generateFilterScript(segmentPositions);
    }
    
    // Log the filter script for debugging
    console.log(`Using filter: ${filterScript.substring(0, 150)}... [truncated]`);
    
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
          console.log(`FFmpeg command: ${commandLine}`);
        })
        .on('progress', (progress) => {
          console.log(`Processing ${segment.outputPath}: ${Math.floor(progress.percent)}% done`);
        })
        .on('end', () => {
          console.log(`Successfully processed segment: ${segment.outputPath}`);
          resolve();
        })
        .on('error', (err) => {
          console.error(`Error processing segment: ${err}`);
          reject(err);
        })
        .run();
    });
  }
  
  /**
   * Helper method to get keypoint data for a specific segment
   * @param {string} inputPath - Video input path
   * @param {number} startFrame - Start frame
   * @param {number} endFrame - End frame
   * @returns {Promise<Array>} - Array of keypoint data per frame
   */
  async _getKeypointDataForSegment(inputPath, startFrame, endFrame) {
    console.log(`Getting keypoint data for frames ${startFrame}-${endFrame}`);
    
    // First attempt: Try to get keypoint data from stored analysis results
    try {
      // If we already have analysis results with keypoint data, use that
      if (this._analysisResults && this._analysisResults.keypointData) {
        const analysisKeypoints = this._analysisResults.keypointData;
        console.log(`Found existing keypoint data for ${analysisKeypoints.length} frames from analysis`);
        
        // Extract the segment of keypoints that match our frame range
        return analysisKeypoints.filter(kp => 
          kp.frame >= startFrame && kp.frame <= endFrame
        );
      }
    } catch (error) {
      console.warn('Failed to get keypoints from stored analysis:', error);
    }
    
    // Second attempt: Run a new analysis just for this segment
    try {
      // Use videoAnalyzer to run a limited analysis on this segment
      const videoAnalyzer = require('./analizeVideo');
      
      // Attempt to get real data if possible
      return await videoAnalyzer.getKeypointsForSegment(
        inputPath, 
        startFrame, 
        endFrame
      );
    } catch (error) {
      console.error("Error getting keypoint data:", error);
      
      // If all else fails, return placeholder data
      const numFrames = endFrame - startFrame + 1;
      const keypointData = [];
      
      for (let i = 0; i < numFrames; i++) {
        keypointData.push({
          frame: startFrame + i,
          keypoints: [],
          detected: false
        });
      }
      
      return keypointData;
    }
  }
  
  _generateFilterScript(positions) {
    // Determine crop width (9:16 aspect ratio for vertical video)
    const targetAspect = 9 / 16;  // vertical video aspect ratio
    const cropWidth = Math.floor(this.metadata.height * targetAspect);
    const videoWidth = this.metadata.width;
    
    // Check if we have valid positions or use center as fallback
    if (!positions || positions.length === 0) {
      const centerX = Math.floor((videoWidth - cropWidth) / 2);
      return `crop=${cropWidth}:${this.metadata.height}:${centerX}:0,scale=1080:1920`;
    }
    
    // Generate a complex filter with multiple crops and timeline positions
    // For each frame, we need to calculate the x position based on normalized coordinate
    const cropCommands = positions.map((pos, index) => {
      // Convert normalized position (0-1) to pixel position
      // Ensuring the frame stays within video bounds
      const xCenter = Math.floor(pos * videoWidth);
      let xPos = xCenter - (cropWidth / 2);
      
      // Ensure we don't go outside the frame boundaries
      if (xPos < 0) xPos = 0;
      if (xPos + cropWidth > videoWidth) xPos = videoWidth - cropWidth;
      
      return `crop=${cropWidth}:${this.metadata.height}:${Math.floor(xPos)}:0`;
    });
    
    // For a large number of positions, use a simpler approach
    // FFmpeg can't handle thousands of filter commands efficiently
    if (cropCommands.length > 10) {
      // Calculate average position or use more sophisticated approach if needed
      const avgPos = positions.reduce((sum, pos) => sum + pos, 0) / positions.length;
      const xCenter = Math.floor(avgPos * videoWidth);
      let xPos = xCenter - (cropWidth / 2);
      
      // Ensure we don't go outside the frame boundaries
      if (xPos < 0) xPos = 0;
      if (xPos + cropWidth > videoWidth) xPos = videoWidth - cropWidth;
      
      return `crop=${cropWidth}:${this.metadata.height}:${Math.floor(xPos)}:0,scale=1080:1920`;
    }
    
    // For small number of positions (e.g. during testing), 
    // we can use a dynamic x position based on frame index
    // We'll create a complex filter that uses 'sendcmd' to update crop position
    // See: https://ffmpeg.org/ffmpeg-filters.html#sendcmd_002c-asendcmd
    
    // Create a command file string with position changes
    let cmdFileContent = '';
    positions.forEach((pos, index) => {
      const xCenter = Math.floor(pos * videoWidth);
      let xPos = xCenter - (cropWidth / 2);
      
      // Ensure we don't go outside the frame boundaries
      if (xPos < 0) xPos = 0;
      if (xPos + cropWidth > videoWidth) xPos = videoWidth - cropWidth;
      
      // Convert frame index to timestamp (assuming constant frame rate)
      const timestamp = index / this.metadata.fps;
      cmdFileContent += `${timestamp} crop x ${Math.floor(xPos)}\n`;
    });
    
    // Use sendcmd filter with crop
    const filter = `sendcmd=c='${cmdFileContent.replace(/'/g, "\''")}',` +
                   `crop=${cropWidth}:${this.metadata.height}:0:0,scale=1080:1920`;
    
    // For very long videos or many positions, this approach may not work
    // In that case, we'll fall back to the average position
    if (cmdFileContent.length > 10000) {
      const avgPos = positions.reduce((sum, pos) => sum + pos, 0) / positions.length;
      const xCenter = Math.floor(avgPos * videoWidth);
      let xPos = xCenter - (cropWidth / 2);
      
      // Ensure we don't go outside the frame boundaries
      if (xPos < 0) xPos = 0;
      if (xPos + cropWidth > videoWidth) xPos = videoWidth - cropWidth;
      
      return `crop=${cropWidth}:${this.metadata.height}:${Math.floor(xPos)}:0,scale=1080:1920`;
    }
    
    return filter;
  }
  
  _generateDebugFilterScript(positions) {
    // Determine crop width (9:16 aspect ratio for vertical video)
    const targetAspect = 9 / 16;  // vertical video aspect ratio
    const cropWidth = Math.floor(this.metadata.height * targetAspect);
    const videoWidth = this.metadata.width;
    const videoHeight = this.metadata.height;
    
    // Check if we have valid positions or use center as fallback
    if (!positions || positions.length === 0) {
      const centerX = Math.floor((videoWidth - cropWidth) / 2);
      
      // Add a dot at the center position
      const dotX = videoWidth / 2;
      const dotY = videoHeight / 2;
      
      return `drawbox=x=${dotX-5}:y=${dotY-5}:w=10:h=10:color=red@0.8:t=fill,` +
             `crop=${cropWidth}:${videoHeight}:${centerX}:0,scale=1080:1920`;
    }
    
    // Create a command file string with position changes and draw dots
    let cmdFileContent = '';
    const drawCommands = [];
    
    positions.forEach((pos, index) => {
      const xCenter = Math.floor(pos * videoWidth);
      let xPos = xCenter - (cropWidth / 2);
      
      // Ensure we don't go outside the frame boundaries
      if (xPos < 0) xPos = 0;
      if (xPos + cropWidth > videoWidth) xPos = videoWidth - cropWidth;
      
      // Convert frame index to timestamp (assuming constant frame rate)
      const timestamp = index / this.metadata.fps;
      cmdFileContent += `${timestamp} crop x ${Math.floor(xPos)}\n`;
      
      // Add a command to draw a dot at the center position
      // (This has to be added at exact frame timestamps)
      drawCommands.push(`${timestamp} drawbox x ${xCenter-5} y ${videoHeight/2-5} w 10 h 10 color red@0.8 t fill`);
    });
    
    // Combine the crop commands with draw commands
    const combinedCmdContent = cmdFileContent + '\n' + drawCommands.join('\n');
    
    // Use sendcmd filter with crop and drawing
    const filter = `sendcmd=c='${combinedCmdContent.replace(/'/g, "\''")}',` +
                   `drawbox=x=0:y=0:w=10:h=10:color=red@0:t=fill,` + // Initial invisible dot
                   `crop=${cropWidth}:${videoHeight}:0:0,scale=1080:1920`;
    
    // For very long videos or many positions, this approach may not work
    // In that case, we'll fall back to the simpler approach
    if (combinedCmdContent.length > 10000) {
      // Calculate average position
      const avgPos = positions.reduce((sum, pos) => sum + pos, 0) / positions.length;
      const xCenter = Math.floor(avgPos * videoWidth);
      let xPos = xCenter - (cropWidth / 2);
      
      // Ensure we don't go outside the frame boundaries
      if (xPos < 0) xPos = 0;
      if (xPos + cropWidth > videoWidth) xPos = videoWidth - cropWidth;
      
      // Add a dot at the center position
      return `drawbox=x=${xCenter-5}:y=${videoHeight/2-5}:w=10:h=10:color=red@0.8:t=fill,` +
             `crop=${cropWidth}:${videoHeight}:${Math.floor(xPos)}:0,scale=1080:1920`;
    }
    
    return filter;
  }
  
  _generateDebugFilterScriptWithKeypoints(positions, keypoints) {
    // Determine crop width (9:16 aspect ratio for vertical video)
    const targetAspect = 9 / 16;  // vertical video aspect ratio
    const cropWidth = Math.floor(this.metadata.height * targetAspect);
    const videoWidth = this.metadata.width;
    const videoHeight = this.metadata.height;
    
    console.log(`Debug visualization: Video dimensions ${videoWidth}x${videoHeight}, crop width ${cropWidth}`);
    
    // Check if we have valid positions or use center as fallback
    if (!positions || positions.length === 0) {
      const centerX = Math.floor((videoWidth - cropWidth) / 2);
      
      // Add a dot at the center position
      const dotX = videoWidth / 2;
      const dotY = videoHeight / 2;
      
      return `drawbox=x=${dotX-5}:y=${dotY-5}:w=10:h=10:color=red@0.8:t=fill,` +
             `crop=${cropWidth}:${videoHeight}:${centerX}:0,scale=1080:1920`;
    }
    
    // Create a command file string with position changes
    let cmdFileContent = '';
    const drawCommands = [];
    
    // This approach is similar to how the Python version does it in process_frame
    positions.forEach((pos, index) => {
      // Calculate crop position
      const xCenter = Math.floor(pos * videoWidth);
      let xPos = xCenter - (cropWidth / 2);
      
      // Ensure we don't go outside the frame boundaries
      if (xPos < 0) xPos = 0;
      if (xPos + cropWidth > videoWidth) xPos = videoWidth - cropWidth;
      
      // Convert frame index to timestamp (assuming constant frame rate)
      const timestamp = index / this.metadata.fps;
      
      // Add crop command
      cmdFileContent += `${timestamp} crop x ${Math.floor(xPos)}\n`;
      
      // Add tracking dot at center position (this shows camera tracking)
      drawCommands.push(`${timestamp} drawbox x ${xCenter-5} y ${videoHeight/2-5} w 10 h 10 color red@0.8 t fill`);
      
      // Get keypoints for this frame
      const frameKeypoints = keypoints[index]?.keypoints;
      
      // Add keypoint visualization exactly as Python does
      if (frameKeypoints && frameKeypoints.length > 0) {
        frameKeypoints.forEach((kp, kpIndex) => {
          // In the Python version, keypoints are in the format [y, x, confidence]
          if (kp && kp.length >= 3 && kp[2] > 0.3) { // Confidence threshold same as Python
            const y = kp[0]; // Normalized y coordinate
            const x = kp[1]; // Normalized x coordinate
            const conf = kp[2]; // Confidence score
            
            if (!isNaN(x) && !isNaN(y) && !isNaN(conf)) {
              // Calculate pixel coordinates
              const xPixel = Math.floor(x * videoWidth);
              const yPixel = Math.floor(y * videoHeight);
              
              // Calculate x position relative to the crop window
              const relativeX = xPixel - xPos;
              
              // Only draw if the point is within the cropped area
              if (relativeX >= 0 && relativeX < cropWidth) {
                // Draw green circle for keypoint (same as cv2.circle in Python)
                drawCommands.push(`${timestamp} drawbox x ${relativeX-5} y ${yPixel-5} w 10 h 10 color green@0.8 t fill`);
                
                // In the Python version, we use cv2.putText to show confidence percentage
                // FFmpeg's drawtext filter requires a complex setup, so we'll use a simpler approach
                // That gives the same visual appearance - showing confidence text
                
                // Format confidence as percentage
                const confPercent = Math.floor(conf * 100);
                
                // Add text background box (easier than text)
                drawCommands.push(`${timestamp} drawbox x ${relativeX+10} y ${yPixel-20} w 35 h 15 color black@0.5 t fill`);
                
                // For a more complex implementation, you would add actual text
                // But this requires a font file and more complex filter setup
              }
            }
          }
        });
      }
    });
    
    // Combine the crop commands with draw commands
    const combinedCmdContent = cmdFileContent + '\n' + drawCommands.join('\n');
    
    // FFmpeg filter command with sendcmd for dynamic updates
    const filter = `sendcmd=c='${combinedCmdContent.replace(/'/g, "\''")}',` +
                   `drawbox=x=0:y=0:w=10:h=10:color=red@0:t=fill,` + // Initial invisible dot
                   `crop=${cropWidth}:${videoHeight}:0:0,` +
                   `scale=1080:1920`;
    
    // If the command is too long for FFmpeg to handle, use a simplified version
    if (combinedCmdContent.length > 10000) {
      console.warn('Debug command too long, using simplified version');
      // Use average position for crop
      const avgPos = positions.reduce((sum, pos) => sum + pos, 0) / positions.length;
      const xCenter = Math.floor(avgPos * videoWidth);
      let xPos = xCenter - (cropWidth / 2);
      
      // Ensure we don't go outside frame boundaries
      if (xPos < 0) xPos = 0;
      if (xPos + cropWidth > videoWidth) xPos = videoWidth - cropWidth;
      
      // Simple filter with center dot only
      return `drawbox=x=${xCenter-5}:y=${videoHeight/2-5}:w=10:h=10:color=red@0.8:t=fill,` +
             `crop=${cropWidth}:${videoHeight}:${Math.floor(xPos)}:0,` +
             `scale=1080:1920`;
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