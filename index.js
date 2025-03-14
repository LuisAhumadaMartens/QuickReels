const fs = require('fs');
const path = require('path');
const { analizeVideo } = require('./analizeVideo');
const { processVideo } = require('./processVideo');
const { VideoSegment, VideoProcessor } = require('./videoProcessor');

/**
 * QuickReels - Analyze and process videos to create vertical video crops
 * @param {string} inputPath - Path to input video file
 * @param {string} outputPath - Path to save the output video
 * @param {Object} options - Optional configuration
 * @returns {Promise<Object>} - Processing result
 */
async function quickReels(inputPath, outputPath, options = {}) {
  // Validate input path
  if (!inputPath || !fs.existsSync(inputPath)) {
    throw new Error(`Input file does not exist: ${inputPath}`);
  }

  // Validate output path
  const outputDir = path.dirname(outputPath);
  if (!fs.existsSync(outputDir)) {
    try {
      fs.mkdirSync(outputDir, { recursive: true });
    } catch (error) {
      throw new Error(`Could not create output directory: ${outputDir}`);
    }
  }

  try {
    console.log(`QuickReels: Processing ${inputPath} to ${outputPath}`);
    console.log('Step 1: Analyzing video...');
    
    // Step 1: Analyze the video to get camera tracking data
    const analysis = await analizeVideo(inputPath, outputPath, options);
    
    console.log('Analysis complete.');
    console.log('Step 2: Processing video...');
    
    // Step 2: Process the video based on analysis results
    const processResult = await processVideo(inputPath, outputPath, analysis);
    
    // Step 3: If custom segments are specified, process them using VideoProcessor
    if (options.segments && Array.isArray(options.segments) && options.segments.length > 0) {
      console.log('Step 3: Processing custom segments...');
      
      // Convert segment options to VideoSegment objects
      const videoSegments = options.segments.map(segmentOpt => {
        return new VideoSegment(
          segmentOpt.outputPath,
          segmentOpt.startFrame || null,
          segmentOpt.endFrame || null
        );
      });
      
      // Create video processor instance
      const processor = new VideoProcessor(inputPath, videoSegments);
      
      // Process all segments using the same analysis data
      const segmentResults = await processor.processAll(null, analysis);
      
      // Add segment results to the main result
      processResult.segments = segmentResults;
    }
    
    console.log('Processing complete.');
    return processResult;
  } catch (error) {
    console.error(`Error in QuickReels: ${error.message}`);
    throw error;
  }
}

module.exports = quickReels; 