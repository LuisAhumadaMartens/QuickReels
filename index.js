const fs = require('fs');
const path = require('path');
const { analyzeVideo } = require('./analizeVideo');
const { processVideo } = require('./processVideo');

/**
 * QuickReels - Analyze and process videos to create vertical video crops
 * @param {string} inputPath - Path to input video file
 * @param {string} outputPath - Path to save the output video
 * @returns {Promise<Object>} - Processing result
 */
async function quickReels(inputPath, outputPath) {
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
    console.log(`QuickReels: Processing ${inputPath}`);

    console.log('Analyzing video using MoveNet...');
    const analysis = await analyzeVideo(inputPath, outputPath);
    console.log('Analysis complete.');

    console.log('Processing video by cropping frames...');
    const processResult = await processVideo(inputPath, outputPath, analysis);
    console.log('Processing complete.');

    console.log(`The output file was saved in ${outputPath}`);
    return processResult;
    
  } catch (error) {
    console.error(`Error: ${error.message}`);
    throw error;
  }
}

module.exports = quickReels; 