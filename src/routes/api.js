const express = require('express');
const router = express.Router();
const fs = require('fs');
const http = require('http');
const https = require('https');
const path = require('path');
const url = require('url');
const videoAnalyzer = require('../services/analizeVideo');
const processVideo = require('../services/processVideo');
const multer = require('multer');
const temp = require('temp');

// Track and cleanup temp files
temp.track();

// Set up multer for file uploads
const upload = multer({ 
  dest: temp.dir,
  limits: { fileSize: 100 * 1024 * 1024 } // 100MB limit
});

// Helper function to check if a path exists
function pathExists(filePath) {
  return new Promise((resolve) => {
    fs.access(filePath, fs.constants.F_OK, (err) => {
      resolve(!err);
    });
  });
}

// Helper function to check if a URL is valid (returns 200)
function isUrlValid(urlString) {
  return new Promise((resolve) => {
    try {
      const parsedUrl = new URL(urlString);
      const protocol = parsedUrl.protocol === 'https:' ? https : http;
      
      const req = protocol.request({
        method: 'HEAD',
        host: parsedUrl.hostname,
        path: parsedUrl.pathname + parsedUrl.search
      }, (res) => {
        resolve(res.statusCode === 200);
      });
      
      req.on('error', () => {
        resolve(false);
      });
      
      req.end();
    } catch (error) {
      resolve(false);
    }
  });
}

// Helper function to validate input/output
async function validatePath(pathOrUrl) {
  // Check if it's a URL or file path
  if (pathOrUrl.startsWith('http://') || pathOrUrl.startsWith('https://')) {
    // It's a URL, check if it's valid
    const isValid = await isUrlValid(pathOrUrl);
    if (!isValid) {
      throw new Error(`URL not accessible or invalid: ${pathOrUrl}`);
    }
    return pathOrUrl;
  } else {
    // It's a file path, check if file exists
    // Handle file:/// protocol
    let filePath = pathOrUrl;
    if (pathOrUrl.startsWith('file:///')) {
      filePath = pathOrUrl.replace('file://', '');
    }
    
    const exists = await pathExists(filePath);
    if (!exists) {
      throw new Error(`File does not exist: ${filePath}`);
    }
    return filePath;
  }
}

// Root endpoint
router.get('/', (req, res) => {
  res.send('Welcome to QuickReels API!');
});

// Process reel endpoint
router.post('/process-reel', async (req, res) => {
  try {
    const { input, outputs, debug } = req.body;

    if (!input || !outputs || !Array.isArray(outputs)) {
      return res.status(400).send('Invalid request payload');
    }

    // Validate input
    try {
      await validatePath(input);
    } catch (error) {
      return res.status(400).json({
        error: `Invalid input: ${error.message}`
      });
    }

    // Validate all outputs
    for (const output of outputs) {
      if (!output.url) {
        return res.status(400).json({
          error: 'Missing URL in output'
        });
      }
      
      // For outputs, we validate the directory exists (not the file itself)
      try {
        // If it's a URL, we don't need to validate its directory
        if (!output.url.startsWith('http://') && !output.url.startsWith('https://')) {
          let outputPath = output.url;
          if (output.url.startsWith('file:///')) {
            outputPath = output.url.replace('file://', '');
          }
          
          // Check if the directory exists
          const directory = path.dirname(outputPath);
          const dirExists = await pathExists(directory);
          if (!dirExists) {
            throw new Error(`Output directory does not exist: ${directory}`);
          }
        }
      } catch (error) {
        return res.status(400).json({
          error: `Invalid output: ${error.message}`
        });
      }
    }

    // Log the received data for debugging
    console.log('Received JSON:', req.body);

    // Process the video using TensorFlow
    try {
      // Process the video with the unified function, passing debug option
      const result = await processVideo(input, outputs, { debug });
      
      // Return the results
      res.json({
        message: 'Video processing initiated',
        receivedData: req.body,
        analysis: result.analysis,
        processingResults: result.outputs
      });
    } catch (error) {
      console.error('Error processing video:', error);
      return res.status(500).json({
        error: `Error processing video: ${error.message}`
      });
    }
  } catch (error) {
    console.error('Error processing request:', error);
    res.status(500).json({
      error: 'Server error processing request'
    });
  }
});

// Process video endpoint - similar to Python's /process-video
router.post('/process-video', upload.single('video'), async (req, res) => {
  try {
    // Check if we have a file
    if (!req.file) {
      return res.status(400).json({ error: 'No video file provided' });
    }
    
    // Get segments from form data
    let segments = [];
    try {
      segments = JSON.parse(req.body.segments || '[]');
    } catch (error) {
      return res.status(400).json({ error: 'Invalid segments JSON' });
    }
    
    // Get debug mode from form data (match Python behavior exactly)
    const debug = req.body.debug === 'true';
    console.log('Debug mode enabled:', debug);
    
    console.log('Processing video file:', req.file.path);
    console.log('Segments:', segments);
    
    // Create temporary output files
    const outputFiles = [];
    const outputSegments = segments.map(segment => {
      // Create a temporary file for the output
      const tempOutput = temp.path({ suffix: '.mp4' });
      outputFiles.push(tempOutput);
      
      return {
        url: tempOutput,
        range: segment.start_frame && segment.end_frame ? 
          `[${segment.start_frame}-${segment.end_frame}]` : undefined,
        debug // Pass debug flag to segment
      };
    });
    
    // If no segments provided, process the whole video
    if (outputSegments.length === 0) {
      const tempOutput = temp.path({ suffix: '.mp4' });
      outputFiles.push(tempOutput);
      outputSegments.push({ 
        url: tempOutput, 
        debug // Pass debug flag here too
      });
    }
    
    try {
      // Process the video with the unified function
      console.log('Processing video with segments:', outputSegments);
      const result = await processVideo(req.file.path, outputSegments, { debug });
      
      // Read all processed videos
      const results = [];
      for (const segment of result.outputs) {
        const outputPath = segment.outputPath;
        const fileData = fs.readFileSync(outputPath);
        results.push(fileData);
      }
      
      // Return results
      res.json({
        videos: results.map(video => video.toString('base64'))
      });
      
      // Clean up temp files (handled by temp.track)
    } catch (error) {
      console.error('Error processing video:', error);
      // Clean up any created files that might not be tracked by temp
      outputFiles.forEach(file => {
        try { fs.unlinkSync(file); } catch (e) { /* ignore errors */ }
      });
      
      return res.status(500).json({
        error: `Error processing video: ${error.message}`
      });
    }
  } catch (error) {
    console.error('Error processing request:', error);
    res.status(500).json({
      error: 'Server error processing request'
    });
  }
});

module.exports = router; 