// Node.js Core Modules
const express = require('express');
const fs = require('fs');
const http = require('http');
const https = require('https');
const path = require('path');


// Modules
const config = require('./src/config/config');
const portManager = require('./src/services/portManager');
const apiRoutes = require('./src/routes/api');
const { analizeVideo } = require('./src/services/analizeVideo');
const { processVideo, generateRandomId } = require('./src/services/processVideo');
const { updateProgress } = require('./src/utils/progressTracker');


// ------------------------------------------------------------
// Express JS
const app = express();
app.use(express.json({ limit: '50mb' }));

/**
 * API response format for errors
 * @param {Object} res - Express response object
 * @param {number} statusCode - HTTP status code
 * @param {string} message - Error message
 * @param {string} [code] - Error code
 * @returns {Object} - Formatted error response
 */
const sendErrorResponse = (res, statusCode, message, code = 'GENERAL_ERROR') => {
  return res.status(statusCode).json({
    success: false,
    error: {
      code,
      message
    }
  });
};

/**
 * API response format for success
 * @param {Object} res - Express response object
 * @param {Object} data - Response data
 * @returns {Object} - Formatted success response
 */
const sendSuccessResponse = (res, data) => {
  return res.json({
    success: true,
    data
  });
};

// ------------------------------------------------------------

// Health check (if needed)
app.get('/health', (req, res) => {
  sendSuccessResponse(res, { status: 'ok' });
});

// Path Check
app.post('/process-reel', async (req, res) => {
  try {
    // Support both 'output' and 'outputs' for backward compatibility
    const { input, output, outputs } = req.body;
    const outputPath = output || (outputs && typeof outputs === 'string' ? outputs : null);
    
    // Generate a unique job ID
    const jobId = generateRandomId();
    
    // Validate inputs
    if (!input || !(await isUrlValid(input, true)))
      return sendErrorResponse(res, 400, 'Invalid input URL', 'INVALID_INPUT_URL');
    
    if (!outputPath || !(await isUrlValid(outputPath, false))) {
      return sendErrorResponse(res, 400, 'Invalid output path', 'INVALID_OUTPUT_PATH');
    }
    
    // Initialize progress tracking with all fields
    updateProgress(jobId, {
      analysis: { progress: 0, status: "Starting video analysis..." },
      processing: { progress: 0, status: "Waiting for analysis to complete" },
      encoding: { progress: 0, status: "Waiting for processing" },
      audio: { progress: 0, status: "Waiting for encoding" },
      videoGenerated: false
    });
    
    // Start processing in the background
    (async () => {
      try {
        // Run analysis (progress will be updated by the function)
        const analysis = await analizeVideo(input, jobId);
        
        // Mark analysis as complete
        updateProgress(jobId, {
          analysis: { progress: 100, status: "Analysis complete" }
        });
        
        const result = await processVideo(input, outputPath, analysis, jobId);
      } catch (error) {
        console.error(`Job ID [${jobId}]: Error processing: ${error.message}`);
        
        // Determine which stage had the error
        if (error.message.includes('analiz') || error.message.includes('analys')) {
          updateProgress(jobId, {
            analysis: { progress: -1, status: `Error: ${error.message}` }
          });
        } else {
          updateProgress(jobId, {
            processing: { progress: -1, status: `Error: ${error.message}` }
          });
        }
      }
    })();
    
    // Return immediately with the job ID
    return sendSuccessResponse(res, { 
      jobId,
      message: 'Video processing started',
      input,
      output: outputPath 
    });
  } catch (error) {
    console.error(`Error initializing video processing: ${error.message}`);
    return sendErrorResponse(res, 500, `Error initializing video processing: ${error.message}`, 'PROCESSING_ERROR');
  }
});

// Add endpoint to check processing status
app.get('/status/:jobId', (req, res) => {
  try {
    const { jobId } = req.params;
    
    if (!jobId) {
      return sendErrorResponse(res, 400, 'Job ID is required', 'MISSING_JOB_ID');
    }
    
    // Read progress.json to get the status
    if (!fs.existsSync('progress.json')) {
      return sendErrorResponse(res, 404, 'No active jobs found', 'NO_JOBS_FOUND');
    }
    
    const progressData = JSON.parse(fs.readFileSync('progress.json', 'utf-8'));
    
    // Check if this job exists in the progress data
    if (!progressData[jobId]) {
      return sendErrorResponse(res, 404, `Job ${jobId} not found or completed`, 'JOB_NOT_FOUND');
    }
    
    const jobStatus = progressData[jobId];
    
    // Calculate overall progress (analysis: 30%, processing: 50%, encoding: 15%, audio: 5%)
    const analysisProgress = jobStatus.analysis?.progress || 0;
    const processingProgress = jobStatus.processing?.progress || 0;
    const encodingProgress = jobStatus.encoding?.progress || 0;
    const audioProgress = jobStatus.audio?.progress || 0;
    const overallProgress = Math.round(
      (analysisProgress * 0.3) + 
      (processingProgress * 0.5) + 
      (encodingProgress * 0.15) + 
      (audioProgress * 0.05)
    );
    
    // Determine current phase
    let currentPhase = "initializing";
    let currentStatus = "Starting...";
    
    if (jobStatus.videoGenerated) {
      currentPhase = "complete";
      currentStatus = "Video generation complete";
    } else if (audioProgress > 0 && audioProgress < 100) {
      currentPhase = "audio";
      currentStatus = jobStatus.audio?.status || "Adding audio";
    } else if (audioProgress === 100) {
      currentPhase = "complete";
      currentStatus = "Audio merging complete";
    } else if (encodingProgress > 0 && encodingProgress < 100) {
      currentPhase = "encoding";
      currentStatus = jobStatus.encoding?.status || "Encoding video";
    } else if (encodingProgress === 100) {
      currentPhase = "audio";
      currentStatus = "Starting audio merging";
    } else if (analysisProgress === 100) {
      if (processingProgress === -1) {
        currentPhase = "error";
        currentStatus = jobStatus.processing?.status || "Error in processing";
      } else if (processingProgress === 100) {
        currentPhase = "encoding";
        currentStatus = "Starting encoding";
      } else {
        currentPhase = "processing";
        currentStatus = jobStatus.processing?.status || "Processing video";
      }
    } else if (analysisProgress > 0 && analysisProgress < 100) {
      currentPhase = "analyzing";
      currentStatus = jobStatus.analysis?.status || "Analyzing video";
    } else if (analysisProgress === -1) {
      currentPhase = "error";
      currentStatus = jobStatus.analysis?.status || "Error in analysis";
    }
    
    // Return the job status with additional info
    // Include all progress fields in the response
    return sendSuccessResponse(res, {
      jobId,
      currentPhase,
      currentStatus,
      overallProgress,
      status: jobStatus.status,
      analysis: jobStatus.analysis,
      processing: jobStatus.processing,
      encoding: jobStatus.encoding,
      audio: jobStatus.audio,
      videoGenerated: jobStatus.videoGenerated
    });
  } catch (error) {
    console.error(`Error checking job status: ${error.message}`);
    return sendErrorResponse(res, 500, `Error checking job status: ${error.message}`, 'STATUS_CHECK_ERROR');
  }
});

// Mount API routes
app.use('/', apiRoutes);

// Server Initialization
/**
 * Starts server
 * @returns {Promise<http.Server>} HTTP instance
 */
const startServer = async () => {
  console.log('Initializing QuickReels server...');
  
  try {
    // Find an available port using the port manager
    const port = process.env.PORT || await portManager.findAvailablePort(3000);
    
    const server = app.listen(port, () => {
      console.log(`Server running on port ${port}`);
      // Register this port as active
      portManager.activeQuickReelsPorts.add(port);
    });
    
    // Handle graceful shutdown
    process.on('SIGINT', () => {
      console.log('Closing server...');
      server.close(() => {
        // Remove this port from active ports
        portManager.activeQuickReelsPorts.delete(port);
        console.log('Server closed');
        process.exit(0);
      });
    });
    
    return server;
  } catch (err) {
    console.error(`Failed to start server: ${err.message}`);
    throw err;
  }
};

// URL or Paths Validation
/**
 * Validates if a string is a valid URL (http, https, or file)
 * @param {string} urlString - The URL to validate
 * @param {boolean} isInput - True if this is an input path, false for output path
 * @returns {Promise<boolean>} - True if URL is valid, false otherwise
 */
async function isUrlValid(urlString, isInput = true) {
  return new Promise((resolve) => {
    try {
      if (!urlString || typeof urlString !== 'string') {
        resolve(false);
        return;
      }
      
      // For file paths (non-URLs)
      if ((urlString.startsWith('/') || urlString.includes(':\\')) && !urlString.startsWith('file://')) {
        if (isInput) {
          // For input paths, the file must exist
          resolve(fs.existsSync(urlString));
          return;
        } else {
          // For output paths, we just need write permission to the directory
          try {
            const dirPath = path.dirname(urlString);
            fs.mkdirSync(dirPath, { recursive: true });
            
            // Check if we can write to the directory
            const testFile = path.join(dirPath, `.quickreels_write_test_${Date.now()}`);
            fs.writeFileSync(testFile, 'test');
            fs.unlinkSync(testFile);
            
            resolve(true);
            return;
          } catch (err) {
            console.warn(`Cannot write to directory for output: ${err.message}`);
            resolve(false);
            return;
          }
        }
      }
      
      // Validate URL format
      try {
        const parsedUrl = new URL(urlString);
        
        // For http/https URLs, validate they're reachable
        if (parsedUrl.protocol === 'http:' || parsedUrl.protocol === 'https:') {
          const protocol = parsedUrl.protocol === 'https:' ? https : http;
          
          const req = protocol.request({
            method: 'HEAD',
            host: parsedUrl.hostname,
            path: parsedUrl.pathname + parsedUrl.search
          }, (res) => {
            // Accept 2xx and 3xx status codes for inputs
            // For outputs, we don't need to check the status
            if (isInput) {
              resolve(res.statusCode >= 200 && res.statusCode < 400);
            } else {
              // For outputs, we assume we can write to any URL
              resolve(true);
            }
          });
          
          req.on('error', () => {
            if (isInput) {
              resolve(false);
            } else {
              // For outputs, still assume we can write to any URL even if HEAD fails
              resolve(true);
            }
          });
          
          req.setTimeout(5000, () => {
            req.destroy();
            if (isInput) {
              resolve(false);
            } else {
              // For outputs, still assume we can write to any URL even if timeout
              resolve(true);
            }
          });
          
          req.end();
        } else if (parsedUrl.protocol === 'file:') {
          // For file URLs with file:// protocol
          const filePath = parsedUrl.pathname;
          const dirPath = path.dirname(filePath);
          
          if (isInput) {
            try {
              // Check if file exists and is readable
              fs.accessSync(filePath, fs.constants.R_OK);
              resolve(true);
            } catch {
              resolve(false);
            }
          } else {
            try {
              // For output, ensure directory exists and is writable
              fs.mkdirSync(dirPath, { recursive: true });
              resolve(true);
            } catch {
              resolve(false);
            }
          }
        } else {
          resolve(false);
        }
      } catch (error) {
        // URL constructor will throw if the URL is invalid
        resolve(false);
      }
    } catch (error) {
      resolve(false);
    }
  });
}

// Node Exports
module.exports = {
  app,
  startServer,
  isUrlValid,
  checkPortRange: portManager.checkPortRange,
  activeQuickReelsPorts: portManager.activeQuickReelsPorts,
  analizeVideo,
  processVideo
};

// Start the server if this is the main module
if (require.main === module) {
  startServer().catch(err => {
    console.error(`Failed to start server: ${err.message}`);
  });
}

/*
Example JSON Payload:
{
  "input": "/path/address/origin/video.mp4",
  "output": "/path/to/send/output.mp4"
}

Both "output" and "outputs" fields are supported for backward compatibility.

curl -X POST http://localhost:3000/process-reel \
  -H "Content-Type: application/json" \
  -d '{
        "input": "/Users/luis/Desktop/Demo QuickReels/office.mp4",
        "output": "/Users/luis/Downloads/outputs/output2.mp4"
      }'

or

curl -X POST http://localhost:3000/process-reel \
  -H "Content-Type: application/json" \
  -d '{
        "input": "/Users/luis/Desktop/Demo QuickReels/office.mp4",
        "outputs": "/Users/luis/Downloads/outputs/output2.mp4"
      }'
      
*/