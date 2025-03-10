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
const { processVideo } = require('./src/services/processVideo');


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
    const { input, outputs } = req.body;
    
    // Validate inputs
    if (!input || !(await isUrlValid(input)))
      return sendErrorResponse(res, 400, 'Invalid input URL', 'INVALID_INPUT_URL');
    if (!outputs || !Array.isArray(outputs) || outputs.length === 0)
      return sendErrorResponse(res, 400, 'Invalid outputs configuration', 'INVALID_OUTPUTS');

    
    // Validate each output URL
    for (const output of outputs) {
      if (!output.url || !(await isUrlValid(output.url))) { // Ahora se usa nomas el path para crear el archivo, idealmente usar el URL para saber donde mandar el archivo.
        return sendErrorResponse(res, 400, 'Invalid output URL', 'INVALID_OUTPUT_URL');
      }
    }
    
    // Process the video
    console.log(`Processing reel from: ${input}`);
    const analysis = await analizeVideo(input);
    const results = await processVideo(input, outputs, analysis);
    
    return sendSuccessResponse(res, { results });
  } catch (error) {
    console.error('Error processing video:', error);
    return sendErrorResponse(res, 500, `Error processing video: ${error.message}`, 'PROCESSING_ERROR');
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
    console.error('Failed to start server:', err);
    throw err;
  }
};

// URL or Paths Validation
/**
 * Validates if a string is a valid URL (http, https, or file)
 * @param {string} urlString - The URL to validate
 * @returns {Promise<boolean>} - True if URL is valid, false otherwise
 */
async function isUrlValid(urlString) {
  return new Promise((resolve) => {
    try {
      if (!urlString || typeof urlString !== 'string') {
        resolve(false);
        return;
      }
      
      // For file paths (non-URLs)
      if ((urlString.startsWith('/') || urlString.includes(':\\')) && !urlString.startsWith('file://')) {
        resolve(fs.existsSync(urlString));
        return;
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
            // Accept 2xx and 3xx status codes
            resolve(res.statusCode >= 200 && res.statusCode < 400); // Asumiendo que eso queremos. Revisar si es necesario.
          });
          
          req.on('error', () => {
            resolve(false);
          });
          
          req.setTimeout(5000, () => {
            req.destroy();
            resolve(false);
          });
          
          req.end();
        } else if (parsedUrl.protocol === 'file:') {
          // For file URLs, validate we have write permissions to the directory
          const filePath = parsedUrl.pathname;
          const dirPath = path.dirname(filePath);
          try {
            fs.accessSync(dirPath, fs.constants.W_OK);
            resolve(true);
          } catch {
            resolve(false);
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
    console.error('Failed to start server:', err);
  });
}

/*
Example JSON Payload:
{
  "input": "http://example.com/input.mp4",
  "outputs": [
    {
      "url": "http://example.com/output1.mp4",
      "range": "[123-834]"
    },
    {
      "url": "http://example.com/output2.mp4"
    }
  ]
}

curl -X POST http://localhost:3000/process-reel \
  -H "Content-Type: application/json" \
  -d '{
        "input": "/Users/luis/Desktop/Demo QuickReels/office.mp4",
        "outputs": [
          {
            "url": "/Users/luis/Downloads/outputs/output2.mp4"
          }
        ]
      }'
      
*/

/*

Hay varios cambios que es que hacer dependiendo como funcionara.

Validamos URL porque debemos descargar el video en el directorio, de ahi procesarlo.
El URL de output podria ser donde lo guardaremos. Asi que seria que aisgnar un espacio temporal en el directorio tambien, y de ahi entregarlo a esa URL.
Pero mientras el script usa ambas para asi probar en local.

*/