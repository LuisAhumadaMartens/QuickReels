// quickreels.js

// ===============================
// Node.js Core Modules
// ===============================
const express = require('express');

// ===============================
// Import Custom Modules
// ===============================
const config = require('./src/config/config');
const portManager = require('./src/services/portManager');
const apiRoutes = require('./src/routes/api');
const videoAnalyzer = require('./src/services/analizeVideo');
const processVideo = require('./src/services/processVideo');

// ===============================
// Express App Setup
// ===============================
const app = express();
app.use(express.json());

// Mount API routes
app.use('/', apiRoutes);

// ===============================
// Server Initialization
// ===============================
async function startServer(port = config.PORT) {
  try {
    console.log('Initializing QuickReels server...');
    
    // Initialize TensorFlow.js and wait for it to complete
    console.log('Loading TensorFlow model...');
    const modelLoaded = await videoAnalyzer.initializeTensorFlow();
    
    if (modelLoaded) {
      console.log('TensorFlow model loaded successfully.');
    } else {
      console.warn('WARNING: TensorFlow model could not be loaded. Using mock data for video analysis.');
      console.warn('Video analysis results will not be accurate.');
    }
    
    // Find available port and start the server
    const availablePort = await portManager.findAvailablePort(port);
    const server = app.listen(availablePort, () => {
      portManager.registerPort(availablePort);
      console.log(`QuickReels API listening at http://localhost:${availablePort}`);
      console.log('Server is ready to process video requests.');
    });

    // Remove port from tracking when server closes
    server.on('close', () => {
      portManager.unregisterPort(availablePort);
      console.log(`QuickReels instance on port ${availablePort} closed`);
    });

    return server;
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

// ===============================
// Exports
// ===============================
module.exports = {
  app,
  startServer,
  checkPortRange: portManager.checkPortRange,
  activeQuickReelsPorts: portManager.activeQuickReelsPorts,
  videoAnalyzer,
  processVideo
};

// Start the server if this file is run directly
if (require.main === module) {
  startServer();
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
*/

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
        // Accept 2xx and 3xx status codes
        resolve(res.statusCode >= 200 && res.statusCode < 400);
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