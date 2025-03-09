// quickreels.js

// ===============================
// Node.js Core Modules
// ===============================
const express = require('express');
const fs = require('fs');
const path = require('path');
const http = require('http');
const https = require('https');
const url = require('url');

// ===============================
// Import Custom Modules
// ===============================
const config = require('./src/config/config');
const portManager = require('./src/services/portManager');
const apiRoutes = require('./src/routes/api');
const { analizeVideo } = require('./src/services/analizeVideo');
const { processVideo } = require('./src/services/processVideo');

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
const startServer = async () => {
  console.log('Initializing QuickReels server...');
  
  try {
    // No longer need to set debug mode
    // Remove: analizeVideo.setDebugMode(true);
    
    // Initialize Express app
    const app = express();
    app.use(express.json({ limit: '50mb' }));
    
    // API Routes
    app.post('/process-reel', async (req, res) => {
      try {
        const { input, outputs } = req.body;
        
        // Validate inputs
        if (!input || !isValidUrl(input)) {
          return res.status(400).json({ error: 'Invalid input URL' });
        }
        
        if (!outputs || !Array.isArray(outputs) || outputs.length === 0) {
          return res.status(400).json({ error: 'Invalid outputs configuration' });
        }
        
        for (const output of outputs) {
          if (!output.url || !isValidUrl(output.url)) {
            return res.status(400).json({ error: 'Invalid output URL' });
          }
        }
        
        // Process the video
        console.log(`Processing reel from: ${input}`);
        const analysis = await analizeVideo(input);
        const results = await processVideo(input, outputs, analysis);
        
        return res.json({ results });
      } catch (error) {
        console.error('Error processing video:', error);
        return res.status(500).json({ error: `Error processing video: ${error.message}` });
      }
    });
    
    // Health check endpoint
    app.get('/health', (req, res) => {
      res.json({ status: 'ok' });
    });
    
    // Find an available port
    const port = process.env.PORT || 3000;
    const server = app.listen(port, () => {
      console.log(`Server running on port ${port}`);
    });
    
    process.on('SIGINT', () => {
      console.log('Closing server...');
      server.close(() => {
        console.log('Server closed');
        process.exit(0);
      });
    });
    
  } catch (err) {
    throw err;
  }
};

// Helper function to validate URLs
const isValidUrl = (string) => {
  try {
    // Check if it's a file URL
    if (string.startsWith('file://')) {
      const filePath = new URL(string).pathname;
      return fs.existsSync(filePath);
    }
    
    // Check if it's a regular file path
    if (string.startsWith('/') || string.includes(':\\')) {
      return fs.existsSync(string);
    }
    
    // Check if it's an HTTP/HTTPS URL
    const url = new URL(string);
    return url.protocol === 'http:' || url.protocol === 'https:';
  } catch (err) {
    return false;
  }
};

// ===============================
// Exports
// ===============================
module.exports = {
  app,
  startServer,
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
*/

// ===============================
// URL Validation Helper
// ===============================
function isUrlValid(urlString) {
  return new Promise((resolve) => {
    try {
      if (!urlString || typeof urlString !== 'string') {
        resolve(false);
        return;
      }
      
      // Validate URL format
      const parsedUrl = url.parse(urlString);
      if (!parsedUrl.protocol || !parsedUrl.hostname) {
        resolve(false);
        return;
      }
      
      // For http/https URLs, validate they're reachable
      if (parsedUrl.protocol === 'http:' || parsedUrl.protocol === 'https:') {
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
      } else {
        // For file URLs, validate file exists
        if (parsedUrl.protocol === 'file:') {
          const filePath = parsedUrl.path;
          resolve(fs.existsSync(filePath));
        } else {
          resolve(false);
        }
      }
    } catch (error) {
      resolve(false);
    }
  });
}