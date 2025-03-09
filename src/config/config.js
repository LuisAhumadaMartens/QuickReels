const fs = require('fs');
const path = require('path');

// Read package.json to get the port number
const packageJson = JSON.parse(fs.readFileSync(path.join(__dirname, '../../package.json'), 'utf8'));
const PORT = packageJson.config.port;

module.exports = {
  PORT,
  // Add other configuration settings here as needed
}; 