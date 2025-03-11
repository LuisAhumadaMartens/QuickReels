const net = require('net');

const activeQuickReelsPorts = new Set();

function isPortAvailable(port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    
    server.once('error', () => {
      resolve({
        available: false,
        usedByQuickReels: activeQuickReelsPorts.has(port)
      });
    });
    
    server.once('listening', () => {
      server.close();
      resolve({
        available: true,
        usedByQuickReels: false
      });
    });
    
    server.listen(port);
  });
}

async function findAvailablePort(startPort) {
  let port = startPort;
  let portStatus;
  do {
    portStatus = await isPortAvailable(port);
    if (!portStatus.available) port++;
  } while (!portStatus.available);
  return port;
}

async function checkPortRange(startPort, endPort) {
  console.log('\nChecking port availability:');
  for (let port = startPort; port <= endPort; port++) {
    const status = await isPortAvailable(port);
    if (status.available) {
      console.log(`Port ${port}: ✅ Available`);
    } else if (status.usedByQuickReels) {
      console.log(`Port ${port}: 🎬 In use by QuickReels`);
    } else {
      console.log(`Port ${port}: ❌ In use by another application`);
    }
  }
}

function registerPort(port) {
  activeQuickReelsPorts.add(port);
}

function unregisterPort(port) {
  activeQuickReelsPorts.delete(port);
}

module.exports = {
  isPortAvailable,
  findAvailablePort,
  checkPortRange,
  registerPort,
  unregisterPort,
  activeQuickReelsPorts
}; 