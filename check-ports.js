const { checkPortRange, activeQuickReelsPorts } = require('./quickreels');

// Check ports from 3000 to 3010
const startPort = 3000;
const endPort = 3010;

console.log(`Checking ports from ${startPort} to ${endPort}...`);
console.log('Legend:');
console.log('✅ Available');
console.log('🎬 In use by QuickReels');
console.log('❌ In use by another application\n');

checkPortRange(startPort, endPort)
  .then(() => {
    if (activeQuickReelsPorts.size > 0) {
      console.log('\nCurrent QuickReels instances:');
      activeQuickReelsPorts.forEach(port => {
        console.log(`- Running on port ${port}`);
      });
    }
    console.log('\nPort check complete!');
  })
  .catch(console.error);