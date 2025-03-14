const quickReels = require('../index.js');
const path = require('path');

const args = process.argv.slice(2);

if (args.includes('--help') || args.includes('-h') || args.length < 2) {
  console.log('\nUsage:');
  console.log('  quickreels <input_video> <output_video>');
  console.log('\nOptions:');
  console.log('  --help, -h             Show this help message');
  process.exit(args.includes('--help') || args.includes('-h') ? 0 : 1);
}

const inputPath = args[0];
const outputPath = args[1];

quickReels(inputPath, outputPath)
  .then(result => {
    console.log(result);
    process.exit(0);
  })
  .catch(error => {
    console.error('Error:', error.message);
    process.exit(1);
  }); 