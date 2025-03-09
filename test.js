const http = require('http');
const { startServer } = require('./quickreels');

// Define LeetCode-style test cases
const testCases = [
  {
    description: 'Valid input with two outputs, one with range',
    payload: {
      input: 'http://example.com/input.mp4',
      outputs: [
        { url: 'http://example.com/output1.mp4', range: '[123-834]' },
        { url: 'http://example.com/output2.mp4' }
      ]
    },
    expected: 'Video processing initiated'
  },
  {
    description: 'Invalid input: missing outputs',
    payload: {
      input: 'http://example.com/input.mp4'
    },
    expected: 'Invalid request payload'
  },
  {
    description: 'Invalid input: outputs not an array',
    payload: {
      input: 'http://example.com/input.mp4',
      outputs: 'not-an-array'
    },
    expected: 'Invalid request payload'
  }
];

// Function to run a single test case
async function runTestCase(testCase, port) {
  const server = await startServer(port);
  
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(testCase.payload);

    const options = {
      hostname: 'localhost',
      port: server.address().port, // Use the actual port assigned
      path: '/process-reel',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(data)
      }
    };

    const req = http.request(options, (res) => {
      let responseBody = '';

      res.on('data', (chunk) => {
        responseBody += chunk;
      });

      res.on('end', () => {
        const responseJson = JSON.parse(responseBody);
        const result = responseJson.message === testCase.expected ? 'Valid ✅' : 'Not Valid ❌';

        console.log(`\nTest Case: ${testCase.description}`);
        console.log('Expected:', testCase.expected);
        console.log('Actual Response:', responseJson.message);
        console.log('Result:', result);

        server.close(() => resolve());
      });
    });

    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

// Run all test cases sequentially with dynamic ports
async function runTests() {
  for (let i = 0; i < testCases.length; i++) {
    await runTestCase(testCases[i], 3001 + i);
  }
}

// Start the tests
runTests().catch(console.error); 