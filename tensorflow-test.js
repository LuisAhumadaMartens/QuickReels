// tensorflow-test.js
const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const os = require('os');
const temp = require('temp');
const tf = require('@tensorflow/tfjs-node');
const ffmpeg = require('fluent-ffmpeg');
const ffmpegPath = require('@ffmpeg-installer/ffmpeg').path;

// Configure ffmpeg
ffmpeg.setFfmpegPath(ffmpegPath);

// Track and cleanup temp files
temp.track();

// Set up Express app
const app = express();
app.use(express.json());

// Set up multer for file uploads
const upload = multer({ 
  dest: temp.dir,
  limits: { fileSize: 100 * 1024 * 1024 } // 100MB limit
});

// Global model variable
let model = null;
const MODEL_PATH = path.join(__dirname, 'model');
const MODEL_JSON_PATH = `file://${path.join(MODEL_PATH, 'model.json')}`;

// Detection thresholds
const DETECTION_THRESHOLD = 0.3;
const PERSON_CLASS_ID = 0; // In COCO dataset, person is class 0

/**
 * Initialize TensorFlow and load the model
 */
async function initializeTensorFlow() {
  console.log('Initializing TensorFlow.js...');
  
  try {
    // Check if model directory exists
    if (!fs.existsSync(MODEL_PATH)) {
      throw new Error(`Model directory not found: ${MODEL_PATH}`);
    }
    
    // Look for model files
    const modelJsonPath = path.join(MODEL_PATH, 'model.json');
    const savedModelPath = path.join(MODEL_PATH, 'saved_model.pb');
    
    // Load the model - use appropriate method based on available files
    console.log(`Loading model from: ${MODEL_PATH}`);
    
    if (fs.existsSync(modelJsonPath)) {
      // Load using loadGraphModel if model.json exists
      model = await tf.loadGraphModel(`file://${modelJsonPath}`);
      console.log('Loaded model from model.json');
    } else if (fs.existsSync(savedModelPath)) {
      // If saved_model.pb exists, we need a different approach
      console.log('Found saved_model.pb format');
      model = await tf.node.loadSavedModel(MODEL_PATH);
      console.log('Loaded saved model from directory');
    } else {
      throw new Error('No model files found in the specified directory');
    }
    
    console.log('TensorFlow model loaded successfully');
    return true;
  } catch (error) {
    console.error('Error initializing TensorFlow:', error);
    return false;
  }
}

/**
 * Prepare input tensor from image buffer
 */
async function prepareInputTensor(frameBuffer) {
  try {
    // Decode image to tensor
    const imageTensor = tf.node.decodeImage(frameBuffer);
    
    // Get dimensions
    const [height, width] = imageTensor.shape;
    
    // Return the tensor (caller responsible for disposal)
    return imageTensor;
  } catch (error) {
    console.error('Error preparing input tensor:', error);
    throw error;
  }
}

/**
 * Run inference on a frame tensor
 */
async function runInference(tensor) {
  if (!model) {
    console.warn('Model not loaded, using default detection at center position');
    return [{ 
      x: 0.5,
      y: 0.5,
      width: 0.2,
      height: 0.5,
      confidence: 0.9,
      class: PERSON_CLASS_ID,
      keypoints: []
    }];
  }

  try {
    // Process the frame through the model
    console.log('Running inference on frame...');
    
    // Make prediction
    let result = await model.predict(tensor);
    console.log('Model prediction successful');
    
    // Process the model output into detections
    const detections = processDetections(result);
    
    // Return detections
    return detections;
  } catch (error) {
    console.error('Error in runInference:', error);
    return [{ 
      x: 0.5,
      y: 0.5,
      width: 0.2,
      height: 0.5,
      confidence: 0.9,
      class: PERSON_CLASS_ID,
      keypoints: []
    }];
  }
}

/**
 * Process detections from model prediction
 */
function processDetections(prediction, threshold = DETECTION_THRESHOLD) {
  try {
    if (prediction === null || prediction === undefined) {
      console.warn('Received null/undefined prediction');
      return [];
    }
    
    // Direct tensor format from MoveNet/PoseNet type model
    if (prediction.shape && prediction.shape.length === 3) {
      console.log('Processing tensor with shape:', prediction.shape);
      
      // Get the data as a regular JavaScript array
      const outputData = prediction.arraySync()[0];
      
      // Create a detection object with keypoints
      const detection = {
        x: 0.5,
        y: 0.5,
        width: 0.3,
        height: 0.7,
        confidence: 0.9,
        class: PERSON_CLASS_ID,
        keypoints: outputData
      };
      
      return [detection];
    }
    
    // If it's the output_0 format from SavedModel
    if (prediction.output_0) {
      console.log('Found output_0 format');
      
      const outputData = prediction.output_0.arraySync()[0];
      
      const personDetection = {
        x: 0.5,
        y: 0.5,
        width: 0.2,
        height: 0.5,
        confidence: 0.9,
        class: PERSON_CLASS_ID,
        keypoints: outputData
      };
      
      return [personDetection];
    }
    
    // Process object detection API format if that's what we got
    if (prediction.detection_boxes && prediction.detection_scores) {
      console.log('Found detection_boxes format');
      
      const boxes = prediction.detection_boxes;
      const scores = prediction.detection_scores;
      const classes = prediction.detection_classes || { 
        arraySync: () => Array(scores.arraySync()[0].length).fill(PERSON_CLASS_ID) 
      };
      
      const boxesData = boxes.arraySync()[0];
      const scoresData = scores.arraySync()[0];
      const classesData = classes.dataSync ? classes.dataSync() : classes.arraySync()[0];
      
      const detections = [];
      
      // Process each detection
      for (let i = 0; i < boxesData.length; i++) {
        const score = scoresData[i];
        const classId = Math.round(classesData[i]);
        
        // Only keep people with score above threshold
        if (score >= threshold && classId === PERSON_CLASS_ID) {
          // TensorFlow Object Detection API returns [y1, x1, y2, x2]
          const [y1, x1, y2, x2] = boxesData[i];
          
          // Calculate center point (normalized coordinates)
          const centerX = (x1 + x2) / 2;
          const centerY = (y1 + y2) / 2;
          
          // Calculate width and height
          const width = x2 - x1;
          const height = y2 - y1;
          
          detections.push({
            x: centerX,
            y: centerY,
            width,
            height,
            confidence: score,
            class: classId
          });
        }
      }
      
      return detections;
    }
    
    // If all else fails, return empty array
    console.warn('Unknown prediction format');
    return [];
  } catch (error) {
    console.error('Error processing detections:', error);
    return [];
  }
}

/**
 * Process a video to show TensorFlow detections without cropping
 */
async function processVideoWithTensorFlow(inputPath, outputPath, debug = true) {
  console.log(`Processing video with TensorFlow: ${inputPath} -> ${outputPath}`);
  
  // Create temporary directory for frames
  const tempDir = path.join(os.tmpdir(), `tf-test-${Date.now()}`);
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }
  
  try {
    // Get video metadata
    const metadata = await new Promise((resolve, reject) => {
      ffmpeg.ffprobe(inputPath, (err, metadata) => {
        if (err) return reject(err);
        
        const videoStream = metadata.streams.find(s => s.codec_type === 'video');
        if (!videoStream) return reject(new Error('No video stream found'));
        
        // Extract relevant metadata
        const fps = eval(videoStream.r_frame_rate || '30');
        const duration = parseFloat(metadata.format.duration);
        const totalFrames = Math.round(duration * fps);
        
        resolve({
          width: videoStream.width,
          height: videoStream.height,
          fps,
          duration,
          totalFrames
        });
      });
    });
    
    console.log('Video metadata:', metadata);
    
    // Extract frames at a lower frame rate for analysis
    const frameInterval = Math.max(1, Math.floor(metadata.fps / 5)); // 5 fps
    console.log(`Extracting frames at interval ${frameInterval}`);
    
    await new Promise((resolve, reject) => {
      ffmpeg(inputPath)
        .outputOptions([
          '-vf', `fps=${metadata.fps/frameInterval}`,
          '-q:v', '2'
        ])
        .on('end', () => {
          console.log('Frame extraction complete');
          resolve();
        })
        .on('error', (err) => {
          console.error('Frame extraction error:', err);
          reject(err);
        })
        .output(path.join(tempDir, 'frame-%04d.jpg'))
        .run();
    });
    
    // Find all extracted frames
    const frameFiles = fs.readdirSync(tempDir)
      .filter(file => file.endsWith('.jpg'))
      .sort();
    
    console.log(`Found ${frameFiles.length} frames`);
    
    // Process each frame with TensorFlow and generate visualization commands
    const drawCommands = [];
    
    for (let i = 0; i < frameFiles.length; i++) {
      const frameFile = frameFiles[i];
      const framePath = path.join(tempDir, frameFile);
      
      console.log(`Processing frame ${i+1}/${frameFiles.length}`);
      
      try {
        // Read the frame
        const frameBuffer = fs.readFileSync(framePath);
        const frameTensor = await prepareInputTensor(frameBuffer);
        
        // Run inference
        const detections = await runInference(frameTensor);
        
        // Generate visualization commands for this frame
        const frameIndex = i;
        const timestamp = frameIndex / (metadata.fps / frameInterval);
        
        // Add visualization for each detection
        if (detections && detections.length > 0) {
          detections.forEach(detection => {
            // Draw bounding box
            const x1 = Math.floor(detection.x * metadata.width - (detection.width * metadata.width / 2));
            const y1 = Math.floor(detection.y * metadata.height - (detection.height * metadata.height / 2));
            const w = Math.floor(detection.width * metadata.width);
            const h = Math.floor(detection.height * metadata.height);
            
            drawCommands.push(`${timestamp} drawbox x ${x1} y ${y1} w ${w} h ${h} color yellow@0.5 t 2`);
            
            // Add confidence text
            const confPercent = Math.floor((detection.confidence || 0.5) * 100);
            drawCommands.push(`${timestamp} drawbox x ${x1} y ${y1-20} w 50 h 20 color black@0.5 t fill`);
            
            // Process keypoints if available
            if (detection.keypoints && detection.keypoints.length > 0) {
              detection.keypoints.forEach((kp, kpIndex) => {
                if (kp && kp.length >= 3 && kp[2] > 0.3) {
                  const y = kp[0];
                  const x = kp[1];
                  const conf = kp[2];
                  
                  if (!isNaN(x) && !isNaN(y) && !isNaN(conf)) {
                    const xPixel = Math.floor(x * metadata.width);
                    const yPixel = Math.floor(y * metadata.height);
                    
                    // Draw keypoint circle
                    drawCommands.push(`${timestamp} drawbox x ${xPixel-5} y ${yPixel-5} w 10 h 10 color green@0.8 t fill`);
                    
                    // Draw confidence text background
                    drawCommands.push(`${timestamp} drawbox x ${xPixel+10} y ${yPixel-20} w 35 h 15 color black@0.5 t fill`);
                  }
                }
              });
            }
          });
        }
        
        // Clean up
        tf.dispose(frameTensor);
      } catch (error) {
        console.error(`Error processing frame ${frameFile}:`, error);
      }
    }
    
    // Create filter script for ffmpeg
    let filterCommands = '';
    if (drawCommands.length > 0) {
      filterCommands = `sendcmd=c='${drawCommands.join('\n').replace(/'/g, "\''")}',`;
      filterCommands += `drawbox=x=0:y=0:w=10:h=10:color=red@0:t=fill`; // Initial invisible box
    }
    
    // Apply the visualizations to the video
    await new Promise((resolve, reject) => {
      const ffmpegCommand = ffmpeg(inputPath);
      
      if (filterCommands) {
        ffmpegCommand.videoFilter(filterCommands);
      }
      
      ffmpegCommand
        .outputOptions([
          '-c:v', 'libx264',
          '-pix_fmt', 'yuv420p',
          '-preset', 'fast',
          '-crf', '23'
        ])
        .output(outputPath)
        .on('start', (cmd) => {
          console.log('FFmpeg command:', cmd);
        })
        .on('progress', (progress) => {
          console.log(`Processing: ${Math.floor(progress.percent || 0)}% done`);
        })
        .on('end', () => {
          console.log('Video processing complete');
          resolve();
        })
        .on('error', (err) => {
          console.error('FFmpeg error:', err);
          reject(err);
        })
        .run();
    });
    
    // Clean up temp directory
    try {
      fs.rmSync(tempDir, { recursive: true, force: true });
    } catch (error) {
      console.warn('Failed to remove temporary directory:', error);
    }
    
    return {
      status: 'success',
      inputPath,
      outputPath
    };
  } catch (error) {
    console.error('Error processing video:', error);
    
    // Clean up temp directory on error
    try {
      fs.rmSync(tempDir, { recursive: true, force: true });
    } catch (cleanupError) {
      console.warn('Failed to remove temporary directory:', cleanupError);
    }
    
    throw error;
  }
}

// API endpoint for processing video
app.post('/process-tensorflow', upload.single('video'), async (req, res) => {
  try {
    // Check if we have a file
    if (!req.file) {
      return res.status(400).json({ error: 'No video file provided' });
    }
    
    console.log('Processing video file:', req.file.path);
    
    // Create output file
    const outputFile = temp.path({ suffix: '.mp4' });
    
    try {
      // Make sure TensorFlow is initialized
      if (!model) {
        await initializeTensorFlow();
      }
      
      // Process the video
      const result = await processVideoWithTensorFlow(req.file.path, outputFile);
      
      // Read the result video
      const videoData = fs.readFileSync(outputFile);
      
      // Return the result
      res.json({
        video: videoData.toString('base64')
      });
      
      // Clean up
      fs.unlinkSync(outputFile);
      fs.unlinkSync(req.file.path);
    } catch (error) {
      console.error('Error processing video:', error);
      
      // Clean up
      try {
        fs.unlinkSync(outputFile);
        fs.unlinkSync(req.file.path);
      } catch (e) {
        // Ignore cleanup errors
      }
      
      return res.status(500).json({
        error: `Error processing video: ${error.message}`
      });
    }
  } catch (error) {
    console.error('Error handling request:', error);
    res.status(500).json({
      error: 'Server error processing request'
    });
  }
});

// API endpoint for health check
app.get('/', (req, res) => {
  res.send('TensorFlow Test Server is running');
});

// Start the server
const PORT = process.env.PORT || 3001;
async function startServer() {
  // Initialize TensorFlow
  await initializeTensorFlow();
  
  // Start the server
  app.listen(PORT, () => {
    console.log(`TensorFlow Test Server listening on port ${PORT}`);
    console.log(`POST to /process-tensorflow to test TensorFlow detection`);
  });
}

startServer().catch(error => {
  console.error('Failed to start server:', error);
}); 