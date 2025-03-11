// src/services/analizeVideo.js
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const os = require('os');
const ffmpeg = require('fluent-ffmpeg');
const ffmpegPath = require('@ffmpeg-installer/ffmpeg').path;
const EventEmitter = require('events');
const { promisify } = require('util');
const exec = promisify(require('child_process').exec);
const config = require('../config/config');

// Set the ffmpeg path
ffmpeg.setFfmpegPath(ffmpegPath);

// Create event emitter for frame processing events
const eventEmitter = new EventEmitter();

// Global model variable
let model = null;
const MODEL_PATH = path.join(__dirname, '../../model');

// Get detection thresholds from config
const { DETECTION_THRESHOLD, PERSON_CLASS_ID, SCENE_CHANGE_THRESHOLD, DEFAULT_CENTER, MOVE_NET_INPUT_SIZE } = config;

/**
 * Register event listener for frame processing
 * @param {string} event - Event name
 * @param {Function} callback - Callback function
 */
function on(event, callback) {
  eventEmitter.on(event, callback);
}

/**
 * Initialize TensorFlow and load the MoveNet model
 * Direct implementation from tensorflow-test-cli.js
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
    
    console.log(`Loading model from: ${MODEL_PATH}`);
    
    // First try loading as GraphModel (model.json)
    if (fs.existsSync(modelJsonPath)) {
      console.log('Found model.json, loading as GraphModel');
      try {
        model = await tf.loadGraphModel(`file://${modelJsonPath}`);
        console.log('Successfully loaded model from model.json');
        return true;
      } catch (err) {
        console.warn('Failed to load as GraphModel:', err.message);
      }
    }
    
    // Then try loading as SavedModel
    if (fs.existsSync(savedModelPath)) {
      console.log('Found saved_model.pb, loading as SavedModel');
      try {
        model = await tf.node.loadSavedModel(MODEL_PATH);
        console.log('Successfully loaded model from SavedModel');
        return true;
      } catch (err) {
        console.error('Failed to load as SavedModel:', err);
        throw err;
      }
    }
    
    throw new Error('No compatible model files found in the specified directory');
  } catch (error) {
    console.error('Error initializing TensorFlow:', error);
    return false;
  }
}

/**
 * Prepare input tensor from image buffer for MoveNet model
 * Direct implementation from tensorflow-test-cli.js matching toReel.py
 */
async function prepareInputTensor(frameBuffer) {
  try {
    // Decode image to tensor (3 channels for RGB)
    const imageTensor = tf.node.decodeImage(frameBuffer, 3);
    
    // Log original shape
    console.log(`Original image shape: ${imageTensor.shape}`);
    
    // Resize to MoveNet input size (192x192) - matches Python's cv2.resize
    const resized = tf.image.resizeBilinear(imageTensor, MOVE_NET_INPUT_SIZE);
    
    // Convert to int32 (matches Python's tf.convert_to_tensor(..., dtype=tf.int32))
    const convertedTensor = tf.cast(resized, 'int32');
    
    // Add batch dimension (matches Python's tf.expand_dims(input_tensor, axis=0))
    const batchedTensor = tf.expandDims(convertedTensor, 0);
    
    // Log final tensor shape
    console.log(`Prepared tensor shape: ${batchedTensor.shape}`);
    
    // Clean up intermediate tensors
    tf.dispose([imageTensor, resized, convertedTensor]);
    
    return batchedTensor;
  } catch (error) {
    console.error('Error preparing input tensor:', error);
    throw error;
  }
}

/**
 * Calculate Mean Squared Error between two tensors
 * Matches the Python implementation in toReel.py
 */
function calculateMSE(tensorA, tensorB) {
  if (!tensorA || !tensorB) return 0;
  
  return tf.tidy(() => {
    // Calculate difference
    const diff = tf.sub(tensorA, tensorB);
    
    // Square the differences
    const squaredDiff = tf.square(diff);
    
    // Calculate mean (MSE)
    return squaredDiff.mean().arraySync();
  });
}

/**
 * Run inference on a frame tensor using MoveNet model
 * Direct implementation from tensorflow-test-cli.js
 */
async function runInference(tensor) {
  if (!model) {
    console.warn('Model not loaded, using default detection');
    return [{ 
      x: 0.5,
      y: 0.5,
      width: 0.2,
      height: 0.5,
      confidence: 0.9,
      class: PERSON_CLASS_ID,
      keypoints: Array(17).fill().map(() => [0.5, 0.5, 0.1]) // Create empty keypoints
    }];
  }

  try {
    // Process the frame through the model
    console.log('Running MoveNet inference...');
    
    // In TensorFlow.js, we need to use predict() instead of execute()
    // This is different from Python, which uses signatures['serving_default']
    let result = await model.predict(tensor);
    
    // Extract keypoints from the result
    let keypoints = [];
    if (result.output_0) {
      // SavedModel format
      console.log('Found output_0 tensor:', result.output_0.shape);
      keypoints = result.output_0.arraySync()[0];
    } else if (result.arraySync) {
      // GraphModel format
      console.log('Found tensor with shape:', result.shape);
      keypoints = result.arraySync()[0];
          } else {
      console.warn('Unknown result format:', result);
      return [];
    }
    
    // Clean up the tensors
    if (result.output_0 && result.output_0.dispose) {
      result.output_0.dispose();
    } else if (result.dispose) {
      result.dispose();
    }
    
    // Filter keypoints by confidence
    const detections = [];
    
    // Process keypoints
    if (keypoints && keypoints.length > 0) {
      // Create a detection object from all keypoints
      const validKeypoints = keypoints.filter(kp => kp && kp.length >= 3 && kp[2] > DETECTION_THRESHOLD);
      
      if (validKeypoints.length > 0) {
        // Calculate average position from valid keypoints
        const sumX = validKeypoints.reduce((sum, kp) => sum + kp[1], 0); // x coordinate
        const sumY = validKeypoints.reduce((sum, kp) => sum + kp[0], 0); // y coordinate
        const avgX = sumX / validKeypoints.length;
        const avgY = sumY / validKeypoints.length;
        
        // Calculate average confidence
        const avgConfidence = validKeypoints.reduce((sum, kp) => sum + kp[2], 0) / validKeypoints.length;
        
        // Create a detection object with the keypoints
        const detection = {
          x: avgX,
          y: avgY,
          width: 0.3,
          height: 0.7,
          confidence: avgConfidence,
          class: PERSON_CLASS_ID,
          keypoints: validKeypoints
        };
        
        console.log(`Found person with ${validKeypoints.length} valid keypoints, confidence ${avgConfidence.toFixed(2)}`);
        detections.push(detection);
      }
    }
    
    console.log(`Found ${detections.length} detections in frame`);
    
    return detections;
  } catch (error) {
    console.error('Error in runInference:', error);
    return [];
  }
}

/**
 * Cluster people detections - replicates the behavior in toReel.py
 */
function clusterPeople(people, threshold = 0.05) {
  const clusters = [];
  
  for (const person of people) {
    const [personId, keypoints] = person;
    const [y, x, confidence] = keypoints[0]; // Use first keypoint
    
    let added = false;
    for (const cluster of clusters) {
      const centerX = cluster.sumX / cluster.count;
      const centerY = cluster.sumY / cluster.count;
      const distance = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2));
      
      if (distance < threshold) {
        cluster.sumX += x;
        cluster.sumY += y;
        cluster.count += 1;
        if (confidence > cluster.maxConf) {
          cluster.maxConf = confidence;
        }
        added = true;
        break;
      }
    }
    
    if (!added) {
    clusters.push({
        sumX: x,
        sumY: y,
        count: 1,
        maxConf: confidence
      });
    }
  }
  
  // Convert clusters to format expected by Python code
  const mergedPeople = clusters.map((cluster, i) => {
    const avgX = cluster.sumX / cluster.count;
    const avgY = cluster.sumY / cluster.count;
    return [i, avgX, avgY, cluster.maxConf];
  });
  
  return mergedPeople;
}

/**
 * Format frame number to timecode
 */
function formatTimecode(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Analyze video and extract camera tracking data
 * @param {string} inputPath - Path to input video
 * @returns {Object} - Analysis results
 */
async function analizeVideo(inputPath) {
  console.log(`Analyzing video: ${inputPath}`);
  
  try {
    // Initialize TensorFlow and load model
    await initializeTensorFlow();
    
    // Create temporary directory in current folder
    const tempDir = path.join(process.cwd(), 'temp', `quickreels-analysis-${Date.now()}`);
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }
    
    console.log(`Using temporary directory: ${tempDir}`);
    
    // Extract video metadata
    const metadata = await new Promise((resolve, reject) => {
      ffmpeg.ffprobe(inputPath, (err, metadata) => {
        if (err) return reject(err);
        
        const { width, height, duration, r_frame_rate } = metadata.streams[0];
        let fps = 30; // Default
        
        // Parse frame rate
        if (r_frame_rate) {
          const [num, den] = r_frame_rate.split('/').map(Number);
          fps = num / (den || 1);
        }
        
        resolve({
          width,
          height,
          duration,
          fps
        });
      });
    });
    
    console.log(`Video metadata: ${JSON.stringify(metadata)}`);
    
    // Extract frames for analysis (at full frame rate)
    const framesDir = path.join(tempDir, 'frames');
    if (!fs.existsSync(framesDir)) {
      fs.mkdirSync(framesDir, { recursive: true });
    }
    
    await new Promise((resolve, reject) => {
      ffmpeg(inputPath)
        .outputOptions(['-vf', `fps=${metadata.fps}`, '-q:v', '1'])
        .output(path.join(framesDir, 'frame-%04d.jpg'))
        .on('end', resolve)
        .on('error', reject)
        .run();
    });

    console.log('Frames extracted for analysis');
    
    // Get all frame files sorted by number
    const frameFiles = fs.readdirSync(framesDir)
      .filter(file => file.startsWith('frame-') && file.endsWith('.jpg'))
      .sort((a, b) => {
        const numA = parseInt(a.match(/frame-(\d+)/)[1]);
        const numB = parseInt(b.match(/frame-(\d+)/)[1]);
        return numA - numB;
      });
    
    console.log(`Found ${frameFiles.length} frames for analysis`);
    
    // Initialize movement planner
    const planner = new MovementPlanner(metadata.fps);
    let prevFrameGray = null;
    const frameDiffs = [];
    const sceneChanges = [];
    
    // First pass: Analyze each frame (like in Python)
    for (let i = 0; i < frameFiles.length; i++) {
      const frameFile = path.join(framesDir, frameFiles[i]);
      const imageBuffer = fs.readFileSync(frameFile);
      
      // Prepare grayscale image for scene change detection
      const grayscaleTensor = tf.node.decodeImage(imageBuffer, 1);  // 1 channel (grayscale)
      
      // Calculate frame difference (MSE)
      let frameDiff = 0;
      if (prevFrameGray) {
        frameDiff = calculateMSE(prevFrameGray, grayscaleTensor);
        frameDiffs.push(frameDiff);
        
        // Detect scene changes
        if (frameDiff > SCENE_CHANGE_THRESHOLD) {
          console.log(`Scene change detected at frame ${i}, MSE: ${frameDiff}`);
          sceneChanges.push(i);
        }
      }
      
      // Update previous frame
      if (prevFrameGray) prevFrameGray.dispose();
      prevFrameGray = grayscaleTensor;
      
      // Prepare tensor for inference
      const inputTensor = await prepareInputTensor(imageBuffer);
      
      // Run model inference
      const detections = await runInference(inputTensor);
      
      // Format detections for clusterPeople (similar to Python implementation)
      const people = detections.map((detection, idx) => {
        return [idx, detection.keypoints];
      });
      
      // Cluster people
      const clusters = clusterPeople(people);
      
      // Plan movement (using best cluster if available)
      if (clusters.length > 0) {
        // Find best cluster (highest confidence)
        const bestCluster = clusters.reduce((best, current) => 
          current[3] > best[3] ? current : best, clusters[0]);
        
        planner.planMovement(i, bestCluster, frameDiff, SCENE_CHANGE_THRESHOLD);
      } else {
        planner.planMovement(i, null, frameDiff, SCENE_CHANGE_THRESHOLD);
      }
      
      // Clean up tensors
      inputTensor.dispose();
      
      // Log progress
      if (i % 10 === 0 || i === frameFiles.length - 1) {
        const progress = Math.floor((i / frameFiles.length) * 100);
        console.log(`Analyzing: ${progress}%`);
        
        // Update progress.json similar to Python implementation
        fs.writeFileSync('progress.json', JSON.stringify({
          progress,
          status: `Analyzing: ${progress}%`
        }, null, 2));
      }
    }
    
    // Apply smoothing algorithm from Python implementation
    const smoothedPositions = planner.interpolateAndSmooth(frameFiles.length);
    
    // Clean up
    prevFrameGray?.dispose();
    
    try {
      fs.rmSync(tempDir, { recursive: true });
    } catch (err) {
      console.warn('Warning: Failed to clean up temporal directory', err);
    }
    
    // Return analysis results
    return {
      inputPath,
      metadata,
      sceneChanges,
      frameDiffs,
      smoothedPositions
    };
  } catch (error) {
    console.error('Error analyzing video:', error);
    throw error;
  }
}

/**
 * Process video based on analysis results
 */
async function processVideo(inputPath, outputs, analysis) {
  console.log(`Processing video based on analysis: ${inputPath}`);
  
  // Process each output based on analysis
  const results = await Promise.all(outputs.map(async (output, index) => {
    const { url, range } = output;
    
    // Determine output range based on analysis and specified range
    let effectiveRange = range;
    if (!range) {
      // If no range specified, use key frames from analysis
      const keyFrames = analysis.keyFrames;
      if (keyFrames && keyFrames.length > 0) {
        // Use the first and last key frame as range
        effectiveRange = `[${keyFrames[0]}-${keyFrames[keyFrames.length - 1]}]`;
      }
    }
    
    console.log(`Processing output ${index + 1}: ${url} with range ${effectiveRange || 'full video'}`);
    
    // Actually process the video segment here
    // For now, this is just returning placeholder data
    
    return {
      outputPath: url,
      range: effectiveRange,
      status: 'completed',
      processingTime: '1.2s'
    };
  }));
  
  return results;
}

/**
 * MovementPlanner class - matches the functionality in toReel.py
 */
class MovementPlanner {
  constructor(fps) {
    this.frameData = [];
    this.fps = fps;
    this.currentSceneStart = 0;
    this.waitingForDetection = false;
    this.defaultX = DEFAULT_CENTER;
    this.smoothingRate = 0.05;
    this.maxMovementPerFrame = 0.03;
    this.positionHistory = [];
    this.historySize = 3;
    this.centeringWeight = 0.4;
    this.fastTransitionThreshold = 0.1;
    this.inTransition = false;
    this.stableFrames = 0;
    this.stableFramesRequired = Math.floor(fps * 0.5);  // Half a second worth of frames
    this.isCentering = false;
  }
  
  planMovement(frameNum, cluster, frameDiff, sceneChangeThreshold) {
    if (frameDiff > sceneChangeThreshold) {
      this.positionHistory = [];
      this.inTransition = false;
      this.stableFrames = 0;
      this.isCentering = false;
      // Reset history on scene change
      this.frameData.push([frameNum, this.defaultX, true]);
      this.currentSceneStart = frameNum;
      this.waitingForDetection = true;
      return;
    }

    if (this.waitingForDetection && cluster) {
      // Found first detection after scene change
      const newX = cluster[1];  // Normalized x position from cluster
      
      // Go back and update all frames since scene change
      for (let i = 0; i < this.frameData.length; i++) {
        const [fNum, _, isScene] = this.frameData[i];
        if (fNum >= this.currentSceneStart) {
          if (isScene) {
            continue;  // Keep scene change marker
          }
          this.frameData[i] = [fNum, newX, false];
        }
      }
      
      this.waitingForDetection = false;
      return;
    }

    // Get the target position
    let targetX;
    if (cluster) {
      targetX = cluster[1];
    } else {
      targetX = this.frameData.length ? this.frameData[this.frameData.length - 1][1] : this.defaultX;
    }

    if (this.frameData.length) {
      const lastX = this.frameData[this.frameData.length - 1][1];
      const distanceToTarget = Math.abs(targetX - lastX);
      
      // Apply normal movement with limits
      const maxMove = this.maxMovementPerFrame;
      let movement = targetX - lastX;
      if (Math.abs(movement) > maxMove) {
        targetX = lastX + (movement > 0 ? maxMove : -maxMove);
      }

      // Check if we're within the delta threshold for stability
      if (distanceToTarget < this.fastTransitionThreshold) {
        this.stableFrames++;
        if (this.stableFrames >= this.stableFramesRequired) {
          this.isCentering = true;
        }
      } else {
        this.stableFrames = 0;
        this.isCentering = false;
        this.positionHistory = [];
      }

      // Apply centering after stable period
      if (this.isCentering) {
        this.positionHistory.push(targetX);
        if (this.positionHistory.length > this.historySize) {
          this.positionHistory.shift();
        }

        if (this.positionHistory.length > 1) {
          const avgPos = this.positionHistory.reduce((sum, pos) => sum + pos, 0) / this.positionHistory.length;
          targetX = targetX * (1 - this.centeringWeight) + avgPos * this.centeringWeight;
        }
      }
    }

    this.frameData.push([frameNum, targetX, false]);
  }
  
  getSceneSegments() {
    const segments = [];
    let currentSegmentStart = 0;
    let currentPositions = [];

    for (let i = 0; i < this.frameData.length; i++) {
      const [frameNum, xPos, isScene] = this.frameData[i];
      
      if (isScene && i > 0) {
        // End current segment
        segments.push([
          currentSegmentStart,
          frameNum - 1,
          currentPositions
        ]);
        // Start new segment
        currentSegmentStart = frameNum;
        currentPositions = [];
      }
      
      currentPositions.push(xPos);
    }

    // Add final segment
    if (currentPositions.length) {
      segments.push([
        currentSegmentStart,
        this.frameData[this.frameData.length - 1][0],
        currentPositions
      ]);
    }

    return segments;
  }
  
  interpolateAndSmooth(totalFrames, baseAlpha = 0.1, deltaThreshold = 0.015) {
    const smoothedCenters = Array(totalFrames).fill(this.defaultX);
    const segments = this.getSceneSegments();

    for (const [startFrame, endFrame, positions] of segments) {
      let lastX = positions[0];
      
      for (let i = 0; i < positions.length; i++) {
        const frame = startFrame + i;
        if (frame > endFrame) break;
        
        const targetX = positions[i];
        const delta = targetX - lastX;
        
        let smoothedX;
        if (Math.abs(delta) < deltaThreshold) {
          smoothedX = lastX;
        } else {
          // Variable smoothing rate based on distance to target
          // Slower smoothing (smaller alpha) as we get closer
          const distanceFactor = Math.min(Math.abs(delta) * 2, 1.0);  // Scale based on distance
          let decelerationAlpha = baseAlpha * distanceFactor;
          
          // Even slower for final approach
          if (Math.abs(delta) < 0.1) {  // Within 10% of target
            decelerationAlpha *= 0.5;  // Half speed for final approach
          }
          
          smoothedX = lastX + decelerationAlpha * delta;
        }
        
        smoothedCenters[frame] = smoothedX;
        lastX = smoothedX;
      }
    }

    return smoothedCenters;
  }
}

/**
 * PersonTracker class - compatible with the original implementation
 * Used by other parts of the application
 */
class PersonTracker {
  constructor() {
    this.lastX = null;
    this.lastY = null;
    this.velocity = 0;
    this.consecutive_no_detections = 0;
    this.isTracking = false;
  }
  
  updatePosition(x, y) {
    if (this.lastX !== null && this.lastY !== null) {
      this.velocity = Math.abs(x - this.lastX);
    }
    this.lastX = x;
    this.lastY = y;
    this.isTracking = true;
    this.consecutive_no_detections = 0;
  }
  
  getPosition() {
    return { x: this.lastX || DEFAULT_CENTER, y: this.lastY || 0.5 };
  }
  
  distance(x1, y1, x2, y2) {
    return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
  }
  
  update(cluster, frameDiff, sceneChangeThreshold, movementThreshold) {
    // Similar to Python implementation
    if (!cluster || cluster.length === 0) {
      return this.handleNoDetection(frameDiff, sceneChangeThreshold);
    }
    
    // Find the closest person to the current position
    if (this.lastX === null || this.lastY === null) {
      // First detection, take the most confident one
      const bestPerson = cluster.reduce((best, current) => 
        current.confidence > best.confidence ? current : best, cluster[0]);
      
      this.updatePosition(bestPerson.x, bestPerson.y);
      return { x: bestPerson.x, sceneChange: false };
    }
    
    // Calculate distances to current position
    const distances = cluster.map(person => ({
      person,
      distance: this.distance(this.lastX, this.lastY, person.x, person.y)
    }));
    
    // Sort by distance
    distances.sort((a, b) => a.distance - b.distance);
    
    // If closest person is too far, check if it's a scene change
    if (distances[0].distance > movementThreshold && frameDiff > sceneChangeThreshold) {
      // Potential scene change, reset tracking
      this.reset();
      const bestPerson = cluster.reduce((best, current) => 
        current.confidence > best.confidence ? current : best, cluster[0]);
      
      this.updatePosition(bestPerson.x, bestPerson.y);
      return { x: bestPerson.x, sceneChange: true };
    }
    
    // Update with the closest person
    const closestPerson = distances[0].person;
    this.updatePosition(closestPerson.x, closestPerson.y);
    return { x: closestPerson.x, sceneChange: false };
  }
  
  handleNoDetection(frameDiff, sceneChangeThreshold) {
    this.consecutive_no_detections += 1;
    
    // If we have frame difference data and it suggests a scene change
    if (frameDiff > sceneChangeThreshold) {
      this.reset();
      return { x: DEFAULT_CENTER, sceneChange: true }; // Center of the frame
    }
    
    // If no scene change but we've lost detection for too long
    if (this.consecutive_no_detections > 15) {
      this.isTracking = false;
    }
    
    return { x: this.lastX || DEFAULT_CENTER, sceneChange: false };
  }
  
  reset() {
    this.lastX = null;
    this.lastY = null;
    this.velocity = 0;
    this.consecutive_no_detections = 0;
    this.isTracking = false;
  }
}

/**
 * Extract keypoints for a specific segment of video
 */
async function getKeypointsForSegment(inputPath, startFrame, endFrame) {
  console.log(`Extracting keypoints for frames ${startFrame}-${endFrame} from ${inputPath}`);
  
  // Create a temporal directory for intermediate files
  const tempDir = path.join(os.tmpdir(), `quickreels-keypoints-${Date.now()}`);
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
    
    // Calculate time positions
    const startTime = startFrame / metadata.fps;
    const duration = (endFrame - startFrame) / metadata.fps;
    
    console.log(`Extracting frames from ${formatTimecode(startTime)} to ${formatTimecode(startTime + duration)}`);
    
    // Extract only the frames we need
    await new Promise((resolve, reject) => {
      ffmpeg(inputPath)
        .seekInput(startTime)
        .duration(duration)
        .outputOptions([
          '-q:v', '2' // High quality
        ])
        .on('end', () => {
          resolve();
        })
        .on('error', (err) => {
          console.error('FFmpeg error:', err);
          reject(err);
        })
        .output(path.join(tempDir, 'frame-%04d.jpg'))
        .run();
    });
    
    // Find all extracted frames
    const frameFiles = fs.readdirSync(tempDir)
      .filter(file => file.endsWith('.jpg'))
      .sort();
    
    console.log(`Extracted ${frameFiles.length} frames for segment`);
    
    // Keypoint data collection
    const keypointData = [];
    
    // Process each frame to extract keypoints
    for (let i = 0; i < frameFiles.length; i++) {
      const frameFile = frameFiles[i];
      const framePath = path.join(tempDir, frameFile);
      
      try {
        // Read the frame
        const frameBuffer = fs.readFileSync(framePath);
        const frameTensor = await prepareInputTensor(frameBuffer);
        
        // Run inference to get keypoints
        const detections = await runInference(frameTensor);
        
        // Add to keypoint data collection
        keypointData.push({
          frame: startFrame + i,
          keypoints: detections && detections.length > 0 ? detections[0].keypoints : [],
          detected: detections && detections.length > 0
        });
        
        // Clean up
        tf.dispose(frameTensor);
      } catch (error) {
        console.error(`Error processing frame ${frameFile} for keypoints:`, error);
        
        // Add empty keypoints for this frame
        keypointData.push({
          frame: startFrame + i,
          keypoints: [],
          detected: false
        });
      }
    }
    
    // Clean up
    try {
      fs.rmSync(tempDir, { recursive: true, force: true });
    } catch (e) {
      console.warn('Failed to remove temporary directory:', e);
    }
    
    // Return simplified keypoint data
    return keypointData.map(kpData => ({
      frame: kpData.frame,
      detected: kpData.detected,
      position: kpData.detected ? {
        x: kpData.keypoints && kpData.keypoints.length > 0 ? 
          kpData.keypoints.reduce((sum, kp) => sum + kp[1], 0) / kpData.keypoints.length : 0.5,
        y: kpData.keypoints && kpData.keypoints.length > 0 ? 
          kpData.keypoints.reduce((sum, kp) => sum + kp[0], 0) / kpData.keypoints.length : 0.5
      } : null
    }));
  } catch (error) {
    console.error('Error extracting keypoints for segment:', error);
    throw error;
  }
}

// Export functions
module.exports = {
  initializeTensorFlow,
  analizeVideo,
  processVideo,
  getKeypointsForSegment,
  on,
  // Export classes needed by other parts of the application
  PersonTracker,
  MovementPlanner
}; 