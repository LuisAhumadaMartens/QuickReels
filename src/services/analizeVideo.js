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
const { generateRandomId } = require('./processVideo');
const { updateProgress } = require('../utils/progressTracker');

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
  try {
    // Check if model directory exists
    if (!fs.existsSync(MODEL_PATH)) {
      throw new Error(`Model directory not found: ${MODEL_PATH}`);
    }
    
    // Look for model files
    const modelJsonPath = path.join(MODEL_PATH, 'model.json');
    const savedModelPath = path.join(MODEL_PATH, 'saved_model.pb');
    
    // First try loading as GraphModel (model.json)
    if (fs.existsSync(modelJsonPath)) {
      try {
        model = await tf.loadGraphModel(`file://${modelJsonPath}`);
        return true;
      } catch (err) {
        // Continue to next method
      }
    }
    
    // Then try loading as SavedModel
    if (fs.existsSync(savedModelPath)) {
      try {
        model = await tf.node.loadSavedModel(MODEL_PATH);
        return true;
      } catch (err) {
        throw err;
      }
    }
    
    throw new Error('No compatible model files found in the specified directory');
  } catch (error) {
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
    
    // Resize to MoveNet input size (192x192) - matches Python's cv2.resize
    const resized = tf.image.resizeBilinear(imageTensor, MOVE_NET_INPUT_SIZE);
    
    // Convert to int32 (matches Python's tf.convert_to_tensor(..., dtype=tf.int32))
    const convertedTensor = tf.cast(resized, 'int32');
    
    // Add batch dimension (matches Python's tf.expand_dims(input_tensor, axis=0))
    const batchedTensor = tf.expandDims(convertedTensor, 0);
    
    // Clean up intermediate tensors
    tf.dispose([imageTensor, resized, convertedTensor]);
    
    return batchedTensor;
  } catch (error) {
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
    let result = await model.predict(tensor);
    
    // Extract keypoints from the result
    let keypoints = [];
    if (result.output_0) {
      // SavedModel format
      keypoints = result.output_0.arraySync()[0];
    } else if (result.arraySync) {
      // GraphModel format
      keypoints = result.arraySync()[0];
    } else {
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
        
        detections.push(detection);
      }
    }
    
    return detections;
  } catch (error) {
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
 * @param {string} [jobId] - Optional job ID for progress tracking
 * @returns {Object} - Analysis results
 */
async function analizeVideo(inputPath, jobId = null) {
  const processingId = jobId || generateRandomId();
  console.log(`Job ID [${processingId}]: Analyzing video`);
  
  try {
    // If we have a job ID, update progress
    if (jobId) {
      updateProgress(jobId, {
        analysis: { progress: 0, status: "Initializing analysis..." }
      });
    }
    
    // Initialize TensorFlow and load model
    await initializeTensorFlow();
    
    // Create project-relative temp directory structure
    const projectRoot = process.cwd();
    const tempBaseDir = path.join(projectRoot, 'temp');
    const tempDir = path.join(tempBaseDir, processingId);
    const framesDir = path.join(tempDir, 'frames');
    const analysisFramesDir = path.join(framesDir, 'analysis');
    const tempInputPath = path.join(tempDir, 'input.mp4');
    
    // Create directories
    if (!fs.existsSync(tempBaseDir)) {
      fs.mkdirSync(tempBaseDir, { recursive: true });
    }
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }
    if (!fs.existsSync(framesDir)) {
      fs.mkdirSync(framesDir, { recursive: true });
    }
    if (!fs.existsSync(analysisFramesDir)) {
      fs.mkdirSync(analysisFramesDir, { recursive: true });
    }
    
    // Copy input file to temp directory
    fs.copyFileSync(inputPath, tempInputPath);
    
    // Extract video metadata - still at 0% progress, just update the status text
    if (jobId) {
      updateProgress(jobId, {
        analysis: { progress: 0, status: "Extracting video metadata..." }
      });
    }
    
    const metadata = await new Promise((resolve, reject) => {
      ffmpeg.ffprobe(tempInputPath, (err, metadata) => {
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
    
    // Extract frames for analysis
    if (jobId) {
      updateProgress(jobId, {
        analysis: { progress: 0, status: "Extracting frames for analysis..." }
      });
    }
    
    await new Promise((resolve, reject) => {
      ffmpeg(tempInputPath)
        .outputOptions(['-vf', `fps=${metadata.fps}`, '-q:v', '1'])
        .output(path.join(analysisFramesDir, 'frame_%04d.jpg'))
        .on('end', resolve)
        .on('error', reject)
        .run();
    });

    // Get all frame files sorted by number
    const frameFiles = fs.readdirSync(analysisFramesDir)
      .filter(file => file.startsWith('frame_') && file.endsWith('.jpg'))
      .sort((a, b) => {
        const numA = parseInt(a.match(/frame_(\d+)/)[1]);
        const numB = parseInt(b.match(/frame_(\d+)/)[1]);
        return numA - numB;
      });
    
    console.log(`Job ID [${processingId}]: Found ${frameFiles.length} frames for analysis`);
    
    // Initialize movement planner
    const planner = new MovementPlanner(metadata.fps);
    let prevFrameGray = null;
    const frameDiffs = [];
    const sceneChanges = [];
    
    // Update progress for analysis starting
    if (jobId) {
      updateProgress(jobId, {
        analysis: { progress: 0, status: "Analyzing frames: 0%" }
      });
    }
    
    // Track current percentage for progress updates
    let lastReportedProgress = 0;
    
    // First pass: Analyze each frame (like in Python)
    // Calculate the interval between progress updates to get exactly 100 steps
    const progressInterval = Math.max(1, Math.floor(frameFiles.length / 100));

    for (let i = 0; i < frameFiles.length; i++) {
      const frameFile = path.join(analysisFramesDir, frameFiles[i]);
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
      
      // Log progress exactly 100 times during the process (once per percent)
      if (i % progressInterval === 0 || i === frameFiles.length - 1) {
        const progress = Math.floor((i / frameFiles.length) * 100);
        
        // Only log if this is a new percentage
        if (progress !== lastReportedProgress) {
          // Update progress with the direct percentage value
          if (jobId) {
            updateProgress(jobId, {
              analysis: { 
                progress: progress, 
                status: `Analyzing frames: ${progress}%` 
              }
            });
            
            // Don't log here - updateProgress() likely already logs this message
            lastReportedProgress = progress;
          }
        }
      }
    }
    
    // Apply smoothing algorithm from Python implementation
    const smoothedPositions = planner.interpolateAndSmooth(frameFiles.length);
    
    // Update progress for completion of analysis
    if (jobId) {
      updateProgress(jobId, {
        analysis: { progress: 100, status: "Analysis complete" }
      });
    }
    
    // Clean up
    prevFrameGray?.dispose();
    
    // Return analysis results
    return {
      inputPath: tempInputPath,
      metadata,
      sceneChanges,
      frameDiffs,
      smoothedPositions,
      jobId: processingId,
      tempDir: tempDir
    };
  } catch (error) {
    console.error(`Job ID [${processingId}]: Error analyzing video: ${error.message}`);
    
    // Update progress with error if we have a job ID
    if (jobId) {
      updateProgress(jobId, {
        analysis: { progress: -1, status: `Error: ${error.message}` }
      });
    }
    
    throw error;
  }
}

/**
 * Process video based on analysis results
 */
async function processVideo(inputPath, outputs, analysis) {

  
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
          reject(err);
        })
        .output(path.join(tempDir, 'frame-%04d.jpg'))
        .run();
    });
    
    // Find all extracted frames
    const frameFiles = fs.readdirSync(tempDir)
      .filter(file => file.endsWith('.jpg'))
      .sort();
    
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
      // Silently continue if cleanup fails
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