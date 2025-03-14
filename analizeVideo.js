const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const ffmpeg = require('fluent-ffmpeg');
const ffmpegPath = require('@ffmpeg-installer/ffmpeg').path;
const EventEmitter = require('events');

// Import configuration from central config.js
const config = require('./config');

// Set the ffmpeg path
ffmpeg.setFfmpegPath(ffmpegPath);

// Create event emitter for frame processing events
const eventEmitter = new EventEmitter();

// Global model variable
let model = null;

// Constants for detection and processing
const DETECTION_THRESHOLD = config.DETECTION_THRESHOLD;
const PERSON_CLASS_ID = config.PERSON_CLASS_ID;
const SCENE_CHANGE_THRESHOLD = config.SCENE_CHANGE_THRESHOLD;
const DEFAULT_CENTER = config.DEFAULT_CENTER;
const MOVE_NET_INPUT_SIZE = config.MOVE_NET_INPUT_SIZE;

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
 * @param {string} modelPath - Path to the model directory
 */
async function initializeTensorFlow(modelPath) {
  try {
    console.log(`Initializing TensorFlow with model at: ${modelPath}`);
    // Check if model directory exists
    if (!fs.existsSync(modelPath)) {
      throw new Error(`Model directory not found: ${modelPath}`);
    }
    
    // Look for model files
    const modelJsonPath = path.join(modelPath, 'model.json');
    const savedModelPath = path.join(modelPath, 'saved_model.pb');
    
    // First try loading as GraphModel (model.json)
    if (fs.existsSync(modelJsonPath)) {
      try {
        console.log("Found model.json, loading as GraphModel");
        model = await tf.loadGraphModel(`file://${modelJsonPath}`);
        console.log("Model loaded successfully");
        return true;
      } catch (err) {
        console.log(`Error loading GraphModel: ${err.message}`);
        // Continue to next method
      }
    }
    
    // Then try loading as SavedModel
    if (fs.existsSync(savedModelPath)) {
      try {
        console.log("Found saved_model.pb, loading as SavedModel");
        model = await tf.node.loadSavedModel(modelPath);
        console.log("Model loaded successfully");
        return true;
      } catch (err) {
        throw err;
      }
    }
    
    throw new Error('No compatible model files found in the specified directory');
  } catch (error) {
    console.error(`Error initializing TensorFlow: ${error.message}`);
    return false;
  }
}

/**
 * Prepare input tensor from image buffer for MoveNet model
 */
async function prepareInputTensor(frameBuffer) {
  try {
    // Decode image to tensor (3 channels for RGB)
    const imageTensor = tf.node.decodeImage(frameBuffer, 3);
    
    // Resize to MoveNet input size (192x192)
    const resized = tf.image.resizeBilinear(imageTensor, MOVE_NET_INPUT_SIZE);
    
    // Convert to int32
    const convertedTensor = tf.cast(resized, 'int32');
    
    // Add batch dimension
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
    console.error(`Inference error: ${error.message}`);
    return [];
  }
}

/**
 * Cluster people detections
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
  
  // Convert clusters to format expected
  const mergedPeople = clusters.map((cluster, i) => {
    const avgX = cluster.sumX / cluster.count;
    const avgY = cluster.sumY / cluster.count;
    return [i, avgX, avgY, cluster.maxConf];
  });
  
  return mergedPeople;
}

/**
 * MovementPlanner class - handles camera movement planning
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
      if (positions.length === 0) continue;
      
      let lastX = positions[0];
      
      for (let i = 0; i < positions.length; i++) {
        const frame = startFrame + i;
        if (frame > endFrame || frame >= totalFrames) break;
        
        const targetX = positions[i];
        const delta = targetX - lastX;
        
        let smoothedX;
        if (Math.abs(delta) < deltaThreshold) {
          smoothedX = lastX;
        } else {
          // Variable smoothing rate based on distance to target
          const distanceFactor = Math.min(Math.abs(delta) * 2, 1.0);
          let decelerationAlpha = baseAlpha * distanceFactor;
          
          // Even slower for final approach
          if (Math.abs(delta) < 0.1) {
            decelerationAlpha *= 0.5;
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
 * Analyze video and extract camera tracking data
 * @param {string} inputPath - Path to input video
 * @param {string} outputPath - Path for output (not used in analysis)
 * @param {Object} options - Optional configuration
 * @returns {Object} - Analysis results
 */
async function analizeVideo(inputPath, outputPath, options = {}) {
  try {
    console.log(`Analyzing video: ${inputPath}`);
    
    // Create temp directory for frames
    const processingId = path.basename(outputPath, path.extname(outputPath));
    const tempDir = path.join(process.cwd(), 'temp', processingId);
    const framesDir = path.join(tempDir, 'frames');
    const analysisFramesDir = path.join(framesDir, 'analysis');
    
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }
    if (!fs.existsSync(framesDir)) {
      fs.mkdirSync(framesDir, { recursive: true });
    }
    if (!fs.existsSync(analysisFramesDir)) {
      fs.mkdirSync(analysisFramesDir, { recursive: true });
    }

    // Initialize TensorFlow model
    const modelPath = options.modelPath || path.join(process.cwd(), 'model');
    await initializeTensorFlow(modelPath);
    
    console.log("Extracting video metadata...");
    
    // Extract video metadata
    const metadata = await new Promise((resolve, reject) => {
      ffmpeg.ffprobe(inputPath, (err, metadata) => {
        if (err) return reject(err);
        
        const { width, height, duration } = metadata.streams[0];
        let fps = 30; // Default
        
        // Parse frame rate
        if (metadata.streams[0].r_frame_rate) {
          const [num, den] = metadata.streams[0].r_frame_rate.split('/').map(Number);
          fps = num / (den || 1);
        }
        
        console.log(`Video metadata: ${width}x${height}, ${fps} fps, ${duration}s duration`);
        
        resolve({
          width,
          height,
          duration,
          fps,
          totalFrames: Math.round(duration * fps)
        });
      });
    });
    
    console.log("Extracting frames for analysis...");
    
    // Extract frames for analysis
    await new Promise((resolve, reject) => {
      ffmpeg(inputPath)
        .outputOptions(['-vf', `fps=${metadata.fps}`, '-q:v', '1'])
        .output(path.join(analysisFramesDir, 'frame_%04d.jpg'))
        .on('progress', (progress) => {
          console.log(`Extracting frames: ${Math.floor(progress.percent || 0)}%`);
        })
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
    
    console.log(`Found ${frameFiles.length} frames for analysis`);
    
    // Initialize movement planner
    const planner = new MovementPlanner(metadata.fps);
    let prevFrameGray = null;
    const frameDiffs = [];
    const sceneChanges = [];
    
    console.log("Starting frame analysis...");
    
    // First pass: Analyze each frame
    for (let i = 0; i < frameFiles.length; i++) {
      const frameFile = path.join(analysisFramesDir, frameFiles[i]);
      const imageBuffer = fs.readFileSync(frameFile);
      
      // Emit frame processing event
      eventEmitter.emit('frameProcessing', { frameNum: i, total: frameFiles.length });
      
      // Calculate and log progress percentage
      const progressPercent = Math.floor((i / frameFiles.length) * 100);
      if (i % 10 === 0 || i === frameFiles.length - 1) {
        console.log(`Analyzing frame ${i+1}/${frameFiles.length} (${progressPercent}%)`);
      }
      
      // Prepare grayscale image for scene change detection
      const grayscaleTensor = tf.node.decodeImage(imageBuffer, 1);  // 1 channel (grayscale)
      
      // Calculate frame difference (MSE)
      let frameDiff = 0;
      if (prevFrameGray) {
        frameDiff = calculateMSE(prevFrameGray, grayscaleTensor);
        frameDiffs.push(frameDiff);
        
        // Detect scene changes
        if (frameDiff > SCENE_CHANGE_THRESHOLD) {
          console.log(`Scene change detected at frame ${i}`);
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
      
      // Format detections for clusterPeople
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
    }
    
    console.log("Frame analysis complete. Applying smoothing algorithm...");
    
    // Apply smoothing algorithm
    const smoothedPositions = planner.interpolateAndSmooth(frameFiles.length);
    
    console.log("Analysis complete");
    
    // Clean up
    prevFrameGray?.dispose();
    
    // Emit analysis complete event
    eventEmitter.emit('analysisComplete', {
      inputPath,
      outputPath,
      sceneChanges,
      smoothedPositions
    });
    
    // Return analysis results
    return {
      inputPath,
      outputPath,
      metadata,
      sceneChanges,
      frameDiffs,
      smoothedPositions,
      tempDir
    };
  } catch (error) {
    console.error('Error analyzing video:', error);
    throw error;
  }
}

module.exports = {
  analizeVideo,
  initializeTensorFlow,
  MovementPlanner,
  on
}; 