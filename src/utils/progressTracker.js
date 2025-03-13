/**
 * Progress Tracking Service
 * This module handles progress tracking for video analysis and processing tasks
 * with optimized file I/O and in-memory caching for better performance.
 */

const fs = require('fs').promises;
const path = require('path');
const fsSync = require('fs'); // For sync operations only where absolutely needed
const config = require('../config/config');

// In-memory progress cache
const progressCache = new Map();
// Set of completed jobs
const completedJobs = new Set();
// Queue of pending writes
let writeQueue = [];
// Write in progress flag
let isWriting = false;
// Throttle setting
const WRITE_THROTTLE_MS = 1000; // 1 second

/**
 * Centralized logging for job progress and status 
 * @param {string} jobId - The job ID
 * @param {string} message - The message to log
 * @param {boolean} isError - Whether this is an error message
 */
function logJobMessage(jobId, message, isError = false) {
  const logPrefix = `Job ID [${jobId}]:`;
  if (isError) {
    console.error(`${logPrefix} ${message}`);
  } else {
    console.log(`${logPrefix} ${message}`);
  }
}

/**
 * Log an error message for a job
 * @param {string} jobId - The job ID
 * @param {string} message - The error message
 * @param {Error} [error] - Optional error object for stack trace
 */
function logJobError(jobId, message, error = null) {
  logJobMessage(jobId, message, true);
  if (error && error.stack) {
    console.error(`Job ID [${jobId}]: Stack trace: ${error.stack}`);
  }
}

/**
 * Update progress in memory and queue a write (async implementation)
 * @param {string} jobId - The job ID
 * @param {Object} statusUpdate - The status update object
 */
async function updateProgressAsync(jobId, statusUpdate) {
  // Don't process updates for completed jobs
  if (completedJobs.has(jobId)) {
    return;
  }
  
  // Update in-memory cache first
  if (!progressCache.has(jobId)) {
    progressCache.set(jobId, {
      status: "Initializing...",
      analysis: 0,
      processing: 0,
      encoding: 0,
      audio: 0
    });
  }
  
  // Apply updates to in-memory cache
  const jobData = progressCache.get(jobId);
  
  // Update each phase if needed - helper function
  const updatePhase = (phase, phaseUpdate) => {
    if (!phaseUpdate) return false;
    
    let wasUpdated = false;
    
    // Update progress if provided and higher than current (or error state)
    if (phaseUpdate.progress !== undefined) {
      const newProgress = phaseUpdate.progress;
      if (newProgress === -1 || newProgress >= jobData[phase]) {
        jobData[phase] = newProgress;
        wasUpdated = true;
      }
    }
    
    // Update status message if provided
    if (phaseUpdate.status) {
      jobData.status = phaseUpdate.status;
      wasUpdated = true;
    }
    
    return wasUpdated;
  };
  
  // Update each phase
  let wasUpdated = false;
  if (statusUpdate.analysis) wasUpdated = updatePhase('analysis', statusUpdate.analysis) || wasUpdated;
  if (statusUpdate.processing) wasUpdated = updatePhase('processing', statusUpdate.processing) || wasUpdated;
  if (statusUpdate.encoding) wasUpdated = updatePhase('encoding', statusUpdate.encoding) || wasUpdated;
  if (statusUpdate.audio) wasUpdated = updatePhase('audio', statusUpdate.audio) || wasUpdated;
  
  // Check if videoGenerated flag is set
  if (statusUpdate.videoGenerated !== undefined) {
    jobData.videoGenerated = statusUpdate.videoGenerated;
    wasUpdated = true;
  }
  
  // Check for job completion
  const isComplete = statusUpdate.processing && 
                      statusUpdate.processing.status === "Processing complete";
  
  const hasError = statusUpdate.processing && 
                    statusUpdate.processing.status && 
                    statusUpdate.processing.status.startsWith("Error");
  
  // Handle job completion
  if (isComplete || hasError) {
    if (isComplete) {
      // Set final status for regular completion
      jobData.status = "Video generation complete";
      logJobMessage(jobId, "Video generation complete");
      
      // Ensure all phases are set to 100% for completion
      if (!hasError) {
        jobData.analysis = 100;
        jobData.processing = 100;
        jobData.encoding = 100;
        jobData.audio = 100;
      }
    }
    
    // Mark job as completed
    completedJobs.add(jobId);
    wasUpdated = true;
  }
  
  // If anything was updated, queue a write
  if (wasUpdated) {
    queueProgressWrite();
  }
}

/**
 * Synchronous version of updateProgress for backward compatibility
 * @param {string} jobId - The job ID
 * @param {Object} statusUpdate - The status update object
 */
function updateProgress(jobId, statusUpdate) {
  try {
    // If the job has been marked as completed, ignore further updates
    if (completedJobs.has(jobId)) {
      return;
    }
    
    // Read existing progress data
    let progressData = {};
    const progressFilePath = config.PROGRESS_FILE;
    
    if (fsSync.existsSync(progressFilePath)) {
      progressData = JSON.parse(fsSync.readFileSync(progressFilePath, 'utf-8'));
    }
    
    // Initialize job entry if it doesn't exist
    if (!progressData[jobId]) {
      progressData[jobId] = {
        status: "Initializing...",
        analysis: 0,
        processing: 0,
        encoding: 0,
        audio: 0
      };
    }
    
    // Generalized function to update a specific phase
    const updatePhase = (phase, phaseUpdate) => {
      if (!phaseUpdate) return false;
      
      let wasUpdated = false;
      
      // Update progress if provided and higher than current (or error state)
      if (phaseUpdate.progress !== undefined) {
        const newProgress = phaseUpdate.progress;
        if (newProgress === -1 || newProgress >= progressData[jobId][phase]) {
          progressData[jobId][phase] = newProgress;
          wasUpdated = true;
        }
      }
      
      // Update status message if provided
      if (phaseUpdate.status) {
        progressData[jobId].status = phaseUpdate.status;
        // Centralized logging - determine if this is an error message
        const isError = phaseUpdate.status.toLowerCase().includes('error');
        logJobMessage(jobId, phaseUpdate.status, isError);
        wasUpdated = true;
      }
      
      return wasUpdated;
    };
    
    // Update each phase if needed
    const analysisUpdated = updatePhase('analysis', statusUpdate.analysis);
    const processingUpdated = updatePhase('processing', statusUpdate.processing);
    const encodingUpdated = updatePhase('encoding', statusUpdate.encoding);
    const audioUpdated = updatePhase('audio', statusUpdate.audio);
    
    // Check if videoGenerated flag is set
    const videoGeneratedUpdated = statusUpdate.videoGenerated !== undefined;
    if (videoGeneratedUpdated) {
      progressData[jobId].videoGenerated = statusUpdate.videoGenerated;
    }
    
    // Only write to file if anything changed
    if (analysisUpdated || processingUpdated || encodingUpdated || audioUpdated || videoGeneratedUpdated) {
      // Check for completion conditions
      const isComplete = statusUpdate.processing && 
                         statusUpdate.processing.status === "Processing complete";
      
      const hasError = statusUpdate.processing && 
                       statusUpdate.processing.status && 
                       statusUpdate.processing.status.startsWith("Error");
      
      // Handle job completion - regular completion or error
      if (isComplete || hasError) {
        if (isComplete) {
          // Set final status for regular completion
          progressData[jobId].status = "Video generation complete";
          logJobMessage(jobId, "Video generation complete");
          
          // Ensure all phases are set to 100% for completion
          if (!hasError) {
            progressData[jobId].analysis = 100;
            progressData[jobId].processing = 100;
            progressData[jobId].encoding = 100;
            progressData[jobId].audio = 100;
          }
        }
        
        // Write the progress file with the completion status
        fsSync.writeFileSync(progressFilePath, JSON.stringify(progressData, null, 2));
        
        // Mark job as completed
        completedJobs.add(jobId);
        
        // Remove the job from progress tracking
        delete progressData[jobId];
        logJobMessage(jobId, `Job ${hasError ? 'completed with error' : 'completed'}, removed from tracking`);
        
        // Write the updated file without the job
        fsSync.writeFileSync(progressFilePath, JSON.stringify(progressData, null, 2));
      } else {
        // Regular update - write once at the end
        fsSync.writeFileSync(progressFilePath, JSON.stringify(progressData, null, 2));
      }
    }
  } catch (err) {
    logJobError(jobId, `Warning: Could not update progress: ${err.message}`, err);
  }
  
  // Also queue an async update to keep the cache in sync
  updateProgressAsync(jobId, statusUpdate).catch(err => {
    logJobError(jobId, `Warning: Async progress update failed: ${err.message}`, err);
  });
}

/**
 * Queue a progress write with throttling
 */
function queueProgressWrite() {
  // Add to write queue if not already pending
  if (writeQueue.length === 0) {
    writeQueue.push(Date.now());
    
    // Process the queue if not already processing
    if (!isWriting) {
      processWriteQueue();
    }
  }
}

/**
 * Process the write queue with throttling
 */
async function processWriteQueue() {
  if (writeQueue.length === 0) {
    isWriting = false;
    return;
  }
  
  isWriting = true;
  
  try {
    // Write current state to disk
    await writeProgressToDisk();
    
    // Clear the queue
    writeQueue = [];
    
    // Schedule next check after delay
    setTimeout(() => {
      processWriteQueue();
    }, WRITE_THROTTLE_MS);
  } catch (error) {
    console.error('Error writing progress file:', error);
    isWriting = false;
    
    // Retry after delay
    setTimeout(() => {
      if (writeQueue.length > 0) {
        processWriteQueue();
      }
    }, WRITE_THROTTLE_MS);
  }
}

/**
 * Write the current progress state to disk
 */
async function writeProgressToDisk() {
  try {
    const progressFilePath = path.resolve(process.cwd(), config.PROGRESS_FILE);
    
    // Convert cache to the format needed for the file
    const progressData = {};
    for (const [jobId, data] of progressCache.entries()) {
      // Skip completed jobs
      if (completedJobs.has(jobId)) continue;
      
      progressData[jobId] = { ...data };
    }
    
    // Write to file
    await fs.writeFile(progressFilePath, JSON.stringify(progressData, null, 2));
    
    // Cleanup completed jobs from memory periodically
    cleanupCompletedJobs();
  } catch (error) {
    console.warn(`Warning: Could not update progress file: ${error.message}`);
    throw error; // Re-throw to handle in caller
  }
}

/**
 * Remove completed jobs from the cache after a certain time
 */
function cleanupCompletedJobs() {
  // Remove completed jobs from memory after they've been processed
  for (const jobId of completedJobs) {
    progressCache.delete(jobId);
  }
  
  // Clear completed jobs only if they've been persisted
  completedJobs.clear();
}

// Export functions
module.exports = {
  updateProgress,       // Sync version for backward compatibility
  updateProgressAsync,  // Async version for better performance
  logJobMessage,        // For general job logging
  logJobError           // For job error logging
}; 