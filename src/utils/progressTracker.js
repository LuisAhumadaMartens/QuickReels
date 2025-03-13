/**
 * Progress Tracking Service
 * This module handles progress tracking for video analysis and processing tasks
 * with console logging only (no file I/O).
 * 
 * DEPRECATED: File-based progress tracking has been removed. Progress is now only logged to the console.
 */

// Keep only the necessary imports
const config = require('../config/config');

// In-memory progress cache (maintained for backward compatibility)
const progressCache = new Map();
// Set of completed jobs
const completedJobs = new Set();

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
 * Enhanced logging for progress updates
 * @param {string} jobId - The job ID 
 * @param {string} phase - The processing phase
 * @param {number} progress - Progress percentage
 * @param {string} status - Status message
 */
function logProgressUpdate(jobId, phase, progress, status) {
  if (status) {
    const isError = status.toLowerCase().includes('error');
    logJobMessage(jobId, status, isError);
  }
  
  if (progress !== undefined) {
    // Only log meaningful progress changes (not minor increments)
    if (progress === 100) {
      logJobMessage(jobId, `${phase.toUpperCase()} phase completed (100%)`);
    } else if (progress % 25 === 0) {
      logJobMessage(jobId, `${phase.toUpperCase()} progress: ${progress}%`);
    }
  }
}

/**
 * Update progress in memory (async implementation)
 * @param {string} jobId - The job ID
 * @param {Object} statusUpdate - The status update object
 */
async function updateProgressAsync(jobId, statusUpdate) {
  // Don't process updates for completed jobs
  if (completedJobs.has(jobId)) {
    return;
  }
  
  // Update in-memory cache first (for backwards compatibility)
  if (!progressCache.has(jobId)) {
    progressCache.set(jobId, {
      status: "Initializing...",
      analysis: 0,
      processing: 0,
      encoding: 0,
      audio: 0
    });
    
    logJobMessage(jobId, "Job initialized");
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
        const previousProgress = jobData[phase];
        jobData[phase] = newProgress;
        wasUpdated = true;
        
        // Log meaningful progress changes
        if (previousProgress !== newProgress) {
          logProgressUpdate(jobId, phase, newProgress, phaseUpdate.status);
        }
      }
    } else if (phaseUpdate.status) {
      // Status update only
      logProgressUpdate(jobId, phase, undefined, phaseUpdate.status);
      jobData.status = phaseUpdate.status;
      wasUpdated = true;
    }
    
    return wasUpdated;
  };
  
  // Update each phase
  if (statusUpdate.analysis) updatePhase('analysis', statusUpdate.analysis);
  if (statusUpdate.processing) updatePhase('processing', statusUpdate.processing);
  if (statusUpdate.encoding) updatePhase('encoding', statusUpdate.encoding);
  if (statusUpdate.audio) updatePhase('audio', statusUpdate.audio);
  
  // Check if videoGenerated flag is set
  if (statusUpdate.videoGenerated !== undefined) {
    jobData.videoGenerated = statusUpdate.videoGenerated;
    if (statusUpdate.videoGenerated) {
      logJobMessage(jobId, "Video file has been generated");
    }
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
    
    // Remove from in-memory cache after a short delay (for backward compatibility)
    setTimeout(() => {
      if (completedJobs.has(jobId)) {
        progressCache.delete(jobId);
        completedJobs.delete(jobId);
      }
    }, 5000);
  }
}

/**
 * Synchronous version of updateProgress for backward compatibility
 * @param {string} jobId - The job ID
 * @param {Object} statusUpdate - The status update object
 * @deprecated Use updateProgressAsync instead. File-based progress tracking has been removed.
 */
function updateProgress(jobId, statusUpdate) {
  try {
    // If the job has been marked as completed, ignore further updates
    if (completedJobs.has(jobId)) {
      return;
    }
    
    // Initialize job entry if it doesn't exist in memory
    if (!progressCache.has(jobId)) {
      progressCache.set(jobId, {
        status: "Initializing...",
        analysis: 0,
        processing: 0,
        encoding: 0,
        audio: 0
      });
      
      logJobMessage(jobId, "Job initialized");
    }
    
    const progressData = progressCache.get(jobId);
    
    // Generalized function to update a specific phase
    const updatePhase = (phase, phaseUpdate) => {
      if (!phaseUpdate) return false;
      
      let wasUpdated = false;
      
      // Update progress if provided and higher than current (or error state)
      if (phaseUpdate.progress !== undefined) {
        const newProgress = phaseUpdate.progress;
        const previousProgress = progressData[phase];
        
        if (newProgress === -1 || newProgress >= progressData[phase]) {
          progressData[phase] = newProgress;
          wasUpdated = true;
          
          // Log meaningful progress changes
          if (previousProgress !== newProgress) {
            logProgressUpdate(jobId, phase, newProgress, phaseUpdate.status);
          }
        }
      } else if (phaseUpdate.status) {
        // Status update only
        progressData.status = phaseUpdate.status;
        logProgressUpdate(jobId, phase, undefined, phaseUpdate.status);
        wasUpdated = true;
      }
      
      return wasUpdated;
    };
    
    // Update each phase if needed
    updatePhase('analysis', statusUpdate.analysis);
    updatePhase('processing', statusUpdate.processing);
    updatePhase('encoding', statusUpdate.encoding);
    updatePhase('audio', statusUpdate.audio);
    
    // Check if videoGenerated flag is set
    if (statusUpdate.videoGenerated !== undefined) {
      progressData.videoGenerated = statusUpdate.videoGenerated;
      if (statusUpdate.videoGenerated) {
        logJobMessage(jobId, "Video file has been generated");
      }
    }
    
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
        progressData.status = "Video generation complete";
        logJobMessage(jobId, "Video generation complete");
        
        // Ensure all phases are set to 100% for completion
        if (!hasError) {
          progressData.analysis = 100;
          progressData.processing = 100;
          progressData.encoding = 100;
          progressData.audio = 100;
        }
      }
      
      // Mark job as completed
      completedJobs.add(jobId);
      
      // Log completion
      logJobMessage(jobId, `Job ${hasError ? 'completed with error' : 'completed'}, removed from tracking`);
      
      // Remove from in-memory cache after a short delay (for backward compatibility)
      setTimeout(() => {
        if (completedJobs.has(jobId)) {
          progressCache.delete(jobId);
          completedJobs.delete(jobId);
        }
      }, 5000);
    }
  } catch (err) {
    logJobError(jobId, `Warning: Could not update progress: ${err.message}`, err);
  }
  
  // Also update async tracking for consistency
  updateProgressAsync(jobId, statusUpdate).catch(err => {
    logJobError(jobId, `Warning: Async progress update failed: ${err.message}`, err);
  });
}

/**
 * Get current progress for a job from memory (for backward compatibility)
 * @param {string} jobId - The job ID
 * @returns {Object|null} The job progress data or null if not found
 */
function getJobProgress(jobId) {
  return progressCache.has(jobId) ? { ...progressCache.get(jobId) } : null;
}

// Export functions
module.exports = {
  updateProgress,       // Sync version for backward compatibility
  updateProgressAsync,  // Async version for better performance
  logJobMessage,        // For general job logging
  logJobError,          // For job error logging
  getJobProgress        // For backward compatibility with code that reads progress
}; 