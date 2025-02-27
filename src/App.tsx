import React, { useState } from 'react';
import { PlusCircle, Wand2 } from 'lucide-react';
import InputMask from 'react-input-mask';

interface TimeStampPair {
  id: number;
  start: string;
  end: string;
}

function formatDuration(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
}

function App() {
  const [currentPage, setCurrentPage] = useState<'home' | 'editor' | 'preview'>('home');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [mode, setMode] = useState<'manual' | 'ai'>('manual');
  const [timeStamps, setTimeStamps] = useState<TimeStampPair[]>([
    { id: 1, start: '', end: '' },
  ]);
  const [isUploading, setIsUploading] = useState(false);
  const [savedFilePath, setSavedFilePath] = useState<string | null>(null);
  const [consoleOutput, setConsoleOutput] = useState<string[]>([]);
  const [outputPaths, setOutputPaths] = useState<string[]>([]);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoDuration, setVideoDuration] = useState<string>("00:00");
  const [processingProgress, setProcessingProgress] = useState<number>(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string>("");

  function showUrlInput() {
    const urlInput = document.querySelector('.url-input');
    urlInput?.classList.add('show');
  }

  const appendToConsole = (text: string) => {
    setConsoleOutput(prev => [...prev, text]);
  };

  async function handleFileUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setVideoUrl(url);

    // Get video duration
    const video = document.createElement('video');
    video.src = url;
    video.onloadedmetadata = () => {
      setVideoDuration(formatDuration(video.duration));
    };
    
    appendToConsole(`File selected: ${file.name}`);
    appendToConsole(`Size: ${(file.size / (1024 * 1024)).toFixed(2)} MB`);
    appendToConsole(`Format: ${file.name.split('.').pop()?.toLowerCase()}`);

    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await fetch('http://localhost:8000/save-file', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      
      if (response.ok) {
        setSavedFilePath(data.filepath);
        appendToConsole(`File saved successfully at: ${data.filepath}`);
        setCurrentPage('editor');
      } else {
        throw new Error(data.error || 'Failed to save file');
      }
    } catch (error) {
      appendToConsole(`Error saving file: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsUploading(false);
    }
  }

  function handleYoutubeSubmit() {
    if (youtubeUrl) {
      setCurrentPage('editor');
    }
  }

  function addTimeStampPair() {
    if (timeStamps.length < 5) {
      setTimeStamps([
        ...timeStamps,
        { id: timeStamps.length + 1, start: '', end: '' },
      ]);
    }
  }

  const handleTimeChange = (
    e: React.ChangeEvent<HTMLInputElement>,
    id: number,
    field: 'start' | 'end'
  ) => {
    const value = e.target.value;
    
    setTimeStamps(
      timeStamps.map((pair) =>
        pair.id === id ? { ...pair, [field]: value } : pair
      )
    );
  };

  // Update isTimestampsValid function
  const isTimestampsValid = () => {
    // Convert video duration (MM:SS) to seconds
    const [durationMin, durationSec] = videoDuration.split(':').map(Number);
    const maxDurationInSeconds = durationMin * 60 + durationSec;

    const isValid = timeStamps.every(pair => {
      // If both timestamps are empty, consider it valid
      if (!pair.start && !pair.end) return true;
      
      // Convert timestamps to seconds for comparison
      let startSeconds = 0;
      let endSeconds = maxDurationInSeconds;

      if (pair.start) {
        const [startMin, startSec] = pair.start.split(':').map(Number);
        startSeconds = startMin * 60 + startSec;
        // Check if start is valid (between 0 and video duration)
        if (startSeconds < 0 || startSeconds > maxDurationInSeconds) {
          return false;
        }
      }

      if (pair.end) {
        const [endMin, endSec] = pair.end.split(':').map(Number);
        endSeconds = endMin * 60 + endSec;
        // Check if end is valid (between 0 and video duration)
        if (endSeconds < 0 || endSeconds > maxDurationInSeconds) {
          return false;
        }
      }

      // Check if start is before end
      return startSeconds < endSeconds;
    });

    return isValid;
  };

  async function handleCreateReels() {
    if (!savedFilePath) {
      appendToConsole('No saved file path available');
      return;
    }

    setIsProcessing(true);
    setProcessingProgress(0);
    setStatusMessage("Initializing...");
    appendToConsole('Starting video processing...');
    
    // Start polling for progress immediately
    const progressInterval = setInterval(async () => {
      try {
        const progressResponse = await fetch('http://localhost:8000/get-progress');
        const progressData = await progressResponse.json();
        
        if (progressData.progress) {
          setProcessingProgress(progressData.progress);
          setStatusMessage(progressData.status);
        }
      } catch (error) {
        console.error('Error fetching progress:', error);
      }
    }, 1000);
    
    try {
      const requestBody = {
        input_path: savedFilePath,
        output_type: mode === 'manual' ? 'multiple' : 'single',
        ...(mode === 'manual' && {
          crops: timeStamps
            .map(pair => ({
              start: pair.start || '00:00',
              end: pair.end || videoDuration
            }))
        })
      };

      // Start the processing
      const processResponse = await fetch('http://localhost:8000/run-script', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      const data = await processResponse.json();
      
      if (processResponse.ok) {
        if (mode === 'manual') {
          appendToConsole('Multiple reels created successfully!');
          setOutputPaths(data.outputs || []);
          data.outputs?.forEach((output: string, index: number) => {
            appendToConsole(`Reel ${index + 1} saved as: ${output}`);
          });
        } else {
          appendToConsole('Single reel created successfully!');
          setOutputPaths([data.output]);
          appendToConsole(`Output saved as: ${data.output}`);
        }
        setCurrentPage('preview');
      } else {
        throw new Error(data.error || 'Processing failed');
      }
    } catch (error) {
      appendToConsole(`Error processing video: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      // Clear the interval and reset states
      clearInterval(progressInterval);
      setIsProcessing(false);
      setProcessingProgress(0);
    }
  }

  if (currentPage === 'home') {
    return (
      <div className="container">
        <h1>QuickReels</h1>
        <p className="subtitle">
          Create shortform content from your videos with one click!
        </p>
        <div className="buttons">
          <input
            type="file"
            id="fileInput"
            accept="video/*"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
          <button 
            className="button" 
            onClick={() => document.getElementById('fileInput')?.click()}
            disabled={isUploading}
          >
            {isUploading ? 'Uploading...' : 'Upload a video'}
          </button>
          <button 
            className="button" 
            onClick={showUrlInput}
            disabled={true}
          >
            Enter YouTube URL
          </button>
        </div>
      </div>
    );
  }

  if (currentPage === 'preview') {
    return (
      <div className="container preview-container">
        <h1 className="editor-logo" onClick={() => window.location.reload()} style={{ cursor: 'pointer' }}>QuickReels</h1>
        <h2 className="preview-title">Your reels are here!</h2>
        <div className="reels-grid">
          {outputPaths.map((outputPath, index) => (
            <div key={index} className="reel-container">
              <div className="reel-number">{index + 1}</div>
              <div className="reel-video-container">
                <video
                  src={`http://localhost:8000/uploads/output_${index + 1}.mp4`}
                  controls
                  className="reel-video"
                  playsInline
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="container editor-container">
      <h1 className="editor-logo" onClick={() => window.location.reload()} style={{ cursor: 'pointer' }}>QuickReels</h1>
      <div className="video-preview">
        {selectedFile && videoUrl ? (
          <video 
            src={videoUrl}
            controls 
            className="video-player"
          />
        ) : (
          <div className="youtube-preview">
            <p>YouTube URL: {youtubeUrl}</p>
          </div>
        )}
      </div>

      {!isProcessing ? (
        <>
          <div className="mode-toggle">
            <button
              className={`toggle-btn ${mode === 'manual' ? 'active' : ''}`}
              onClick={() => setMode('manual')}
            >
              Manual
            </button>
            <button
              className={`toggle-btn ${mode === 'ai' ? 'active' : ''}`}
              onClick={() => setMode('ai')}
              disabled={true}
            >
              Auto-Highlight
            </button>
          </div>

          {mode === 'manual' ? (
            <div className="timestamp-container">
              {timeStamps.map((pair) => (
                <div key={pair.id} className="timestamp-pair">
                  <InputMask
                    mask="99:99"
                    maskChar="0"
                    value={pair.start}
                    onChange={(e) => handleTimeChange(e, pair.id, 'start')}
                    className="timestamp-input"
                    placeholder="00:00"
                  />
                  <span className="timestamp-separator">to</span>
                  <InputMask
                    mask="99:99"
                    maskChar="0"
                    value={pair.end}
                    onChange={(e) => handleTimeChange(e, pair.id, 'end')}
                    className="timestamp-input"
                    placeholder={videoDuration}
                  />
                </div>
              ))}
              {timeStamps.length < 5 && (
                <button className="add-timestamp-btn" onClick={addTimeStampPair}>
                  <PlusCircle size={20} />
                  Add Timestamp
                </button>
              )}
            </div>
          ) : (
            <div className="ai-message">
              <Wand2 size={24} className="wand-icon" />
              <p>We'll take it from here then. Hit create!</p>
            </div>
          )}

          <button 
            className={`button generate-btn create-reels-btn ${mode === 'manual' && !isTimestampsValid() ? 'disabled' : ''}`}
            onClick={handleCreateReels}
            disabled={mode === 'manual' && !isTimestampsValid()}
          >
            Generate Reels
          </button>
        </>
      ) : (
        <div className="processing-container">
          <p className="status-message">{statusMessage}</p>
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ width: `${processingProgress}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;