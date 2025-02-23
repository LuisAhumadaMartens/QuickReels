import React, { useState } from 'react';
import { PlusCircle, Wand2 } from 'lucide-react';

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

  function updateTimeStamp(
    id: number,
    field: 'start' | 'end',
    value: string
  ) {
    setTimeStamps(
      timeStamps.map((pair) =>
        pair.id === id ? { ...pair, [field]: value } : pair
      )
    );
  }

  // Add function to validate timestamps
  const isTimestampsValid = () => {
    const isValid = timeStamps.every(pair => {
      if (!pair.start || !pair.end) return true;
      
      const [startMin, startSec] = pair.start.split(':').map(Number);
      const [endMin, endSec] = pair.end.split(':').map(Number);
      
      const startSeconds = startMin * 60 + startSec;
      const endSeconds = endMin * 60 + endSec;
      
      return endSeconds > startSeconds;
    });
    
    console.log('Timestamps valid:', isValid);
    return isValid;
  };

  async function handleCreateReels() {
    if (!savedFilePath) {
      appendToConsole('No saved file path available');
      return;
    }

    appendToConsole('Starting video processing...');
    
    try {
      const requestBody = {
        input_path: savedFilePath,
        output_type: mode === 'manual' ? 'multiple' : 'single',
        // Only include crops if in manual mode and there are valid timestamps
        ...(mode === 'manual' && {
          crops: timeStamps
            .filter(pair => pair.start && pair.end)
            .map(pair => ({
              start: pair.start,
              end: pair.end
            }))
        })
      };

      const response = await fetch('http://localhost:8000/run-script', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      const data = await response.json();
      
      if (response.ok) {
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
        <h1 className="editor-logo">QuickReels</h1>
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
      <h1 className="editor-logo">QuickReels</h1>
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
              <input
                type="text"
                placeholder="00:00"
                value={pair.start}
                onChange={(e) => updateTimeStamp(pair.id, 'start', e.target.value)}
                className="timestamp-input"
                maxLength={5}
              />
              <span className="timestamp-separator">to</span>
              <input
                type="text"
                placeholder={videoDuration}
                value={pair.end}
                onChange={(e) => updateTimeStamp(pair.id, 'end', e.target.value)}
                className="timestamp-input"
                maxLength={5}
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
    </div>
  );
}

export default App;