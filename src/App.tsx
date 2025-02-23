import React, { useState } from 'react';
import { PlusCircle, Wand2 } from 'lucide-react';

interface TimeStampPair {
  id: number;
  start: string;
  end: string;
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
          data.outputs?.forEach((output: string, index: number) => {
            appendToConsole(`Reel ${index + 1} saved as: ${output}`);
          });
        } else {
          appendToConsole('Single reel created successfully!');
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
          <button 
            className="button generate-btn"
            onClick={handleYoutubeSubmit}
            disabled={!selectedFile}
          >
            Generate
          </button>
        </div>
        {consoleOutput.length > 0 && (
          <div className="console">
            {consoleOutput.map((line, index) => (
              <div key={index}>{line}</div>
            ))}
          </div>
        )}
      </div>
    );
  }

  if (currentPage === 'preview') {
    const numReels = mode === 'manual' 
      ? timeStamps.filter(pair => pair.start && pair.end).length 
      : 1;
    const reels = Array.from({ length: numReels }, (_, i) => i + 1);

    return (
      <div className="container preview-container">
        <h1 className="editor-logo">QuickReels</h1>
        <h2 className="preview-title">Your reels are here!</h2>
        <div className="reels-grid">
          {reels.map((reelNum) => (
            <div key={reelNum} className="reel-container">
              <div className="reel-number">{reelNum}</div>
              <div className="reel-video-container">
                {selectedFile ? (
                  <video
                    src={URL.createObjectURL(selectedFile)}
                    controls
                    className="reel-video"
                    playsInline
                  />
                ) : (
                  <div className="reel-placeholder">
                    <p>Reel {reelNum}</p>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
        {consoleOutput.length > 0 && (
          <div className="console">
            {consoleOutput.map((line, index) => (
              <div key={index}>{line}</div>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="container editor-container">
      <h1 className="editor-logo">QuickReels</h1>
      <div className="video-preview">
        {selectedFile ? (
          <video 
            src={URL.createObjectURL(selectedFile)} 
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
        >
          AI
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
              />
              <span className="timestamp-separator">to</span>
              <input
                type="text"
                placeholder="00:00"
                value={pair.end}
                onChange={(e) => updateTimeStamp(pair.id, 'end', e.target.value)}
                className="timestamp-input"
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
        className="button generate-btn create-reels-btn"
        onClick={handleCreateReels}
      >
        Create my reels!
      </button>
    </div>
  );
}

export default App;