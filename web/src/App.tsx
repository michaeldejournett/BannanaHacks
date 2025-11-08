import { useState } from 'react'
import FallingBananas from './FallingBananas'
import ImageUpload from './ImageUpload'
import './App.css'

const API_BASE_URL = 'http://localhost:8000/api';

interface AnalysisResult {
  is_banana: boolean;
  confidence: number;
  class_name: string;
  probabilities: {
    banana: number;
    not_banana: number;
  };
}

function App() {
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);

  const handleImageSelect = async (file: File) => {
    setUploadStatus('uploading');
    setStatusMessage('Uploading and analyzing image...');
    setAnalysisResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE_URL}/banana-analysis`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.message === 'error') {
        setUploadStatus('error');
        setStatusMessage(data.error || 'Error processing image. Please try again.');
        setAnalysisResult(null);
      } else if (data.message === 'success') {
        setUploadStatus('success');
        setAnalysisResult({
          is_banana: data.is_banana,
          confidence: data.confidence,
          class_name: data.class_name,
          probabilities: data.probabilities
        });
        setStatusMessage('');
      } else {
        setUploadStatus('error');
        setStatusMessage('Unexpected response from server.');
        setAnalysisResult(null);
      }
    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus('error');
      setStatusMessage('Failed to upload image. Please make sure the API server is running.');
      setAnalysisResult(null);
    }
  };

  const handleImageRemove = () => {
    setUploadStatus('idle');
    setStatusMessage('');
    setAnalysisResult(null);
  };

  return (
    <>
      <FallingBananas />
      <div className="content">
        <h1>🍌 BannanaHacks 🍌</h1>
        <div className="card">
          <h2>Upload Banana Image</h2>
          <ImageUpload 
            onImageSelect={handleImageSelect}
            onImageRemove={handleImageRemove}
          />
          <p className="upload-description">
            Upload a PNG, JPEG, or JPG image to detect if it contains a banana.
          </p>
          {uploadStatus === 'uploading' && (
            <div className={`status-message ${uploadStatus}`}>
              ⏳ {statusMessage}
            </div>
          )}
          {uploadStatus === 'error' && statusMessage && (
            <div className={`status-message ${uploadStatus}`}>
              ❌ {statusMessage}
            </div>
          )}
          {analysisResult && (
            <div className="analysis-result">
              <div className={`result-header ${analysisResult.is_banana ? 'banana' : 'not-banana'}`}>
                {analysisResult.is_banana ? '🍌 BANANA DETECTED!' : '🚫 NO BANANA DETECTED'}
              </div>
              <div className="result-details">
                <div className="confidence-bar">
                  <div className="confidence-label">Confidence:</div>
                  <div className="confidence-value">
                    {(analysisResult.confidence * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="probabilities">
                  <div className="probability-item">
                    <span className="prob-label">🍌 Banana:</span>
                    <span className="prob-value">
                      {(analysisResult.probabilities.banana * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="probability-item">
                    <span className="prob-label">🚫 Not Banana:</span>
                    <span className="prob-value">
                      {(analysisResult.probabilities.not_banana * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}

export default App
