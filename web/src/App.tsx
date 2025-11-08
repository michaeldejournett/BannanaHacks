import { useState } from 'react'
import FallingBananas from './FallingBananas'
import ImageUpload from './ImageUpload'
import './App.css'

const API_BASE_URL = 'http://localhost:8000/api';

function App() {
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [statusMessage, setStatusMessage] = useState<string>('');

  const handleImageSelect = async (file: File) => {
    setUploadStatus('uploading');
    setStatusMessage('Uploading image...');

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
        setStatusMessage('Error: Invalid file type. Please upload a PNG, JPEG, or JPG file.');
      } else if (data.message === 'successful upload!') {
        setUploadStatus('success');
        setStatusMessage('Successfully uploaded! The image is being analyzed...');
      } else {
        setUploadStatus('error');
        setStatusMessage('Unexpected response from server.');
      }
    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus('error');
      setStatusMessage('Failed to upload image. Please make sure the API server is running.');
    }
  };

  const handleImageRemove = () => {
    setUploadStatus('idle');
    setStatusMessage('');
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
            Upload a PNG, JPEG, or JPG image of a banana to check its ripeness.
          </p>
          {statusMessage && (
            <div className={`status-message ${uploadStatus}`}>
              {uploadStatus === 'uploading' && '⏳ '}
              {uploadStatus === 'success' && '✅ '}
              {uploadStatus === 'error' && '❌ '}
              {statusMessage}
            </div>
          )}
        </div>
      </div>
    </>
  )
}

export default App
