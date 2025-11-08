import { useState, useRef } from 'react';
import './ImageUpload.css';

interface ImageUploadProps {
  onImageSelect?: (file: File) => void;
  onImageRemove?: () => void;
}

export default function ImageUpload({ onImageSelect, onImageRemove }: ImageUploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const acceptedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
  const acceptedExtensions = ['.png', '.jpeg', '.jpg'];

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    setError(null);

    if (!file) {
      return;
    }

    // Check file type
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    const isValidType = acceptedTypes.includes(file.type) || 
                       acceptedExtensions.includes(fileExtension);

    if (!isValidType) {
      setError('Please upload a PNG, JPEG, or JPG file only.');
      setSelectedFile(null);
      setPreview(null);
      return;
    }

    // Check file size (optional: limit to 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB.');
      setSelectedFile(null);
      setPreview(null);
      return;
    }

    setSelectedFile(file);
    onImageSelect?.(file);

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setError(null);

    const file = event.dataTransfer.files[0];
    if (!file) return;

    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    const isValidType = acceptedTypes.includes(file.type) || 
                       acceptedExtensions.includes(fileExtension);

    if (!isValidType) {
      setError('Please upload a PNG, JPEG, or JPG file only.');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB.');
      return;
    }

    setSelectedFile(file);
    onImageSelect?.(file);

    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleRemove = () => {
    setSelectedFile(null);
    setPreview(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    onImageRemove?.();
  };

  return (
    <div className="image-upload-container">
      <div
        className={`upload-area ${preview ? 'has-image' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".png,.jpeg,.jpg,image/png,image/jpeg,image/jpg"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        {preview ? (
          <div className="preview-container">
            <img src={preview} alt="Preview" className="preview-image" />
            <div className="preview-overlay">
              <p className="file-name">{selectedFile?.name}</p>
              <button
                type="button"
                className="remove-button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemove();
                }}
              >
                Remove
              </button>
            </div>
          </div>
        ) : (
          <div className="upload-placeholder">
            <div className="upload-icon">🍌</div>
            <p className="upload-text">
              Click or drag and drop a banana image here
            </p>
            <p className="upload-hint">
              PNG, JPEG, or JPG files only
            </p>
          </div>
        )}
      </div>
      {error && <p className="error-message">{error}</p>}
    </div>
  );
}

