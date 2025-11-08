import { useState, useRef } from 'react'
import FallingBananas from './components/FallingBananas'
import './App.css'

function App() {
  const [image, setImage] = useState<string | null>(null)
  const [imageName, setImageName] = useState<string>('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      // Validate file type
      const validTypes = ['image/jpeg', 'image/jpg', 'image/png']
      if (!validTypes.includes(file.type)) {
        alert('Please select a JPEG, JPG, or PNG image file.')
        return
      }

      // Create object URL for preview
      const imageUrl = URL.createObjectURL(file)
      setImage(imageUrl)
      setImageName(file.name)
    }
  }

  const handleRemoveImage = () => {
    if (image) {
      URL.revokeObjectURL(image)
    }
    setImage(null)
    setImageName('')
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="app-container">
      <FallingBananas />
      <div className="content">
        <div className="header">
          <h1 className="title">
            <span className="banana-emoji">🍌</span>
            Banana Hacks
            <span className="banana-emoji">🍌</span>
          </h1>
          <p className="subtitle">Detect bananas and analyze ripeness level</p>
        </div>
        
        <div className="card">
          <div className="image-upload-container">
            {!image ? (
              <>
                <div className="upload-area">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".jpeg,.jpg,.png,image/jpeg,image/jpg,image/png"
                    onChange={handleFileChange}
                    className="file-input"
                    id="image-upload"
                  />
                  <label htmlFor="image-upload" className="upload-label">
                    <div className="upload-icon">📸</div>
                    <div className="upload-text">
                      <span className="upload-title">Upload Image</span>
                      <span className="upload-subtitle">JPEG, JPG, or PNG</span>
                    </div>
                  </label>
                </div>
                <p className="description">
                  Upload an image to detect if it's a banana and get its ripeness level
                </p>
              </>
            ) : (
              <>
                <div className="image-preview-container">
                  <img src={image} alt="Uploaded" className="image-preview" />
                  <button 
                    className="remove-button"
                    onClick={handleRemoveImage}
                    aria-label="Remove image"
                  >
                    ✕
                  </button>
                </div>
                <p className="image-name">{imageName}</p>
                <button 
                  className="banana-button change-image-button"
                  onClick={() => fileInputRef.current?.click()}
                >
                  🍌 Change Image
                </button>
              </>
            )}
          </div>
        </div>

        <div className="card project-goal-card">
          <h2 className="project-goal-title">Project Goal</h2>
          <p className="project-goal-text">
            The goal of this project was to create an API and Machine Learning model for{' '}
            <span className="hackathon-name">
              <span className="strikethrough">Cornhacks</span>
              <span className="slanted">BananaHacks</span>
            </span>{' '}
            hackathon. We wanted to create an open source software that could be used in industry to detect fruit ripeness levels. We used publicly available datasets to create the project.
          </p>
        </div>
      </div>
    </div>
  )
}

export default App
