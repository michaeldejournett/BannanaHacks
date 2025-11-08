import { useState, useRef, useEffect } from 'react'
import FallingBananas from './components/FallingBananas'
import './App.css'

interface AnalysisResult {
  is_banana: boolean
  ripeness_level: string
  confidence: number
  predicted_class: number
  all_probabilities: Record<string, number>
  message: string
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [image, setImage] = useState<string | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imageName, setImageName] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(false)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
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
      setImageFile(file)
      setImageName(file.name)
      setAnalysisResult(null)
      setError(null)

      // Automatically analyze the image
      await analyzeImage(file)
    }
  }

  const analyzeImage = async (file: File) => {
    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_BASE_URL}/api/banana/analyze`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      const result: AnalysisResult = await response.json()
      setAnalysisResult(result)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to analyze image'
      setError(errorMessage)
      console.error('Error analyzing image:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleReanalyze = () => {
    if (imageFile) {
      analyzeImage(imageFile)
    }
  }

  const handleRemoveImage = () => {
    if (image) {
      URL.revokeObjectURL(image)
    }
    setImage(null)
    setImageFile(null)
    setImageName('')
    setAnalysisResult(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  useEffect(() => {
    // Cleanup object URL on unmount
    return () => {
      if (image) {
        URL.revokeObjectURL(image)
      }
    }
  }, [image])

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
                
                {loading && (
                  <div className="loading-container">
                    <div className="loading-spinner"></div>
                    <p className="loading-text">Analyzing banana...</p>
                  </div>
                )}

                {error && (
                  <div className="error-container">
                    <p className="error-text">❌ {error}</p>
                    <button 
                      className="banana-button retry-button"
                      onClick={handleReanalyze}
                    >
                      🔄 Retry Analysis
                    </button>
                  </div>
                )}

                {analysisResult && !loading && (
                  <div className="analysis-result">
                    <div className={`result-card ${analysisResult.is_banana ? 'success' : 'warning'}`}>
                      <h3 className="result-title">
                        {analysisResult.is_banana ? '🍌 Banana Detected!' : '⚠️ Not a Banana'}
                      </h3>
                      {analysisResult.is_banana && (
                        <>
                          <div className="ripeness-info">
                            <p className="ripeness-level">
                              Ripeness Level: <span className="ripeness-value">{analysisResult.ripeness_level}</span>
                            </p>
                            <p className="confidence">
                              Confidence: <span className="confidence-value">{(analysisResult.confidence * 100).toFixed(1)}%</span>
                            </p>
                          </div>
                          <div className="probabilities">
                            <p className="probabilities-title">All Ripeness Levels:</p>
                            <div className="probabilities-list">
                              {Object.entries(analysisResult.all_probabilities).map(([level, prob]) => (
                                <div key={level} className="probability-item">
                                  <span className="prob-level">{level}:</span>
                                  <div className="prob-bar-container">
                                    <div 
                                      className="prob-bar" 
                                      style={{ width: `${prob * 100}%` }}
                                    ></div>
                                  </div>
                                  <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </>
                      )}
                      {!analysisResult.is_banana && (
                        <p className="not-banana-message">{analysisResult.message}</p>
                      )}
                    </div>
                    <button 
                      className="banana-button retry-button"
                      onClick={handleReanalyze}
                    >
                      🔄 Reanalyze
                    </button>
                  </div>
                )}

                {!loading && !analysisResult && !error && (
                  <button 
                    className="banana-button analyze-button"
                    onClick={handleReanalyze}
                  >
                    🍌 Analyze Banana
                  </button>
                )}

                <button 
                  className="banana-button change-image-button"
                  onClick={() => fileInputRef.current?.click()}
                >
                  📁 Change Image
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
