import { useState } from 'react'
import FallingBananas from './FallingBananas'
import ImageUpload from './ImageUpload'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  const handleImageSelect = (file: File) => {
    // Image will be processed by the separate scanning part of the project
    console.log('Selected image:', file.name);
    // You can add additional handling here if needed
  }

  return (
    <>
      <FallingBananas />
      <div className="content">
        <h1>🍌 BannanaHacks 🍌</h1>
        <div className="card">
          <h2>Upload Banana Image</h2>
          <ImageUpload onImageSelect={handleImageSelect} />
          <p className="upload-description">
            Upload a PNG, JPEG, or JPG image of a banana to check its ripeness.
          </p>
        </div>
      </div>
    </>
  )
}

export default App
