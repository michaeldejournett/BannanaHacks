# Banana Ripeness Classifier Web App

A web application for classifying banana ripeness using a trained machine learning model.

## Setup

1. Make sure you have the virtual environment activated:
```bash
source ../venv/bin/activate
```

2. Install webapp dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have a trained model checkpoint at:
```
../Model/checkpoints/best_model.pth
```

If your model is in a different location, update the `checkpoint_path` in `app.py`.

## Running the App

```bash
python app.py
```

The app will be available at `http://localhost:5000`

## API Endpoints

### POST `/api/predict`
Upload an image to get ripeness prediction.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `image` (file)

**Response:**
```json
{
  "predicted_class": "ripe",
  "confidence": 0.95,
  "probabilities": {
    "overripe": 0.02,
    "ripe": 0.95,
    "rotten": 0.01,
    "unripe": 0.02
  }
}
```

### GET `/api/health`
Check if the API is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

## Features

- Drag and drop image upload
- Real-time ripeness prediction
- Visual probability bars for all classes
- Responsive design
- Error handling

