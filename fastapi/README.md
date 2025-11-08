# BannanaHacks FastAPI Backend

FastAPI backend for banana image analysis.

## Setup

1. Create a virtual environment (if not already created):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST `/api/banana-analysis`

Upload a banana image for analysis.

**Request:**
- Content-Type: `multipart/form-data`
- Body: File upload (PNG, JPEG, or JPG)

**Response:**
- Success: `{"message": "successful upload!"}`
- Error: `{"message": "error"}` (if file is not a valid image type)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/api/banana-analysis" \
  -F "file=@path/to/banana.jpg"
```

## API Documentation

Once the server is running, you can access:
- Interactive API docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

