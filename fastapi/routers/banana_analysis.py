from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from PIL import Image
import io
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.model_loader import predict_banana, load_model

router = APIRouter()

# Allowed image MIME types
ALLOWED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg"
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {".png", ".jpeg", ".jpg"}


def is_valid_image_file(file: UploadFile) -> bool:
    """
    Validate that the uploaded file is a PNG, JPEG, or JPG image.
    
    Args:
        file: The uploaded file
        
    Returns:
        True if the file is a valid image type, False otherwise
    """
    # Check MIME type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        return False
    
    # Check file extension as additional validation
    filename = file.filename.lower() if file.filename else ""
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return False
    
    return True


@router.post("/banana-analysis")
async def analyze_banana(file: UploadFile = File(...)):
    """
    Upload a banana image for analysis.
    
    Accepts PNG, JPEG, or JPG image files.
    Uses the bananaOrNot.pth model to detect if the image contains a banana.
    
    Args:
        file: The image file to upload
        
    Returns:
        Dictionary with detection results:
        {
            "message": "success" or "error",
            "is_banana": bool (if successful),
            "confidence": float (if successful),
            "class_name": str (if successful)
        }
    """
    # Validate file type
    if not is_valid_image_file(file):
        return {
            "message": "error",
            "error": "Invalid file type. Please upload a PNG, JPEG, or JPG file."
        }
    
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Open image with PIL
        image = Image.open(io.BytesIO(contents))
        
        # Ensure model is loaded
        try:
            load_model()
        except Exception as e:
            return {
                "message": "error",
                "error": f"Model loading failed: {str(e)}"
            }
        
        # Make prediction
        try:
            result = predict_banana(image)
            
            return {
                "message": "success",
                "is_banana": result["is_banana"],
                "confidence": result["confidence"],
                "class_name": result["class_name"],
                "probabilities": result["probabilities"]
            }
        except Exception as e:
            return {
                "message": "error",
                "error": f"Prediction failed: {str(e)}"
            }
            
    except Exception as e:
        return {
            "message": "error",
            "error": f"Image processing failed: {str(e)}"
        }

