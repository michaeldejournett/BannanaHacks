"""
Banana Analysis Controller for FastAPI.
Handles image uploads and banana ripeness prediction.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io

# Lazy import for PIL
Image = None

# Lazy imports for torch - only import when needed
torch = None
transforms = None
BananaRipenessModel = None

def _import_pil():
    """Lazy import PIL (Pillow)."""
    global Image
    if Image is None:
        try:
            from PIL import Image
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Pillow (PIL) not available. Please install pillow."
            )
    return Image

def _import_torch():
    """Lazy import torch and related modules."""
    global torch, transforms, BananaRipenessModel
    if torch is None:
        try:
            import torch
            import torchvision.transforms as transforms
            # Add parent directory to path to import model
            sys.path.append(str(Path(__file__).parent.parent.parent))
            from src.models.model import BananaRipenessModel
        except ImportError as e:
            raise HTTPException(
                status_code=500,
                detail=f"PyTorch not available: {str(e)}. Please install torch and torchvision."
            )
    return torch, transforms, BananaRipenessModel

router = APIRouter(prefix="/api/banana", tags=["banana"])

# Model configuration
MODEL_CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT_PATH", "checkpoints/best_model.pth")
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "5"))

# Class names for ripeness levels (adjust based on your model)
RIPENESS_CLASSES = [
    "Unripe",
    "Slightly Ripe",
    "Ripe",
    "Very Ripe",
    "Overripe"
]

# Global model variable (loaded on startup)
_model: Optional[BananaRipenessModel] = None


def load_model():
    """Load the banana ripeness model."""
    global _model
    if _model is None:
        torch, _, BananaRipenessModel = _import_torch()
        
        if not os.path.exists(MODEL_CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"Model checkpoint not found at {MODEL_CHECKPOINT_PATH}. "
                "Please train the model first or set MODEL_CHECKPOINT_PATH environment variable."
            )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _model = BananaRipenessModel(num_classes=NUM_CLASSES).to(device)
        checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
        _model.load_state_dict(checkpoint['model_state_dict'])
        _model.eval()
        print(f"Model loaded successfully from {MODEL_CHECKPOINT_PATH}")
    
    return _model


def preprocess_image(image_bytes: bytes):
    """
    Preprocess an image for inference.
    
    Args:
        image_bytes: Image file bytes
        
    Returns:
        Preprocessed image tensor
    """
    Image = _import_pil()
    _, transforms, _ = _import_torch()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")


@router.post("/analyze")
async def analyze_banana(file: UploadFile = File(...)):
    """
    Analyze an uploaded image to detect if it's a banana and determine ripeness level.
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with analysis results
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPEG, JPG, or PNG image."
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Import torch
        torch, _, _ = _import_torch()
        
        # Preprocess image
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_tensor = preprocess_image(image_bytes).to(device)
        
        # Load model (lazy loading)
        model = load_model()
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        
        # Get class name
        class_name = RIPENESS_CLASSES[predicted_class] if predicted_class < len(RIPENESS_CLASSES) else f"Class {predicted_class}"
        
        # Get all probabilities
        all_probabilities = probabilities.cpu().numpy()[0].tolist()
        probability_dict = {
            RIPENESS_CLASSES[i] if i < len(RIPENESS_CLASSES) else f"Class {i}": prob
            for i, prob in enumerate(all_probabilities)
        }
        
        # Determine if it's a banana (you might want to add a separate classification step)
        # For now, we'll assume if confidence is high enough, it's a banana
        is_banana = confidence > 0.5
        
        result = {
            "is_banana": is_banana,
            "ripeness_level": class_name,
            "confidence": round(confidence, 4),
            "predicted_class": predicted_class,
            "all_probabilities": probability_dict,
            "message": f"Detected: {class_name} (Confidence: {confidence:.2%})" if is_banana else "Image does not appear to be a banana."
        }
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        torch, _, _ = _import_torch()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "device": str(device)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
