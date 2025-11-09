#!/usr/bin/env python3
"""
Model loader utility for FastAPI banana classification.
Provides functions to load the model and make predictions on PIL images.
"""

import os
import sys
from typing import Dict, Optional

import torch
from PIL import Image
import torchvision.transforms as transforms

# Set AMD GPU environment variables BEFORE importing model
os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '10.3.0')

# Add Model directory to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
model_dir = os.path.join(project_root, 'Model')
sys.path.insert(0, model_dir)

# Try importing the model
try:
    from src.models.model import BananaRipenessModel
except ImportError:
    # Fallback: try direct import
    try:
        sys.path.insert(0, os.path.join(model_dir, 'src', 'models'))
        from model import BananaRipenessModel
    except ImportError as e:
        raise ImportError(f"Could not import BananaRipenessModel. Model directory: {model_dir}, Error: {e}")

# Class names for binary classification
CLASS_NAMES = ['banana', 'not_banana']

# Global variables to store loaded model and device
_model = None
_device = None
_transform = None


def get_device() -> torch.device:
    """Get the best available device (GPU or CPU)."""
    device = torch.device('cpu')
    
    if torch.cuda.is_available():
        try:
            # Test GPU functionality
            test_tensor = torch.rand(10, device='cuda:0')
            del test_tensor
            torch.cuda.empty_cache()
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"GPU test failed: {e}")
            print("Falling back to CPU")
            device = torch.device('cpu')
    else:
        print("Using CPU")
    
    return device


def load_model() -> None:
    """Load the banana classification model into global variables."""
    global _model, _device, _transform
    
    if _model is not None:
        return  # Model already loaded
    
    # Set device
    _device = get_device()
    
    # Define preprocessing transform
    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Model checkpoint path
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'banana_or_not.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    _model = BananaRipenessModel(num_classes=2).to(_device)
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model.eval()
    print("✓ Model loaded successfully!")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess a PIL image for inference."""
    if _transform is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    image_tensor = _transform(image).unsqueeze(0)
    return image_tensor


def predict_banana(image: Image.Image) -> Dict[str, any]:
    """
    Make a prediction on a PIL image.
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with prediction results:
        {
            'is_banana': bool,
            'class_name': str,
            'confidence': float,
            'probabilities': dict
        }
    """
    if _model is None or _device is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # Preprocess image
    image_tensor = preprocess_image(image).to(_device)
    
    # Make prediction
    with torch.no_grad():
        outputs = _model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    predicted_class = predicted_class.item()
    confidence = confidence.item()
    all_probs = probabilities.cpu().numpy()[0]
    
    # Format results
    class_name = CLASS_NAMES[predicted_class]
    is_banana = class_name == 'banana'
    
    result = {
        'is_banana': is_banana,
        'class_name': class_name,
        'confidence': confidence,
        'probabilities': {
            CLASS_NAMES[i]: float(all_probs[i]) for i in range(len(CLASS_NAMES))
        }
    }
    
    return result

