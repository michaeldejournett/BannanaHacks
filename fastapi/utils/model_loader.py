"""
Model loading and inference utilities for banana detection.
"""

import os
from PIL import Image
from pathlib import Path
import sys

# Lazy imports - only import torch when needed
_torch = None
_torchvision = None
_BananaRipenessModel = None

# Model configuration
MODEL_PATH = Path(__file__).parent.parent / "models" / "bananaOrNot.pth"
CLASS_NAMES = ['banana', 'not_banana']
NUM_CLASSES = 2  # Default, will be detected from checkpoint if different

# Global model instance (loaded once on startup)
_model = None
_device = None
_transform = None
_detected_num_classes = None


def _import_torch():
    """Lazy import of torch and related modules."""
    global _torch, _torchvision, _BananaRipenessModel, _transform
    
    if _torch is None:
        try:
            import torch
            import torchvision.transforms as transforms
            
            # Import the model architecture
            project_root = Path(__file__).parent.parent.parent
            model_src_path = project_root / "Model" / "src"
            if str(model_src_path) not in sys.path:
                sys.path.insert(0, str(model_src_path))
            
            from models.model import BananaRipenessModel
            
            _torch = torch
            _torchvision = transforms
            _BananaRipenessModel = BananaRipenessModel
            
            # Create transform
            _transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
            
        except OSError as e:
            if "WinError 126" in str(e) or "DLL" in str(e):
                raise ImportError(
                    "PyTorch failed to load. This is likely due to missing Microsoft Visual C++ Redistributable.\n"
                    "Please download and install it from: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                    f"Original error: {e}"
                )
            raise
        except ImportError as e:
            raise ImportError(
                f"Failed to import PyTorch or model dependencies: {e}\n"
                "Please ensure PyTorch is installed: pip install torch torchvision"
            )
    
    return _torch, _torchvision, _BananaRipenessModel, _transform


def get_device():
    """Get the appropriate device (CPU or GPU)."""
    global _device
    if _device is None:
        torch, _, _, _ = _import_torch()
        if torch.cuda.is_available():
            try:
                # Test if CUDA works
                test_tensor = torch.rand(10, device='cuda:0')
                del test_tensor
                torch.cuda.empty_cache()
                _device = torch.device('cuda:0')
            except:
                _device = torch.device('cpu')
        else:
            _device = torch.device('cpu')
    return _device


def _detect_num_classes_from_checkpoint(checkpoint, torch):
    """
    Detect the number of classes from the checkpoint by examining fc2 layer.
    
    Args:
        checkpoint: The loaded checkpoint dictionary or state dict
        torch: The torch module
        
    Returns:
        int: Number of classes detected from the checkpoint
    """
    # Get the state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Check fc2 layer (final classification layer)
    if 'fc2.weight' in state_dict:
        num_classes = state_dict['fc2.weight'].shape[0]
    elif 'fc2.bias' in state_dict:
        num_classes = state_dict['fc2.bias'].shape[0]
    else:
        # Fallback to default
        num_classes = NUM_CLASSES
    
    return num_classes


def load_model():
    """
    Load the banana detection model.
    This should be called once at application startup.
    Automatically detects the number of classes from the checkpoint.
    """
    global _model, _device
    
    if _model is not None:
        return _model
    
    # Import torch and model architecture
    torch, _, BananaRipenessModel, _ = _import_torch()
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    _device = get_device()
    
    try:
        # Load checkpoint first to detect number of classes
        checkpoint = torch.load(MODEL_PATH, map_location=_device)
        
        # Detect number of classes from checkpoint
        detected_num_classes = _detect_num_classes_from_checkpoint(checkpoint, torch)
        
        if detected_num_classes != NUM_CLASSES:
            print(f"⚠ Warning: Checkpoint has {detected_num_classes} classes, but expected {NUM_CLASSES}.")
            print(f"   Using {detected_num_classes} classes from checkpoint.")
        
        # Initialize model with detected number of classes
        model = BananaRipenessModel(num_classes=detected_num_classes).to(_device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If checkpoint is just the state dict
            model.load_state_dict(checkpoint)
        
        model.eval()
        _model = model
        
        # Store detected num classes for prediction mapping
        global _detected_num_classes
        _detected_num_classes = detected_num_classes
        
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def preprocess_image(image: Image.Image):
    """
    Preprocess an image for model inference.
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed tensor ready for model input (torch.Tensor)
    """
    global _transform
    
    # Import torch and get transform
    _, _, _, transform = _import_torch()
    if _transform is None:
        _transform = transform
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    image_tensor = _transform(image).unsqueeze(0)
    return image_tensor


def predict_banana(image: Image.Image) -> dict:
    """
    Predict if an image contains a banana.
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with prediction results:
        {
            'is_banana': bool,
            'confidence': float,
            'class_name': str,
            'probabilities': dict
        }
    """
    global _model, _device, _detected_num_classes
    
    # Import torch
    torch, _, _, _ = _import_torch()
    
    # Ensure model is loaded
    if _model is None:
        _model = load_model()
    
    if _device is None:
        _device = get_device()
    
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
    
    # Map prediction to binary classification
    # If model has more than 2 classes, we need to map them
    if _detected_num_classes and _detected_num_classes > 2:
        # For models with 4 classes, assume:
        # - Class 0 might be "banana" or we need to check probabilities
        # - We'll map based on which class has highest probability
        # - For now, assume class 0 is banana, or we sum probabilities of banana classes
        # Since we don't know the exact mapping, we'll use class 0 as banana
        # and others as not_banana
        is_banana = predicted_class == 0
        banana_prob = float(all_probs[0])
        not_banana_prob = float(sum(all_probs[1:]))
    else:
        # Binary classification (2 classes)
        is_banana = predicted_class == 0  # Class 0 is 'banana'
        banana_prob = float(all_probs[0])
        not_banana_prob = float(all_probs[1])
    
    class_name = 'banana' if is_banana else 'not_banana'
    
    result = {
        'is_banana': is_banana,
        'confidence': float(confidence),
        'class_name': class_name,
        'probabilities': {
            'banana': banana_prob,
            'not_banana': not_banana_prob
        }
    }
    
    return result

