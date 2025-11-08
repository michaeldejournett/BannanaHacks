"""
Inference script for banana ripeness prediction.
"""

import os
import argparse
from typing import List

import torch
from PIL import Image
import torchvision.transforms as transforms

# Set AMD GPU environment variables BEFORE importing model
os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '10.3.0')

from src.models.model import BananaRipenessModel

# Class names (must match training order)
CLASS_NAMES = ['overripe', 'ripe', 'rotten', 'unripe']


def load_model(checkpoint_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        num_classes (int): Number of output classes
        device (torch.device): Device to load model on
        
    Returns:
        torch.nn.Module: Loaded model
    """
    model = BananaRipenessModel(num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Preprocess an image for inference.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def predict(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    class_names: List[str] = None
) -> dict:
    """
    Make a prediction on a single image.
    
    Args:
        model: Trained model
        image_path (str): Path to image
        device: Device to run inference on
        class_names (list, optional): List of class names
        
    Returns:
        dict: Prediction results
    """
    # Preprocess image
    image_tensor = preprocess_image(image_path).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    predicted_class = predicted_class.item()
    confidence = confidence.item()
    
    # Get all class probabilities
    all_probs = probabilities.cpu().numpy()[0]
    
    result = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': all_probs.tolist()
    }
    
    if class_names:
        result['class_name'] = class_names[predicted_class]
        result['all_probabilities'] = {
            class_names[i]: float(all_probs[i]) for i in range(len(class_names))
        }
    
    return result


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description='Predict Banana Ripeness from an image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py image.jpg
  python predict.py --image banana.jpg --checkpoint checkpoints/best_model.pth
  python predict.py --image banana.jpg --checkpoint checkpoints/best_model.pth --num-classes 4
        """
    )
    parser.add_argument('image', type=str, nargs='?',
                       help='Path to image file')
    parser.add_argument('--image', type=str, dest='image_arg',
                       help='Path to image file (alternative to positional argument)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint (default: checkpoints/best_model.pth)')
    parser.add_argument('--num-classes', type=int, default=4,
                       help='Number of ripeness classes (default: 4)')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    args = parser.parse_args()
    
    # Get image path from either positional or named argument
    image_path = args.image or args.image_arg
    if not image_path:
        parser.error("Image path is required. Use: python predict.py <image_path>")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Model checkpoint not found: {args.checkpoint}")
        print(f"Please train a model first or specify the correct checkpoint path.")
        return
    
    # Set device - test GPU first, fallback to CPU if it fails
    device = torch.device('cpu')  # Default to CPU
    use_gpu = False
    
    if args.force_cpu:
        print("Using CPU (--force-cpu flag)")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        try:
            # Test GPU with a simple operation
            test_tensor = torch.rand(10, device='cuda:0')
            del test_tensor
            torch.cuda.empty_cache()
            
            # Test model forward pass on GPU
            print("Testing GPU inference...")
            test_model = BananaRipenessModel(num_classes=args.num_classes).to('cuda:0')
            test_input = torch.randn(1, 3, 224, 224, device='cuda:0')
            with torch.no_grad():
                _ = test_model(test_input)
            del test_model, test_input
            torch.cuda.empty_cache()
            
            # GPU test passed
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            use_gpu = True
            print(f"✓ GPU test passed! Using device: {device} ({torch.cuda.get_device_name(0)})")
        except Exception as e:
            print(f"⚠ GPU test failed: {e}")
            print("Falling back to CPU for inference")
            device = torch.device('cpu')
            use_gpu = False
    else:
        print(f"Using device: {device} (no GPU available)")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    try:
        model = load_model(args.checkpoint, args.num_classes, device)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Make prediction
    print(f"\nAnalyzing image: {image_path}")
    try:
        result = predict(model, image_path, device, CLASS_NAMES)
    except RuntimeError as e:
        if "CUDA" in str(e) or "HIP" in str(e) or "segmentation" in str(e).lower():
            print(f"\n⚠ GPU inference failed: {e}")
            print("Retrying with CPU...")
            # Reload model on CPU
            device = torch.device('cpu')
            model = load_model(args.checkpoint, args.num_classes, device)
            result = predict(model, image_path, device, CLASS_NAMES)
        else:
            raise
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print results
    print("\n" + "=" * 60)
    print("BANANA RIPENESS PREDICTION")
    print("=" * 60)
    print(f"\n🍌 Predicted Ripeness: {result['class_name'].upper()}")
    print(f"📊 Confidence: {result['confidence']:.1%}")
    print(f"\nAll Probabilities:")
    print("-" * 60)
    
    # Sort probabilities for display
    sorted_probs = sorted(
        result['all_probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for class_name, prob in sorted_probs:
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"{class_name:12s} {prob:6.2%} {bar}")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
