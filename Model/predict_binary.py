"""
Prediction script for binary banana classification (banana vs not banana).
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

# Class names for binary classification
CLASS_NAMES = ['banana', 'not_banana']


def load_model(checkpoint_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """Load a trained model from checkpoint."""
    model = BananaRipenessModel(num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def preprocess_image(image_path: str) -> torch.Tensor:
    """Preprocess an image for inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def predict(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    class_names: List[str] = None
) -> dict:
    """Make a prediction on a single image."""
    image_tensor = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    predicted_class = predicted_class.item()
    confidence = confidence.item()
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
        description='Predict if image is a banana or not',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('image', type=str, nargs='?',
                       help='Path to image file')
    parser.add_argument('--image', type=str, dest='image_arg',
                       help='Path to image file')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints_binary/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU usage')
    args = parser.parse_args()
    
    image_path = args.image or args.image_arg
    if not image_path:
        parser.error("Image path is required")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Model checkpoint not found: {args.checkpoint}")
        return
    
    # Set device
    device = torch.device('cpu')
    if args.force_cpu:
        print("Using CPU (--force-cpu flag)")
    elif torch.cuda.is_available():
        try:
            test_tensor = torch.rand(10, device='cuda:0')
            del test_tensor
            torch.cuda.empty_cache()
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        except:
            print("GPU test failed, using CPU")
            device = torch.device('cpu')
    else:
        print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    try:
        model = load_model(args.checkpoint, 2, device)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Make prediction
    print(f"\nAnalyzing image: {image_path}")
    try:
        result = predict(model, image_path, device, CLASS_NAMES)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    # Print results
    print("\n" + "=" * 60)
    print("BANANA CLASSIFICATION RESULT")
    print("=" * 60)
    print(f"\n{'🍌 BANANA' if result['class_name'] == 'banana' else '🚫 NOT BANANA'}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"\nProbabilities:")
    print("-" * 60)
    
    sorted_probs = sorted(
        result['all_probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for class_name, prob in sorted_probs:
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        emoji = "🍌" if class_name == "banana" else "🚫"
        print(f"{emoji} {class_name:12s} {prob:6.2%} {bar}")
    
    print("=" * 60)


if __name__ == '__main__':
    main()

