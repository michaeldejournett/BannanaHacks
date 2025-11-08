"""
Inference script for banana ripeness prediction.
"""

import argparse
from typing import List

import torch
from PIL import Image
import torchvision.transforms as transforms

from src.models.model import BananaRipenessModel


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
    
    result = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities.cpu().numpy()[0].tolist()
    }
    
    if class_names:
        result['class_name'] = class_names[predicted_class]
    
    return result


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Predict Banana Ripeness')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num-classes', type=int, default=5,
                       help='Number of ripeness classes')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, args.num_classes, device)
    
    # Make prediction
    print(f"Predicting ripeness for: {args.image}")
    result = predict(model, args.image, device)
    
    # Print results
    print(f"\nPrediction Results:")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nAll Class Probabilities:")
    for i, prob in enumerate(result['probabilities']):
        print(f"  Class {i}: {prob:.2%}")


if __name__ == '__main__':
    main()
