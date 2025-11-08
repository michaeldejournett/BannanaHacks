"""
Flask API for banana ripeness classification.
"""

import os
import sys
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import torchvision.transforms as transforms

# Set environment variables for AMD GPU
os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')  # Use GPU 0 (discrete GPU)
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '10.3.0')  # Use gfx1030 kernels (compatible with gfx1100)

# Add parent directory to path to import model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Model'))
from src.models.model import BananaRipenessModel

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Class names (must match training order)
CLASS_NAMES = ['overripe', 'ripe', 'rotten', 'unripe']

# Global model variable
model = None
device = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model(checkpoint_path: str, num_classes: int = 4):
    """Load the trained model."""
    global model, device
    
    # Check for GPU availability (works with both NVIDIA CUDA and AMD ROCm)
    if torch.cuda.is_available():
        # Set to use GPU 0 explicitly (discrete GPU)
        os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        print(f"GPU detected! Using device: {device}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Selected GPU 0 (discrete GPU) for inference")
    else:
        device = torch.device('cpu')
        print(f"No GPU detected. Using CPU: {device}")
        print("Note: For AMD GPU support, install ROCm and PyTorch with ROCm support")
    
    model = BananaRipenessModel(num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")


def preprocess_image(image_path: str) -> torch.Tensor:
    """Preprocess an image for inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def predict_ripeness(image_path: str) -> dict:
    """Make a prediction on an uploaded image."""
    global model, device
    
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
        'predicted_class': CLASS_NAMES[predicted_class],
        'confidence': float(confidence),
        'probabilities': {
            CLASS_NAMES[i]: float(all_probs[i]) for i in range(len(CLASS_NAMES))
        }
    }
    
    return result


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for banana ripeness prediction."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server configuration.'}), 500
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_ripeness(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else None
    })


if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Load model (update path to your trained model checkpoint)
    checkpoint_path = os.path.join(
        os.path.dirname(__file__), '..', 'Model', 'checkpoints', 'best_model.pth'
    )
    
    if os.path.exists(checkpoint_path):
        load_model(checkpoint_path, num_classes=4)
    else:
        print(f"Warning: Model checkpoint not found at {checkpoint_path}")
        print("Please train a model first or update the checkpoint path in app.py")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)

