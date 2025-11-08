# How to Use the Banana Ripeness Prediction Script

## Quick Start

### Step 1: Activate Virtual Environment
```bash
cd /home/michaeldejournett/BannanaHacks
source venv/bin/activate
```

### Step 2: Run Prediction

**Option A: Simple usage (from project root)**
```bash
python predict_banana.py "path/to/your/banana/image.jpg"
```

**Option B: Using Model/predict.py directly**
```bash
cd Model
python predict.py "path/to/your/banana/image.jpg"
```

## Examples

### Example 1: Predict on a test image
```bash
# From project root
python predict_banana.py "data/raw/Banana Ripeness Classification Dataset/test/ripe/some_image.jpg"
```

### Example 2: Predict on any image file
```bash
# From project root
python predict_banana.py "/home/michaeldejournett/Pictures/my_banana.jpg"
```

### Example 3: Specify custom checkpoint
```bash
cd Model
python predict.py \
  --image "../data/raw/Banana Ripeness Classification Dataset/test/ripe/image.jpg" \
  --checkpoint checkpoints/best_model.pth
```

### Example 4: Use full path
```bash
python Model/predict.py --image "/full/path/to/image.jpg"
```

## Command Line Options

```bash
python predict.py [IMAGE_PATH] [OPTIONS]

Positional Arguments:
  image                 Path to image file

Options:
  --image IMAGE         Path to image file (alternative to positional)
  --checkpoint PATH     Path to model checkpoint (default: checkpoints/best_model.pth)
  --num-classes N       Number of classes (default: 4)
  -h, --help           Show help message
```

## What You Need

1. **Trained Model**: Make sure you have a trained model checkpoint at:
   - `Model/checkpoints/best_model.pth`
   
   If you haven't trained yet, run:
   ```bash
   cd Model
   python train.py --train-dir "../data/raw/Banana Ripeness Classification Dataset/train" \
                    --val-dir "../data/raw/Banana Ripeness Classification Dataset/valid" \
                    --num-classes 4 --epochs 50
   ```

2. **Image File**: Any banana image (JPG, PNG, etc.)

## Output Example

When you run the script, you'll see:

```
Using device: cuda:0 (AMD Radeon RX 6650M)
Loading model from: checkpoints/best_model.pth
✓ Model loaded successfully!

Analyzing image: path/to/image.jpg

============================================================
BANANA RIPENESS PREDICTION
============================================================

🍌 Predicted Ripeness: RIPE
📊 Confidence: 95.3%

All Probabilities:
------------------------------------------------------------
ripe        95.30% ████████████████████████████████████████
unripe       3.20% ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
overripe     1.10% ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
rotten       0.40% ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
============================================================
```

## Troubleshooting

### "Model checkpoint not found"
- Train a model first, or specify the correct checkpoint path with `--checkpoint`

### "Image file not found"
- Check the image path is correct
- Use absolute path if relative path doesn't work

### "No module named 'torch'"
- Make sure virtual environment is activated: `source venv/bin/activate`

## Quick Test

Try predicting on a test image:
```bash
cd /home/michaeldejournett/BannanaHacks
source venv/bin/activate

# Find a test image
TEST_IMAGE=$(find "data/raw/Banana Ripeness Classification Dataset/test" -name "*.jpg" | head -1)

# Predict
python predict_banana.py "$TEST_IMAGE"
```

That's it! The script will automatically use your GPU if available, or fall back to CPU.

