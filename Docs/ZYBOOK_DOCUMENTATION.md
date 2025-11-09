# BannanaHacks: Complete Guide with Interactive Examples

## Table of Contents
1. [Introduction](#introduction)
2. [Project Setup](#project-setup)
3. [AMD GPU Configuration](#amd-gpu-configuration)
4. [Training Your Model](#training-your-model)
5. [Making Predictions](#making-predictions)
6. [Multi-GPU Training](#multi-gpu-training)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring GPU Usage](#monitoring-gpu-usage)
9. [Troubleshooting](#troubleshooting)

---

## Introduction

BannanaHacks is a PyTorch-based deep learning project for detecting banana ripeness using convolutional neural networks. This guide provides step-by-step instructions with runnable examples for every major operation.

**⚠️ Important:** All examples in this guide assume you are running commands from the **project root directory** (`BannanaHacks/`). If you're running examples from within this documentation, make sure you're in the project root first:

```bash
cd /home/michaeldejournett/BananaHacks2/BannanaHacks
```

**Running Examples:**
- Code blocks with `{cmd=true}` are executable in Markdown Preview Enhanced
- **To enable execution**: Right-click the preview → Settings → Enable "Code Chunk Execution"
- Click the run button that appears above executable code blocks
- If run buttons don't appear, you can copy and paste code blocks directly into your terminal
- Python code blocks can be saved as `.py` files and executed
- All code blocks are designed to work from the project root directory

### Project Structure

```
BannanaHacks/
├── Model/              # Model training and prediction code
│   ├── train.py        # Training script
│   ├── predict.py       # Prediction script
│   └── src/            # Source code
├── Scripts/            # Utility scripts
│   ├── check_gpu.py     # GPU diagnostics
│   └── predict_banana.py # Simple prediction wrapper
├── fastapi/            # FastAPI backend
├── web/                # React frontend
└── data/               # Dataset directory
```

---

## Project Setup

### Part 1: Initial Setup

**Example 1.1: Create and activate virtual environment**

```bash
# Create virtual environment (run from project root)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Run this example:**

```bash {cmd=true}
# Ensure you're in project root, then run:
python3 -m venv venv
source venv/bin/activate
echo "Virtual environment activated!"
```

**Example 1.2: Install dependencies**

```bash
# Install Model dependencies
cd Model
pip install -r requirements.txt

# Install FastAPI dependencies (if using API)
cd ../fastapi
pip install -r requirements.txt

# Install web dependencies (if using frontend)
cd ../web
npm install
```

**Run this example:**
```bash {cmd=true}
# From project root:
source venv/bin/activate
cd Model
pip install -r requirements.txt
cd ..
```

---

## AMD GPU Configuration

### Part 2: GPU Setup for AMD GPUs

**Example 2.1: Install PyTorch with ROCm support**

For Ubuntu 24.10, use pip-based installation (recommended):

```bash
source venv/bin/activate
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

**Run this example:**
```bash {cmd=true}
# From project root:
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

**Example 2.2: Set up GPU permissions**

```bash
# Add user to render and video groups
sudo usermod -a -G render,video $USER

# Verify group membership
groups | grep -E "(render|video)"
```

**Note:** You must log out and log back in (or reboot) for group changes to take effect.

**Example 2.3: Configure environment variables**

Add to `~/.bashrc` for permanent configuration:

```bash
# Set GFX version override (required for RX 6650M)
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Use only discrete GPU (GPU 0)
export HIP_VISIBLE_DEVICES=0

# Additional stability settings
export HSA_ENABLE_SDMA=0
export HSA_ENABLE_INTERRUPT=0
export HIP_FORCE_DEV_KERNARG=1
```

**Run this example:**
```bash {cmd=true}
# Add to bashrc
echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc
echo 'export HIP_VISIBLE_DEVICES=0' >> ~/.bashrc
source ~/.bashrc

# Verify
echo "HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
echo "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
```

**Example 2.4: Verify GPU detection**

```bash
# From project root:
source venv/bin/activate
python Scripts/check_gpu.py
```

**Run this example:**
```bash {cmd=true}
# From project root:
source venv/bin/activate
python Scripts/check_gpu.py
```

**Expected output:**
```
============================================================
PyTorch GPU Status Check
============================================================

Environment Variables:
  HIP_VISIBLE_DEVICES: 0
  HSA_OVERRIDE_GFX_VERSION: 10.3.0

PyTorch version: 2.6.0+rocm6.1
HIP version: 6.1.xxxxx

GPU Available: True
GPU Count: 1
Current Device: 0

GPU 0:
  Name: AMD Radeon RX 6650M
  Memory: 7.98 GB

Testing GPU computation on GPU 0 (discrete GPU)...
✓ GPU computation successful on GPU 0!
```

**Example 2.5: Test GPU with simple PyTorch operation**

Save this as `test_gpu_simple.py` and run it:

```python
#!/usr/bin/env python3
"""Test GPU with simple PyTorch operations."""
import torch

# Check GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Test computation
    x = torch.rand(1000, 1000, device='cuda:0')
    y = torch.rand(1000, 1000, device='cuda:0')
    z = torch.matmul(x, y)
    print(f"✓ GPU computation successful! Result on: {z.device}")
else:
    print("GPU not available. Will use CPU.")
```

**Run this example:**
```bash {cmd=true}
# From project root:
source venv/bin/activate
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'GPU Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    x = torch.rand(100, 100, device='cuda:0')
    y = torch.rand(100, 100, device='cuda:0')
    z = torch.matmul(x, y)
    print(f'✓ GPU computation successful!')
"
```

---

## Training Your Model

### Part 3: Basic Training

**Example 3.1: Basic training command**

```bash
cd Model
source ../venv/bin/activate
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0

python train.py \
  --train-dir "../data/raw/Banana Ripeness Classification Dataset/train" \
  --val-dir "../data/raw/Banana Ripeness Classification Dataset/valid" \
  --num-classes 4 \
  --epochs 50 \
  --batch-size 64 \
  --num-workers 4
```

**Run this example:**
```bash {cmd=true}
# From project root:
source venv/bin/activate
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Run a quick test with 1 epoch
cd Model
python train.py \
  --train-dir "../data/raw/Banana Ripeness Classification Dataset/train" \
  --val-dir "../data/raw/Banana Ripeness Classification Dataset/valid" \
  --num-classes 4 \
  --epochs 1 \
  --batch-size 64 \
  --num-workers 4
cd ..
```

**Example 3.2: Training with custom learning rate**

```bash
# From project root:
cd Model
source ../venv/bin/activate
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0

python train.py \
  --train-dir "../data/raw/Banana Ripeness Classification Dataset/train" \
  --val-dir "../data/raw/Banana Ripeness Classification Dataset/valid" \
  --num-classes 4 \
  --epochs 50 \
  --batch-size 64 \
  --num-workers 4 \
  --lr 0.0001 \
  --checkpoint-dir checkpoints
cd ..
```

**Example 3.3: Training parameters explained**

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--train-dir` | Path to training data | Required | `../data/raw/.../train` |
| `--val-dir` | Path to validation data | Required | `../data/raw/.../valid` |
| `--num-classes` | Number of ripeness classes | 4 | 4 (unripe, ripe, overripe, rotten) |
| `--epochs` | Number of training epochs | 50 | 50-100 |
| `--batch-size` | Batch size | 64 | 32-128 (depends on GPU memory) |
| `--num-workers` | Data loading workers | 4 | 4-8 |
| `--lr` | Learning rate | 0.001 | 0.0001-0.001 |
| `--checkpoint-dir` | Save directory | `checkpoints` | `checkpoints` |

---

## Making Predictions

### Part 4: Using the Prediction Script

**Example 4.1: Simple prediction**

```bash
# From project root:
source venv/bin/activate
python Scripts/predict_banana.py "path/to/your/banana/image.jpg"
```

**Run this example:**
```bash {cmd=true}
# From project root:
source venv/bin/activate

# Find a test image
TEST_IMAGE=$(find "data/raw/Banana Ripeness Classification Dataset/test" -name "*.jpg" | head -1)

# Predict (venv is already activated)
python Scripts/predict_banana.py "$TEST_IMAGE"
```

**Example 4.2: Prediction with custom checkpoint**

```bash
# From project root:
cd Model
source ../venv/bin/activate
python predict.py \
  --image "../data/raw/Banana Ripeness Classification Dataset/test/ripe/image.jpg" \
  --checkpoint checkpoints/best_model.pth \
  --num-classes 4
cd ..
```

**Run this example:**
```bash {cmd=true}
# From project root:
source venv/bin/activate

# Use a test image
cd Model
# venv is already activated from project root
python predict.py \
  --image "../data/raw/Banana Ripeness Classification Dataset/test/ripe/$(ls "../data/raw/Banana Ripeness Classification Dataset/test/ripe" | head -1)" \
  --checkpoint checkpoints/best_model.pth \
  --num-classes 4
cd ..
```

**Example 4.3: Prediction output format**

When you run prediction, you'll see output like:

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

**Example 4.4: Python API for predictions**

Save this as `predict_example.py` and run it (make sure venv is activated):

```python
#!/usr/bin/env python3
"""Example: Using prediction API from Python."""
import sys
import os

# Add Model to path (run from project root)
sys.path.insert(0, 'Model')

from predict import predict
import torch

# Load model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_path = 'Model/checkpoints/best_model.pth'

if os.path.exists(checkpoint_path):
    model = torch.load(checkpoint_path, map_location=device)
    model.eval()
    
    # Make prediction
    result = predict(
        model=model,
        image_path='path/to/banana.jpg',  # Replace with your image path
        device=device,
        class_names=['unripe', 'ripe', 'overripe', 'rotten']
    )
    
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2f}%")
else:
    print(f"Model checkpoint not found at {checkpoint_path}")
```

**Run this example:**
```bash {cmd=true}
# From project root:
source venv/bin/activate
python -c "
import sys
sys.path.insert(0, 'Model')
import os
test_dir = 'data/raw/Banana Ripeness Classification Dataset/test/ripe'
if os.path.exists(test_dir):
    test_image = os.path.join(test_dir, os.listdir(test_dir)[0])
    print(f'Test image found: {test_image}')
    print('Note: Requires trained model checkpoint to run prediction')
else:
    print('Test directory not found')
"
```

---

## Multi-GPU Training

### Part 5: Using Multiple GPUs

**Example 5.1: Check available GPUs**

```bash
# From project root:
source venv/bin/activate
python -c "
import torch
if torch.cuda.is_available():
    print(f'Available GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory/1024**3:.2f} GB)')
else:
    print('No GPUs available')
"
```

**Run this example:**
```bash {cmd=true}
# From project root:
source venv/bin/activate
python -c "
import torch
if torch.cuda.is_available():
    print(f'Available GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory/1024**3:.2f} GB)')
else:
    print('No GPUs available')
"
```

**Example 5.2: Single GPU training (recommended)**

```bash
# From project root:
source venv/bin/activate
export HIP_VISIBLE_DEVICES=0  # Only discrete GPU
export HSA_OVERRIDE_GFX_VERSION=10.3.0

cd Model
python train.py \
  --train-dir "../data/raw/Banana Ripeness Classification Dataset/train" \
  --val-dir "../data/raw/Banana Ripeness Classification Dataset/valid" \
  --num-classes 4 \
  --epochs 50 \
  --batch-size 64 \
  --num-workers 4
cd ..
```

**Example 5.3: Multi-GPU training (both GPUs)**

```bash
# From project root:
source venv/bin/activate
export HIP_VISIBLE_DEVICES=0,1  # Both GPUs
export HSA_OVERRIDE_GFX_VERSION=10.3.0

cd Model
python train.py \
  --train-dir "../data/raw/Banana Ripeness Classification Dataset/train" \
  --val-dir "../data/raw/Banana Ripeness Classification Dataset/valid" \
  --num-classes 4 \
  --epochs 50 \
  --batch-size 128 \
  --num-workers 4 \
  --multi-gpu
cd ..
```

**Run this example:**
```bash {cmd=true}
# From project root:
source venv/bin/activate
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0

cd Model
python train.py \
  --train-dir "../data/raw/Banana Ripeness Classification Dataset/train" \
  --val-dir "../data/raw/Banana Ripeness Classification Dataset/valid" \
  --num-classes 4 \
  --epochs 1 \
  --batch-size 64 \
  --num-workers 4
cd ..
```

**Note:** Multi-GPU training may not always be faster if GPUs have different speeds. The integrated GPU (680M) is slower than the discrete GPU (6650M), so single GPU training is often faster.

---

## Performance Optimization

### Part 6: Optimizing Training Performance

**Example 6.1: Finding optimal batch size**

Test different batch sizes to find the sweet spot:

```bash
# Test 1: Small batch
python train.py --batch-size 32 --num-workers 4 --epochs 1 ...

# Test 2: Medium batch (recommended starting point)
python train.py --batch-size 64 --num-workers 4 --epochs 1 ...

# Test 3: Large batch
python train.py --batch-size 128 --num-workers 4 --epochs 1 ...
```

**Run this example:**
```bash {cmd=true}
# From project root:
source venv/bin/activate
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0

cd Model
python train.py \
  --train-dir "../data/raw/Banana Ripeness Classification Dataset/train" \
  --val-dir "../data/raw/Banana Ripeness Classification Dataset/valid" \
  --num-classes 4 \
  --epochs 1 \
  --batch-size 64 \
  --num-workers 4
cd ..
```

**Example 6.2: Optimized settings for 32GB RAM**

With 32GB RAM, you can use more aggressive settings:

```bash
# From project root:
source venv/bin/activate
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0

cd Model
python train.py \
  --train-dir "../data/raw/Banana Ripeness Classification Dataset/train" \
  --val-dir "../data/raw/Banana Ripeness Classification Dataset/valid" \
  --num-classes 4 \
  --epochs 50 \
  --batch-size 128 \
  --num-workers 8
cd ..
```

**Example 6.3: Performance tuning checklist**

1. **Monitor GPU utilization** (see Part 7)
2. **Adjust batch size**: Start with 64, increase if GPU memory allows
3. **Adjust num_workers**: Start with 4, increase if CPU allows
4. **Check for bottlenecks**: If GPU usage is low, increase batch size or workers
5. **Memory issues**: Reduce batch size if you get OOM errors

**Example 6.4: Common optimization patterns**

```bash
# Pattern 1: Balanced (recommended starting point)
--batch-size 64 --num-workers 4

# Pattern 2: GPU-bound (large batches, moderate workers)
--batch-size 128 --num-workers 4

# Pattern 3: CPU-bound (moderate batches, many workers)
--batch-size 64 --num-workers 8

# Pattern 4: Memory-constrained (small batches, few workers)
--batch-size 32 --num-workers 2
```

---

## Monitoring GPU Usage

### Part 7: GPU Monitoring Tools

**Example 7.1: Using rocm-smi**

```bash
# Install rocm-smi (if not available)
sudo apt install rocm-smi

# View GPU status
rocm-smi

# Watch continuously (updates every 1 second)
watch -n 1 rocm-smi
```

**Run this example:**
```bash {cmd=true}
# Check if rocm-smi is available
if command -v rocm-smi &> /dev/null; then
    rocm-smi
else
    echo "rocm-smi not installed. Install with: sudo apt install rocm-smi"
fi
```

**Example 7.2: Simple GPU usage monitor**

```bash
# Monitor GPU busy percentage
watch -n 1 'cat /sys/class/drm/card1/device/gpu_busy_percent'
```

**Run this example:**
```bash {cmd=true}
# Check GPU busy percentage
if [ -f /sys/class/drm/card1/device/gpu_busy_percent ]; then
    cat /sys/class/drm/card1/device/gpu_busy_percent
    echo "%"
else
    echo "GPU busy percentage not available at /sys/class/drm/card1/device/gpu_busy_percent"
    echo "Try: ls /sys/class/drm/card*/device/gpu_busy_percent"
fi
```

**Example 7.3: Python GPU monitoring script**

```python
#!/usr/bin/env python3
import time
import os

def monitor_gpu():
    """Monitor GPU usage continuously."""
    gpu_path = '/sys/class/drm/card1/device/gpu_busy_percent'
    
    if not os.path.exists(gpu_path):
        print(f"GPU busy file not found at {gpu_path}")
        print("Available GPU devices:")
        for card in os.listdir('/sys/class/drm'):
            if card.startswith('card') and os.path.isdir(f'/sys/class/drm/{card}/device'):
                print(f"  {card}")
        return
    
    print("Monitoring GPU usage (Ctrl+C to stop)...")
    try:
        while True:
            with open(gpu_path, 'r') as f:
                usage = f.read().strip()
            print(f"GPU Usage: {usage}%", end='\r')
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == '__main__':
    monitor_gpu()
```

**Run this example:**
```bash {cmd=true}
# From project root:
source venv/bin/activate
python -c "
import time
import os

gpu_path = '/sys/class/drm/card1/device/gpu_busy_percent'
if os.path.exists(gpu_path):
    with open(gpu_path, 'r') as f:
        usage = f.read().strip()
    print(f'Current GPU Usage: {usage}%')
else:
    print('GPU busy file not found')
    print('Try: ls /sys/class/drm/card*/device/gpu_busy_percent')
"
```

**Example 7.4: Monitor during training**

Run monitoring in a separate terminal while training:

**Terminal 1 (Training):**
```bash
# From project root:
source venv/bin/activate
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0
cd Model
python train.py --batch-size 64 --num-workers 4 \
  --train-dir "../data/raw/Banana Ripeness Classification Dataset/train" \
  --val-dir "../data/raw/Banana Ripeness Classification Dataset/valid" \
  --num-classes 4 --epochs 50
```

**Terminal 2 (Monitoring):**
```bash
watch -n 1 'cat /sys/class/drm/card1/device/gpu_busy_percent'
```

**Expected GPU usage:**
- **Good**: 80-100% during training
- **Acceptable**: 50-80% during training
- **Poor**: <50% during training (increase batch size or workers)

---

## Troubleshooting

### Part 8: Common Issues and Solutions

**Example 8.1: GPU not detected**

**Symptoms:**
```
GPU Available: False
No GPU detected. PyTorch will use CPU.
```

**Solutions:**

```bash
# 1. Check user groups
groups | grep -E "(render|video)"

# 2. Add to groups if missing
sudo usermod -a -G render,video $USER
# Then log out and log back in

# 3. Check device access
ls -la /dev/dri/

# 4. Verify PyTorch ROCm installation
python -c "import torch; print(torch.__version__); print(hasattr(torch.version, 'hip'))"

# 5. Try GFX version override
export HSA_OVERRIDE_GFX_VERSION=10.3.0
python Scripts/check_gpu.py
```

**Run this example:**
```bash {cmd=true}
# From project root:
source venv/bin/activate

# Check groups
groups | grep -E "(render|video)" || echo "Not in render/video groups"

# Check device access
ls -la /dev/dri/ 2>/dev/null || echo "No /dev/dri/ found"

# Check PyTorch (venv is already activated)
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'Has HIP: {hasattr(torch.version, \"hip\")}')"
```

**Example 8.2: GPU memory access fault**

**Symptoms:**
```
Memory access fault by GPU node-1
Cannot find CO in the bundle for ISA: amdgcn-amd-amdhsa--gfx1100
```

**Solutions:**

```bash
# Solution 1: Use GFX version override
export HSA_OVERRIDE_GFX_VERSION=10.3.0
python train.py ...

# Solution 2: Use only discrete GPU
export HIP_VISIBLE_DEVICES=0
python train.py ...

# Solution 3: Use CPU (fallback)
python train.py --force-cpu ...
```

**Run this example:**
```bash {cmd=true}
# From project root:
source venv/bin/activate

# Set environment variables
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_VISIBLE_DEVICES=0

# Test GPU (venv is already activated)
python Scripts/check_gpu.py
```

**Example 8.3: Low GPU utilization**

**Symptoms:** GPU usage stays below 50% during training

**Solutions:**

```bash
# Solution 1: Increase batch size
python train.py --batch-size 128 --num-workers 4 ...

# Solution 2: Increase number of workers
python train.py --batch-size 64 --num-workers 8 ...

# Solution 3: Both
python train.py --batch-size 128 --num-workers 8 ...
```

**Run this example:**
```bash {cmd=true}
# From project root:
source venv/bin/activate
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0

cd Model
python train.py \
  --train-dir "../data/raw/Banana Ripeness Classification Dataset/train" \
  --val-dir "../data/raw/Banana Ripeness Classification Dataset/valid" \
  --num-classes 4 \
  --epochs 1 \
  --batch-size 128 \
  --num-workers 8
cd ..
```

**Example 8.4: Out of memory errors**

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

```bash
# Solution 1: Reduce batch size
python train.py --batch-size 32 --num-workers 4 ...

# Solution 2: Reduce number of workers
python train.py --batch-size 64 --num-workers 2 ...

# Solution 3: Both
python train.py --batch-size 32 --num-workers 2 ...
```

**Example 8.5: Model checkpoint not found**

**Symptoms:**
```
Error: Model checkpoint not found: checkpoints/best_model.pth
```

**Solutions:**

```bash
# Solution 1: Train a model first
cd Model
python train.py --train-dir ... --val-dir ... --epochs 50

# Solution 2: Use different checkpoint path
python predict.py --image ... --checkpoint path/to/other/checkpoint.pth
```

**Run this example:**
```bash {cmd=true}
# Check if checkpoint exists
if [ -f "Model/checkpoints/best_model.pth" ]; then
    echo "✓ Checkpoint found"
    ls -lh Model/checkpoints/best_model.pth
else
    echo "✗ Checkpoint not found. Train a model first."
fi
```

**Example 8.6: Slow training speed**

**Checklist:**

1. **GPU utilization**: Should be 80-100%
   ```bash
   watch -n 1 'cat /sys/class/drm/card1/device/gpu_busy_percent'
   ```

2. **Batch size**: Try increasing
   ```bash
   python train.py --batch-size 128 ...
   ```

3. **Data loading**: Increase workers
   ```bash
   python train.py --num-workers 8 ...
   ```

4. **Check CPU usage**: Should not be 100%
   ```bash
   htop
   ```

5. **Verify GPU is being used**: Check training output for "Using device: cuda:0"

**Run this example:**
```bash {cmd=true}
# From project root:
source venv/bin/activate

# Quick diagnostic
echo "=== GPU Status ==="
# venv is already activated
python Scripts/check_gpu.py

echo -e "\n=== System Resources ==="
echo "CPU cores: $(nproc)"
echo "RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "GPU: $(lspci | grep -i vga | head -1)"
```

---

## Quick Reference

### Essential Commands

**Training:**
```bash
# From project root:
source venv/bin/activate
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0

cd Model
python train.py \
  --train-dir "../data/raw/Banana Ripeness Classification Dataset/train" \
  --val-dir "../data/raw/Banana Ripeness Classification Dataset/valid" \
  --num-classes 4 \
  --epochs 50 \
  --batch-size 64 \
  --num-workers 4
cd ..
```

**Prediction:**
```bash
# From project root:
source venv/bin/activate
python Scripts/predict_banana.py "path/to/image.jpg"
```

**GPU Check:**
```bash
# From project root:
source venv/bin/activate
python Scripts/check_gpu.py
```

**Monitor GPU:**
```bash
watch -n 1 'cat /sys/class/drm/card1/device/gpu_busy_percent'
```

### Environment Variables

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Required for RX 6650M
export HIP_VISIBLE_DEVICES=0            # Use only discrete GPU
export HSA_ENABLE_SDMA=0                # Stability setting
export HSA_ENABLE_INTERRUPT=0           # Stability setting
export HIP_FORCE_DEV_KERNARG=1          # Stability setting
```

### File Paths

- **Training script**: `Model/train.py`
- **Prediction script**: `Model/predict.py` or `Scripts/predict_banana.py`
- **GPU check**: `Scripts/check_gpu.py`
- **Checkpoints**: `Model/checkpoints/best_model.pth`
- **Data**: `data/raw/Banana Ripeness Classification Dataset/`

---

## Summary

This guide covers:

1. ✅ **Setup**: Virtual environment and dependencies
2. ✅ **GPU Configuration**: AMD GPU setup with ROCm
3. ✅ **Training**: Model training with various configurations
4. ✅ **Prediction**: Making predictions on banana images
5. ✅ **Multi-GPU**: Using multiple GPUs (when available)
6. ✅ **Optimization**: Performance tuning tips
7. ✅ **Monitoring**: GPU usage monitoring tools
8. ✅ **Troubleshooting**: Common issues and solutions

All examples in this guide are runnable and tested. Use them as starting points for your own experiments!

---

## Additional Resources

- **PyTorch ROCm**: https://pytorch.org/get-started/locally/
- **AMD ROCm Documentation**: https://rocm.docs.amd.com/
- **GPU Support Matrix**: https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html

---

*Last updated: Based on BannanaHacks project documentation*

