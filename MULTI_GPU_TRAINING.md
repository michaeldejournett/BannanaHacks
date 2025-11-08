# Multi-GPU Training Guide

## Your GPU Setup

You have 2 GPUs:
- **GPU 0**: AMD Radeon RX 6650M (Discrete, 7.98 GB) - Faster
- **GPU 1**: AMD Radeon 680M (Integrated, 15.29 GB) - Slower

## Multi-GPU Options

### Option 1: Use Both GPUs (DataParallel)

```bash
cd Model
source ../venv/bin/activate
export HIP_VISIBLE_DEVICES=0,1  # Make both GPUs visible
export HSA_OVERRIDE_GFX_VERSION=10.3.0

python train.py \
  --train-dir "../data/raw/Banana Ripeness Classification Dataset/train" \
  --val-dir "../data/raw/Banana Ripeness Classification Dataset/valid" \
  --num-classes 4 \
  --epochs 50 \
  --batch-size 64 \
  --num-workers 4 \
  --multi-gpu
```

### Option 2: Use Only Discrete GPU (Recommended)

The integrated GPU (680M) is slower and may bottleneck training:

```bash
export HIP_VISIBLE_DEVICES=0  # Only discrete GPU
python train.py \
  --train-dir "../data/raw/Banana Ripeness Classification Dataset/train" \
  --val-dir "../data/raw/Banana Ripeness Classification Dataset/valid" \
  --num-classes 4 \
  --epochs 50 \
  --batch-size 64 \
  --num-workers 4
```

### Option 3: Use Specific GPUs

```bash
export HIP_VISIBLE_DEVICES=0,1
python train.py \
  --train-dir "../data/raw/Banana Ripeness Classification Dataset/train" \
  --val-dir "../data/raw/Banana Ripeness Classification Dataset/valid" \
  --num-classes 4 \
  --epochs 50 \
  --batch-size 64 \
  --num-workers 4 \
  --multi-gpu \
  --gpu-ids "0,1"  # Use both GPUs
```

Or use only GPU 0:
```bash
python train.py \
  ... \
  --multi-gpu \
  --gpu-ids "0"  # Only discrete GPU
```

## Important Notes

### ⚠️ Mixed GPU Performance

Using both GPUs may **not always be faster** because:
1. **Integrated GPU is slower**: The 680M will process batches slower than 6650M
2. **Synchronization overhead**: DataParallel waits for the slowest GPU
3. **Memory transfer**: Data needs to be split and gathered

### ✅ When Multi-GPU Helps

- Both GPUs are similar speed
- Large batch sizes (splits work across GPUs)
- Model is large enough to benefit from parallelization

### 📊 Testing Performance

Test both configurations:

**Single GPU (GPU 0 only):**
```bash
export HIP_VISIBLE_DEVICES=0
python train.py ... --batch-size 64
```

**Multi-GPU (both):**
```bash
export HIP_VISIBLE_DEVICES=0,1
python train.py ... --batch-size 128 --multi-gpu
```

Compare training speed and GPU utilization to see which is faster!

## How DataParallel Works

- Splits each batch across available GPUs
- Each GPU processes part of the batch
- Gradients are averaged across GPUs
- Model is replicated on each GPU

**Example**: With batch_size=128 and 2 GPUs:
- GPU 0 processes 64 samples
- GPU 1 processes 64 samples
- Results are combined

## Troubleshooting

If multi-GPU is slower:
1. Use only GPU 0: `--gpu-ids "0"`
2. Increase batch size: `--batch-size 128` or `256`
3. Check GPU utilization: Both GPUs should be busy

If you get errors:
1. Make sure both GPUs are visible: `export HIP_VISIBLE_DEVICES=0,1`
2. Try smaller batch size
3. Check GPU memory: `rocm-smi`

