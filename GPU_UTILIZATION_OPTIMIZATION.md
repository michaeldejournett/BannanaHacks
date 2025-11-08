# GPU Utilization Optimization Guide

## Current Issue: Low GPU Utilization (12%)

Your GPU is working but underutilized. This is typically caused by:

1. **Data Loading Bottleneck**: CPU is loading/preprocessing images slower than GPU can process them
2. **Small Batch Size**: GPU sits idle waiting for more data
3. **Single-threaded Data Loading**: `num_workers=0` means all loading happens on main thread

## Optimizations Applied

### 1. Increased Data Loading Workers
- Changed `num_workers` from 0 to 4 (configurable via `--num-workers`)
- Multiple workers load/preprocess images in parallel
- Keeps GPU fed with data

### 2. Enabled Pin Memory
- `pin_memory=True` allows faster CPU→GPU transfer
- Data is pinned in CPU memory for quick GPU access

### 3. Added Prefetching
- `prefetch_factor=2` loads batches ahead of time
- Reduces GPU idle time waiting for data

### 4. Increased Default Batch Size
- Changed default from 32 to 64
- Larger batches = better GPU utilization
- Adjust based on GPU memory (RX 6650M has 8GB)

## Recommended Training Command (32GB RAM)

With 32GB RAM, you can use more aggressive settings:

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
  --batch-size 128 \
  --num-workers 8
```

### Even More Aggressive (if GPU memory allows):

```bash
python train.py \
  --train-dir "../data/raw/Banana Ripeness Classification Dataset/train" \
  --val-dir "../data/raw/Banana Ripeness Classification Dataset/valid" \
  --num-classes 4 \
  --epochs 50 \
  --batch-size 256 \
  --num-workers 8
```

**Note**: With 32GB RAM, you have plenty of memory for:
- Large batch sizes (128-256)
- 8+ data loading workers
- Aggressive prefetching (4x batches ahead)

## Tuning GPU Utilization

### If GPU usage is still low (with 32GB RAM):
1. **Increase batch size**: Try 256 or even 512 (if GPU memory allows)
2. **Increase num_workers**: Try 8-12 workers (you have RAM for it)
3. **Check CPU usage**: With 32GB RAM, CPU shouldn't be the bottleneck
4. **Monitor memory**: `free -h` to ensure you're not hitting swap

### If you get out-of-memory errors:
1. **Reduce batch size**: Try 32 or 16
2. **Reduce num_workers**: Try 2
3. **Enable gradient accumulation**: Process smaller batches but accumulate gradients

## Monitoring GPU Usage

Watch GPU utilization while training:
```bash
# In another terminal
watch -n 1 rocm-smi
# or
watch -n 1 cat /sys/class/drm/card*/device/gpu_busy_percent
```

You should see GPU utilization increase significantly with these optimizations!

