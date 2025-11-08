# Finding the Optimal GPU Training Settings

## The Problem: More Workers/Batch Size = Slower?

Sometimes increasing batch size and workers can actually slow things down due to:
1. **Overhead**: Too many workers create overhead
2. **Memory bandwidth**: Large batches stress memory transfer
3. **AMD GPU specifics**: `pin_memory` can be slower on AMD GPUs
4. **Context switching**: Too many workers cause CPU context switching overhead

## Optimal Settings (Tuned)

### Recommended Starting Point:
```bash
python train.py \
  --train-dir "../data/raw/Banana Ripeness Classification Dataset/train" \
  --val-dir "../data/raw/Banana Ripeness Classification Dataset/valid" \
  --num-classes 4 \
  --epochs 50 \
  --batch-size 64 \
  --num-workers 4
```

### If Still Slow, Try:
```bash
# Smaller batch, fewer workers
python train.py \
  --batch-size 32 \
  --num-workers 2 \
  ...
```

### If GPU Usage is Low, Try:
```bash
# Larger batch, but keep workers moderate
python train.py \
  --batch-size 128 \
  --num-workers 4 \
  ...
```

## Key Changes Made:

1. **pin_memory=False**: AMD GPUs sometimes perform better without pin_memory
2. **num_workers=4**: Sweet spot - not too many, not too few
3. **prefetch_factor=2**: Moderate prefetching
4. **batch_size=64**: Good balance

## Finding Your Sweet Spot:

Test different combinations:
- `--batch-size 32 --num-workers 2` (minimal)
- `--batch-size 64 --num-workers 4` (balanced) ← **Try this first**
- `--batch-size 128 --num-workers 4` (larger batches)
- `--batch-size 64 --num-workers 8` (more workers)

Monitor GPU usage and training speed to find what works best for your system!

