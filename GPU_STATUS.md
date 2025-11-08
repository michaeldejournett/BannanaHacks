# AMD GPU Status Summary

## ✅ What's Working

1. **PyTorch with ROCm installed** (2.6.0+rocm6.1)
2. **GPU detected** - Both GPUs are visible:
   - GPU 0: AMD Radeon RX 6650M (Discrete, 7.98 GB) ✅
   - GPU 1: AMD Radeon 680M (Integrated, 15.29 GB)
3. **Code updated** to use GPU 0 explicitly

## ⚠️ Current Issue

There's a memory access fault when trying to use the GPU for computation. This appears to be a ROCm/Ubuntu 24.10 compatibility issue where ROCm tries to access both GPUs even when `HIP_VISIBLE_DEVICES=0` is set.

**Error:** `Memory access fault by GPU node-1` - suggests ROCm is still trying to use the integrated GPU.

## 🔧 Solutions

### Option 1: Use CPU (Recommended for now)

The code automatically falls back to CPU if GPU isn't available. Training will work, just slower:

```bash
# Training will automatically use CPU
cd Model
python train.py --train-dir ... --val-dir ...
```

### Option 2: Try System ROCm Installation

The pip-installed ROCm wheels might have issues. You could try installing system ROCm:

```bash
# This may not work on Ubuntu 24.10, but worth trying
sudo apt install rocm-dkms rocm-libs
```

### Option 3: Wait for ROCm Updates

Ubuntu 24.10 is very new. ROCm support may improve with future updates.

### Option 4: Use Ubuntu 22.04 LTS

If GPU acceleration is critical, consider using Ubuntu 22.04 LTS which has better ROCm support.

## 📝 Code Status

All code has been updated to:
- Use GPU 0 explicitly when available
- Set `HIP_VISIBLE_DEVICES=0` automatically
- Fall back to CPU gracefully if GPU fails

The training script and webapp will work fine on CPU - they'll just be slower.

## 🧪 Testing

To test if GPU works in your specific use case:

```bash
# Try a simple PyTorch operation
source venv/bin/activate
python -c "import torch; x=torch.rand(10, device='cuda:0' if torch.cuda.is_available() else 'cpu'); print(x.device)"
```

If this works, GPU might work for your specific model operations even if the test script fails.

## 📚 References

- ROCm Documentation: https://rocm.docs.amd.com/
- PyTorch ROCm: https://pytorch.org/get-started/locally/
- GPU Support Matrix: https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html

