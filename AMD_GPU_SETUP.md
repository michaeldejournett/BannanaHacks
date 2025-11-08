# AMD GPU Setup Guide for PyTorch

## Overview
To use your AMD GPU (Radeon RX 6650 XT) with PyTorch, you need PyTorch with ROCm support. For Ubuntu 24.10, the easiest method is to use PyTorch wheels that include ROCm runtime.

## ⚠️ Important for Ubuntu 24.10 Users

Ubuntu 24.10 (Oracular) is very new and may not have official ROCm repository support yet. **We recommend using Option A (pip-based installation)** which doesn't require system ROCm packages.

## Step 1: Install PyTorch with ROCm Support

### Option A: Install PyTorch with ROCm Wheels (Recommended for Ubuntu 24.10)

This method installs PyTorch with bundled ROCm runtime - no system ROCm installation needed!

```bash
source venv/bin/activate
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

**Note:** If rocm6.1 doesn't work, try rocm6.0 or rocm5.7:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

### Option B: Install System ROCm First (For Ubuntu 22.04/23.04)

**⚠️ This may not work on Ubuntu 24.10**

If you want to try system ROCm installation:

1. **Remove any broken ROCm repository (if exists):**
```bash
sudo rm -f /etc/apt/sources.list.d/rocm.list
```

2. **Add AMD ROCm repository (for Ubuntu 22.04 - may work on 24.10):**
```bash
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.1/ jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list
```

3. **Update and install ROCm:**
```bash
sudo apt update
sudo apt install rocm-dkms rocm-libs rocm-dev
```

4. **Add your user to the render group:**
```bash
sudo usermod -a -G render $USER
sudo usermod -a -G video $USER
```

5. **Reboot your system:**
```bash
sudo reboot
```

6. **Verify ROCm installation:**
```bash
rocm-smi
```

7. **Then install PyTorch with ROCm:**
```bash
source venv/bin/activate
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

## Step 3: Verify GPU Detection

Run this Python script to check if PyTorch detects your GPU:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Current device: {torch.cuda.current_device()}")
else:
    print("GPU not detected. Using CPU.")
```

**Note:** Even with ROCm, PyTorch uses `torch.cuda.is_available()` to check for GPU. This is normal - "CUDA" in this context refers to GPU compute capability, not NVIDIA CUDA.

## Step 2: Fix GPU Permissions

**CRITICAL:** You must be in the `render` and `video` groups to access the GPU:

```bash
sudo usermod -a -G render,video $USER
```

**Then log out and log back in** (or reboot) for the changes to take effect.

Verify you're in the groups:
```bash
groups
```

You should see `render` and `video` in the output.

## Step 3: Set Environment Variables (If Needed)

Some AMD GPUs need a GFX version override. Add this to your `~/.bashrc` or run before Python:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

**Important**: For RX 6650M (gfx1100), use `10.3.0` instead of `11.0.0` because PyTorch ROCm wheels don't include gfx1100 binaries. The gfx1030 kernels are compatible and work correctly.

For other GPUs, try:
- `11.0.0` (if PyTorch has gfx1100 support)
- `10.3.0` (RDNA2 - works for many Navi GPUs)
- `10.1.0` (RDNA1 - fallback)

To make it permanent, add to `~/.bashrc`:
```bash
echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc  # Use gfx1030 (compatible with gfx1100)
echo 'export HIP_VISIBLE_DEVICES=0' >> ~/.bashrc  # Use only discrete GPU (GPU 0)
source ~/.bashrc
```

**Important:** 
- `HIP_VISIBLE_DEVICES=0` restricts PyTorch to use only GPU 0 (your discrete RX 6650M)
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` tells ROCm to use gfx1030 kernels (PyTorch ROCm doesn't have gfx1100 binaries)

## Step 4: Verify GPU Detection

Run the GPU check script:
```bash
source venv/bin/activate
python check_gpu.py
```

Or use the test script:
```bash
python test_amd_gpu.py
```

## Troubleshooting

1. **If ROCm installation fails:**
   - Check if your GPU is supported: https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html
   - Your RX 6650 XT should be supported (Navi 23 architecture)

2. **If PyTorch still doesn't detect GPU after Step 1-3:**
   - **Check permissions:** Make sure you're in `render` and `video` groups and logged out/in
   - **Check device access:** `ls -la /dev/dri/` should show render nodes
   - **Try different GFX version:** Change `HSA_OVERRIDE_GFX_VERSION` to `10.3.0` or `10.1.0`
   - **Verify PyTorch ROCm:** `python -c "import torch; print(torch.__version__); print(hasattr(torch.version, 'hip'))"`
   - **Check HIP version:** Should show HIP version if ROCm is working

3. **Common Issues:**
   - **"Permission denied" on /dev/dri/renderD128:** User not in render/video groups
   - **GPU not detected but HIP available:** Try different `HSA_OVERRIDE_GFX_VERSION`
   - **"No HIP devices found":** May need system ROCm libraries (try Option B)

4. **Alternative: Use CPU**
   - If GPU setup is too complex, the code will automatically fall back to CPU
   - Training will be slower but will still work

## Important Notes

- ROCm support varies by GPU model and Linux distribution
- Ubuntu 24.10 is very new - ROCm support may be limited
- The Radeon 680M (integrated GPU) may not be fully supported by ROCm
- Focus on using the RX 6650 XT (discrete GPU) for best results

