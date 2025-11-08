# GPU Memory Fault - Root Cause Analysis

## The Problem

Your GPU is **detected correctly**, but PyTorch's ROCm libraries **don't have compiled code objects** for your GPU architecture (`gfx1100` = Navi 23).

### Error Details

From the diagnostic output:
```
Cannot find CO in the bundle .../libhipblaslt.so for ISA: amdgcn-amd-amdhsa--gfx1100
Missing CO for these ISAs - amdgcn-amd-amdhsa--gfx1100
Memory access fault by GPU node-1
```

### What This Means

1. **GPU Detection**: ✅ Works - PyTorch sees your RX 6650M
2. **GPU Architecture**: `gfx1100` (Navi 23)
3. **Missing Binaries**: PyTorch ROCm wheels don't include compiled kernels for `gfx1100`
4. **Result**: When ROCm tries to run GPU code, it crashes because the binary isn't available

## Why This Happens

- PyTorch ROCm wheels are pre-compiled for specific GPU architectures
- Your RX 6650M uses `gfx1100` architecture
- The PyTorch ROCm 6.1 wheels may not include `gfx1100` support
- Ubuntu 24.10 is very new - ROCm support may be incomplete

## Solutions

### Option 1: Use GFX Version Override (Try This First)

Try overriding to a supported architecture that's close:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Try gfx1030 (RDNA2)
# or
export HSA_OVERRIDE_GFX_VERSION=10.1.0  # Try gfx1010 (RDNA1)
```

Add to `~/.bashrc`:
```bash
echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc
source ~/.bashrc
```

Then test:
```bash
python diagnose_gpu.py
```

**Note**: This tells ROCm to use kernels compiled for a different architecture. It may work but could have performance/compatibility issues.

### Option 2: Use CPU (Current Workaround)

The code automatically falls back to CPU. Training will work, just slower:

```bash
# Training will use CPU automatically
cd Model
python train.py --train-dir ... --val-dir ...
```

### Option 3: Wait for PyTorch Update

PyTorch ROCm may add `gfx1100` support in future releases. Check:
- PyTorch ROCm release notes
- AMD ROCm documentation

### Option 4: Use Docker with Pre-built ROCm

AMD provides Docker images with ROCm that may have better GPU support:

```bash
# Example (check AMD docs for latest)
docker run --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined rocm/pytorch:rocm6.1_ubuntu22.04_py3.11_pytorch_2.6.0
```

### Option 5: Build PyTorch from Source (Advanced)

Build PyTorch with ROCm support for your specific GPU architecture. This is complex and time-consuming.

## Current Status

- ✅ GPU detected
- ✅ PyTorch ROCm installed
- ❌ GPU computation fails (missing binaries)
- ✅ CPU fallback works

## Recommendation

For now, **use CPU** - your training and webapp will work fine, just slower. When PyTorch ROCm adds `gfx1100` support, GPU acceleration will work automatically.

You can try Option 1 (GFX override) but it's not guaranteed to work and may have issues.

