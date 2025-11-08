#!/usr/bin/env python3
"""
Test script to check AMD GPU access with proper environment variables.
"""

import os
import torch

# Set environment variables for AMD GPU
# Try different GFX versions if needed
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'  # For Navi 23 (RX 6650 XT)

print("=" * 60)
print("AMD GPU Detection Test")
print("=" * 60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"HIP version: {getattr(torch.version, 'hip', 'N/A')}")

# Check if CUDA/HIP is available
print(f"\nCUDA/HIP Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        try:
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
        except:
            pass
    
    # Test computation
    print("\nTesting GPU computation...")
    try:
        x = torch.rand(1000, 1000, device='cuda')
        y = torch.rand(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        print(f"✓ GPU computation successful!")
        print(f"  Result tensor device: {z.device}")
    except Exception as e:
        print(f"✗ GPU computation failed: {e}")
else:
    print("\nGPU not detected. Possible issues:")
    print("1. User not in render/video groups (run: sudo usermod -a -G render,video $USER)")
    print("2. Need to log out and back in (or reboot) after adding to groups")
    print("3. GPU may need different GFX version override")
    print("\nTrying alternative GFX versions...")
    
    # Try different GFX versions
    gfx_versions = ['11.0.0', '10.3.0', '10.1.0']
    for gfx in gfx_versions:
        os.environ['HSA_OVERRIDE_GFX_VERSION'] = gfx
        if torch.cuda.is_available():
            print(f"✓ Success with GFX version: {gfx}")
            break
    else:
        print("✗ No working GFX version found")

print("\n" + "=" * 60)


