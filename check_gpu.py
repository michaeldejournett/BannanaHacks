#!/usr/bin/env python3
"""
Utility script to check GPU availability and PyTorch configuration.
"""

import os
import torch
import sys

def check_gpu_status():
    """Check and display GPU/CPU status for PyTorch."""
    print("=" * 60)
    print("PyTorch GPU Status Check")
    print("=" * 60)
    
    print(f"\nEnvironment Variables:")
    print(f"  HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES', 'not set')}")
    print(f"  HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'not set')}")
    
    print(f"\nPyTorch version: {torch.__version__}")
    
    # Check HIP version if available
    if hasattr(torch.version, 'hip'):
        print(f"HIP version: {torch.version.hip}")
    
    # Check if CUDA/ROCm is available
    cuda_available = torch.cuda.is_available()
    print(f"\nGPU Available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # Test GPU computation on GPU 0 (discrete GPU)
        print("\nTesting GPU computation on GPU 0 (discrete GPU)...")
        try:
            # Set to use GPU 0 explicitly
            os.environ['HIP_VISIBLE_DEVICES'] = '0'
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            
            x = torch.rand(1000, 1000, device=device)
            y = torch.rand(1000, 1000, device=device)
            z = torch.matmul(x, y)
            print(f"✓ GPU computation successful on GPU 0!")
            print(f"  Result tensor device: {z.device}")
            print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
            
            # Clean up
            del x, y, z
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ GPU computation failed: {e}")
            print(f"  This may be due to memory access issues with the integrated GPU.")
            print(f"  Try setting HIP_VISIBLE_DEVICES=0 to use only the discrete GPU.")
    else:
        print("\nNo GPU detected. PyTorch will use CPU.")
        print("\nTroubleshooting steps:")
        print("1. Make sure you're in render and video groups:")
        print("   sudo usermod -a -G render,video $USER")
        print("   Then log out and back in (or reboot)")
        print("\n2. Check GPU device access:")
        print("   ls -la /dev/dri/")
        print("\n3. Try different GFX version override:")
        print("   export HSA_OVERRIDE_GFX_VERSION=10.3.0")
        print("   python check_gpu.py")
        print("\n4. Verify PyTorch ROCm installation:")
        print("   python -c \"import torch; print(torch.__version__); print(hasattr(torch.version, 'hip'))\"")
    
    print("\n" + "=" * 60)
    
    return cuda_available

if __name__ == '__main__':
    gpu_available = check_gpu_status()
    sys.exit(0 if gpu_available else 1)

