#!/usr/bin/env python3
"""
Test GPU 0 specifically with proper environment setup.
Run this BEFORE importing torch to ensure environment variables are set.
"""

import os
import sys

# CRITICAL: Set these BEFORE importing torch
os.environ['HIP_VISIBLE_DEVICES'] = '0'  # Use only GPU 0 (discrete)
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

# Now import torch
import torch

print("=" * 60)
print("GPU 0 Test (Discrete GPU Only)")
print("=" * 60)

print(f"\nEnvironment Variables:")
print(f"  HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES', 'not set')}")
print(f"  HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'not set')}")

print(f"\nPyTorch version: {torch.__version__}")
if hasattr(torch.version, 'hip'):
    print(f"HIP version: {torch.version.hip}")

print(f"\nGPU Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Visible GPU Count: {torch.cuda.device_count()}")
    
    if torch.cuda.device_count() > 0:
        print(f"\nUsing GPU 0: {torch.cuda.get_device_name(0)}")
        
        # Test computation
        print("\nTesting GPU computation...")
        try:
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            
            # Small test first
            x = torch.rand(100, 100, device=device)
            y = torch.rand(100, 100, device=device)
            z = torch.matmul(x, y)
            print(f"✓ Small test successful!")
            
            # Larger test
            del x, y, z
            torch.cuda.empty_cache()
            
            x = torch.rand(1000, 1000, device=device)
            y = torch.rand(1000, 1000, device=device)
            z = torch.matmul(x, y)
            print(f"✓ Large test successful!")
            print(f"  Result device: {z.device}")
            
            del x, y, z
            torch.cuda.empty_cache()
            print("\n✓ GPU 0 is working correctly!")
            
        except Exception as e:
            print(f"✗ GPU computation failed: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure HIP_VISIBLE_DEVICES=0 is set BEFORE importing torch")
            print("2. Try rebooting if the issue persists")
            print("3. Check GPU permissions: ls -la /dev/dri/")
    else:
        print("No GPUs visible (HIP_VISIBLE_DEVICES may be filtering them)")
else:
    print("GPU not available")

print("\n" + "=" * 60)

