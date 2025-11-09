#!/usr/bin/env python3
"""
Diagnostic script to understand the GPU memory fault issue.
"""

import os
import sys

# Set environment variables BEFORE importing torch
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

# Try additional ROCm environment variables
os.environ['HSA_ENABLE_SDMA'] = '0'  # Disable SDMA, might help
os.environ['AMD_LOG_LEVEL'] = '3'  # Enable verbose logging

print("=" * 60)
print("GPU Memory Fault Diagnostic")
print("=" * 60)

print("\nEnvironment Variables:")
for key in ['HIP_VISIBLE_DEVICES', 'HSA_OVERRIDE_GFX_VERSION', 'HSA_ENABLE_SDMA']:
    print(f"  {key}: {os.environ.get(key, 'not set')}")

print("\nImporting torch...")
try:
    import torch
    print(f"✓ PyTorch imported successfully")
    print(f"  Version: {torch.__version__}")
    
    if hasattr(torch.version, 'hip'):
        print(f"  HIP version: {torch.version.hip}")
    
    print("\nChecking GPU availability...")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA/HIP available: {cuda_available}")
    
    if cuda_available:
        print(f"  Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        
        print("\nTesting GPU operations (step by step)...")
        
        # Step 1: Set device
        print("  1. Setting device to cuda:0...")
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        print("     ✓ Device set")
        
        # Step 2: Create small tensor
        print("  2. Creating small tensor (10x10)...")
        try:
            x = torch.rand(10, 10, device=device)
            print(f"     ✓ Tensor created on {x.device}")
            
            # Step 3: Simple operation
            print("  3. Performing simple operation...")
            y = x * 2
            print(f"     ✓ Operation successful, result on {y.device}")
            
            # Step 4: Clean up
            print("  4. Cleaning up...")
            del x, y
            torch.cuda.empty_cache()
            print("     ✓ Cleanup successful")
            
            print("\n✓ All GPU operations successful!")
            
        except Exception as e:
            print(f"     ✗ Failed: {e}")
            print(f"     Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
    else:
        print("  GPU not available")
        
except Exception as e:
    print(f"✗ Failed to import or use torch: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)

