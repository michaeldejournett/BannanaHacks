"""
Utility functions for GPU selection and management.
"""

import os
import torch


def get_best_gpu_device():
    """
    Select the best available GPU device.
    Prefers discrete GPUs over integrated GPUs.
    
    Returns:
        torch.device: The best GPU device, or CPU if no GPU available
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')
    
    # Set environment variable to prefer discrete GPU
    # This helps with systems that have both discrete and integrated GPUs
    os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')
    
    # For AMD systems with multiple GPUs, prefer GPU 0 (usually discrete)
    # You can override this by setting HIP_VISIBLE_DEVICES environment variable
    device = torch.device('cuda:0')
    
    # Test if GPU 0 works
    try:
        test_tensor = torch.rand(10, device=device)
        del test_tensor
        torch.cuda.empty_cache()
        return device
    except Exception as e:
        print(f"Warning: GPU 0 failed, trying CPU: {e}")
        return torch.device('cpu')


def print_gpu_info(device=None):
    """Print information about available GPUs."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        print(f"GPU Available: True")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            
            gpu_type = "Discrete" if "RX" in gpu_name or "Radeon RX" in gpu_name else "Integrated"
            
            print(f"\nGPU {i}: {gpu_name}")
            print(f"  Type: {gpu_type}")
            print(f"  Memory: {memory_gb:.2f} GB")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
        
        if device.type == 'cuda':
            print(f"\nSelected Device: GPU {device.index if hasattr(device, 'index') else 0}")
            print(f"Device Name: {torch.cuda.get_device_name(device.index if hasattr(device, 'index') else 0)}")
    else:
        print(f"GPU Available: False")
        print(f"Using Device: CPU")

