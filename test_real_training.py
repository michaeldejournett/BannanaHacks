#!/usr/bin/env python3
"""
Test training with actual dataset to isolate the issue.
"""

import os
# Set environment variables BEFORE importing torch
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HSA_ENABLE_SDMA'] = '0'
os.environ['HSA_ENABLE_INTERRUPT'] = '0'

import sys
sys.path.insert(0, 'Model')

import torch
from src.data.dataset import get_data_loaders
from src.models.model import BananaRipenessModel

print("Testing with actual dataset...")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load actual data
train_dir = "../data/raw/Banana Ripeness Classification Dataset/train"
val_dir = "../data/raw/Banana Ripeness Classification Dataset/valid"

print("\nLoading data loaders...")
try:
    train_loader, val_loader = get_data_loaders(
        train_dir, val_dir, batch_size=4, num_workers=0
    )
    print(f"✓ Data loaders created")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
except Exception as e:
    print(f"✗ Failed to create data loaders: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test loading a batch
print("\nTesting batch loading...")
try:
    for i, (images, labels) in enumerate(train_loader):
        print(f"  Batch {i}: images shape={images.shape}, labels shape={labels.shape}")
        print(f"    Images device: {images.device}, Labels device: {labels.device}")
        if i >= 2:
            break
    print("✓ Batch loading OK")
except Exception as e:
    print(f"✗ Batch loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test moving to GPU
print("\nTesting GPU transfer...")
try:
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        print(f"  Batch {i}: Moved to {device}")
        print(f"    Images device: {images.device}, Labels device: {labels.device}")
        if i >= 1:
            break
    print("✓ GPU transfer OK")
except Exception as e:
    print(f"✗ GPU transfer failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test model forward pass
print("\nTesting model forward pass...")
try:
    model = BananaRipenessModel(num_classes=4).to(device)
    model.eval()
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        print(f"  Batch {i}: Forward pass OK, output shape={outputs.shape}")
        if i >= 1:
            break
    print("✓ Model forward pass OK")
except Exception as e:
    print(f"✗ Model forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test training step
print("\nTesting training step...")
try:
    model = BananaRipenessModel(num_classes=4).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"  Batch {i}: Training step OK, Loss={loss.item():.4f}")
        if i >= 2:
            break
    print("✓ Training step OK")
except Exception as e:
    print(f"✗ Training step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed! GPU training should work.")

