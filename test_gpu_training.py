#!/usr/bin/env python3
"""
Test script to isolate the GPU training issue.
"""

import os
# Set environment variables BEFORE importing torch
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
# Try additional ROCm environment variables
os.environ['HSA_ENABLE_SDMA'] = '0'
os.environ['HSA_ENABLE_INTERRUPT'] = '0'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

print("Testing GPU training components...")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Test 1: Simple model forward pass
print("\n1. Testing simple model forward pass...")
try:
    model = nn.Linear(10, 4).to(device)
    x = torch.randn(32, 10, device=device)
    y = model(x)
    print("   ✓ Forward pass OK")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")

# Test 2: Model with backward pass
print("\n2. Testing backward pass...")
try:
    model = nn.Linear(10, 4).to(device)
    x = torch.randn(32, 10, device=device)
    y = model(x)
    loss = y.sum()
    loss.backward()
    print("   ✓ Backward pass OK")
except Exception as e:
    print(f"   ✗ Backward pass failed: {e}")

# Test 3: DataLoader with GPU tensors
print("\n3. Testing DataLoader...")
class SimpleDataset(Dataset):
    def __init__(self, size=100):
        self.data = torch.randn(size, 3, 224, 224)
        self.labels = torch.randint(0, 4, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

try:
    dataset = SimpleDataset(100)
    loader = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=False)
    
    for i, (data, labels) in enumerate(loader):
        data = data.to(device)
        labels = labels.to(device)
        if i == 0:
            print(f"   ✓ DataLoader batch {i} OK")
            print(f"     Data shape: {data.shape}, Device: {data.device}")
            print(f"     Labels shape: {labels.shape}, Device: {labels.device}")
        if i >= 2:
            break
    print("   ✓ DataLoader OK")
except Exception as e:
    print(f"   ✗ DataLoader failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Full training step
print("\n4. Testing full training step...")
try:
    model = nn.Linear(224*224*3, 4).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    dataset = SimpleDataset(32)
    loader = DataLoader(dataset, batch_size=8, num_workers=0)
    
    for data, labels in loader:
        data = data.to(device)
        labels = labels.to(device)
        
        # Flatten
        data = data.view(data.size(0), -1)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"   ✓ Training step OK, Loss: {loss.item():.4f}")
        break
except Exception as e:
    print(f"   ✗ Training step failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")

