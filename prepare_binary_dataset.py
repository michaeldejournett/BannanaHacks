#!/usr/bin/env python3
"""
Script to prepare binary classification dataset (banana vs not banana).
Combines ripe banana images with garbage classification images.
"""

import os
import shutil
from pathlib import Path

# Paths
BANANA_TRAIN_DIR = "data/raw/Banana Ripeness Classification Dataset/train/ripe"
BANANA_VAL_DIR = "data/raw/Banana Ripeness Classification Dataset/valid/ripe"
GARBAGE_DIR = "/home/michaeldejournett/.cache/kagglehub/datasets/asdasdasasdas/garbage-classification/versions/2"
BINARY_DATA_DIR = "data/binary_classification"

def find_images(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """Find all image files in a directory recursively."""
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                images.append(os.path.join(root, file))
    return images

def prepare_binary_dataset():
    """Prepare binary classification dataset."""
    print("Preparing binary classification dataset...")
    print("=" * 60)
    
    # Create directory structure
    for split in ['train', 'val']:
        for class_name in ['banana', 'not_banana']:
            os.makedirs(os.path.join(BINARY_DATA_DIR, split, class_name), exist_ok=True)
    
    # Get banana images
    print("\n1. Collecting banana images...")
    banana_train_images = find_images(BANANA_TRAIN_DIR)
    banana_val_images = find_images(BANANA_VAL_DIR)
    print(f"   Found {len(banana_train_images)} training banana images")
    print(f"   Found {len(banana_val_images)} validation banana images")
    
    # Copy banana images
    print("\n2. Copying banana images...")
    for img_path in banana_train_images:
        filename = os.path.basename(img_path)
        dest = os.path.join(BINARY_DATA_DIR, 'train', 'banana', filename)
        shutil.copy2(img_path, dest)
    
    for img_path in banana_val_images:
        filename = os.path.basename(img_path)
        dest = os.path.join(BINARY_DATA_DIR, 'val', 'banana', filename)
        shutil.copy2(img_path, dest)
    
    print(f"   ✓ Copied {len(banana_train_images)} training images")
    print(f"   ✓ Copied {len(banana_val_images)} validation images")
    
    # Get garbage images
    print("\n3. Collecting garbage (not banana) images...")
    garbage_images = find_images(GARBAGE_DIR)
    print(f"   Found {len(garbage_images)} garbage images")
    
    # Split garbage images into train/val (80/20)
    import random
    random.seed(42)
    random.shuffle(garbage_images)
    split_idx = int(len(garbage_images) * 0.8)
    garbage_train = garbage_images[:split_idx]
    garbage_val = garbage_images[split_idx:]
    
    # Copy garbage images
    print("\n4. Copying garbage images...")
    for img_path in garbage_train:
        filename = os.path.basename(img_path)
        # Handle duplicate filenames
        dest = os.path.join(BINARY_DATA_DIR, 'train', 'not_banana', filename)
        counter = 1
        while os.path.exists(dest):
            name, ext = os.path.splitext(filename)
            dest = os.path.join(BINARY_DATA_DIR, 'train', 'not_banana', f"{name}_{counter}{ext}")
            counter += 1
        shutil.copy2(img_path, dest)
    
    for img_path in garbage_val:
        filename = os.path.basename(img_path)
        dest = os.path.join(BINARY_DATA_DIR, 'val', 'not_banana', filename)
        counter = 1
        while os.path.exists(dest):
            name, ext = os.path.splitext(filename)
            dest = os.path.join(BINARY_DATA_DIR, 'val', 'not_banana', f"{name}_{counter}{ext}")
            counter += 1
        shutil.copy2(img_path, dest)
    
    print(f"   ✓ Copied {len(garbage_train)} training images")
    print(f"   ✓ Copied {len(garbage_val)} validation images")
    
    # Summary
    print("\n" + "=" * 60)
    print("Dataset Summary:")
    print("=" * 60)
    train_banana = len(os.listdir(os.path.join(BINARY_DATA_DIR, 'train', 'banana')))
    train_not_banana = len(os.listdir(os.path.join(BINARY_DATA_DIR, 'train', 'not_banana')))
    val_banana = len(os.listdir(os.path.join(BINARY_DATA_DIR, 'val', 'banana')))
    val_not_banana = len(os.listdir(os.path.join(BINARY_DATA_DIR, 'val', 'not_banana')))
    
    print(f"\nTraining set:")
    print(f"  Banana:     {train_banana:6d} images")
    print(f"  Not Banana: {train_not_banana:6d} images")
    print(f"  Total:      {train_banana + train_not_banana:6d} images")
    
    print(f"\nValidation set:")
    print(f"  Banana:     {val_banana:6d} images")
    print(f"  Not Banana: {val_not_banana:6d} images")
    print(f"  Total:      {val_banana + val_not_banana:6d} images")
    
    print(f"\n✓ Dataset prepared at: {BINARY_DATA_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    prepare_binary_dataset()

