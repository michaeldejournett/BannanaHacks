"""
Training script for binary banana classification (banana vs not banana).
"""

import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Set AMD GPU environment variables BEFORE importing model
os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '10.3.0')

from src.models.model import BananaRipenessModel
from src.data.dataset import get_data_loaders


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        try:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                acc = 100. * correct / total
                print(f'  Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {avg_loss:.4f}, Acc: {acc:.2f}%')
            
            if device.type == 'cuda' and (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            print(f"Error at batch {batch_idx}: {e}")
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                continue
            raise
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Binary Banana Classifier')
    parser.add_argument('--train-dir', type=str, 
                       default='../data/binary_classification/train',
                       help='Path to training data directory')
    parser.add_argument('--val-dir', type=str,
                       default='../data/binary_classification/val',
                       help='Path to validation data directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_binary',
                       help='Directory to save model checkpoints')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device
    if args.force_cpu:
        device = torch.device('cpu')
        print("Using CPU (--force-cpu flag)")
    elif torch.cuda.is_available():
        try:
            test_tensor = torch.rand(10, device='cuda:0')
            del test_tensor
            torch.cuda.empty_cache()
            os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            print(f"GPU detected! Using device: {device}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"GPU test failed: {e}")
            print("Falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(
        args.train_dir,
        args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Initialize model (2 classes: banana, not_banana)
    print("Initializing model...")
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    
    model = BananaRipenessModel(num_classes=2).to(device)
    
    # Test model
    if device.type == 'cuda':
        print("Testing model with dummy input...")
        try:
            dummy_input = torch.randn(1, 3, 224, 224, device=device)
            with torch.no_grad():
                _ = model(dummy_input)
            print("✓ Model test passed")
            del dummy_input
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ Model test failed: {e}")
            print("Falling back to CPU...")
            device = torch.device('cpu')
            model = model.cpu()
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5
    )
    
    # Training loop
    best_val_acc = 0.0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        try:
            val_loss, val_acc = validate(
                model, val_loader, criterion, device
            )
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        except Exception as e:
            print(f"Validation failed: {e}")
            val_loss, val_acc = 0.0, 0.0
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                'best_model.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"✓ Saved best model with validation accuracy: {val_acc:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

