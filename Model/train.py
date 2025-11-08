"""
Training script template for banana ripeness model.
"""

import os
# Set AMD GPU environment variables BEFORE importing torch
os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '10.3.0')
# Additional ROCm environment variables for stability
os.environ.setdefault('HSA_ENABLE_SDMA', '0')
os.environ.setdefault('HSA_ENABLE_INTERRUPT', '0')
os.environ.setdefault('HIP_FORCE_DEV_KERNARG', '1')

import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.model import BananaRipenessModel
from src.data.dataset import get_data_loaders


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda or cpu)
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use enumerate instead of tqdm for better GPU compatibility
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        try:
            # Move to device with non_blocking for better GPU utilization
            # pin_memory=True in DataLoader makes this faster
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                acc = 100. * correct / total
                print(f'  Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {avg_loss:.4f}, Acc: {acc:.2f}%')
            
            # Clear cache periodically for AMD GPU
            if device.type == 'cuda' and (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            print(f"Error at batch {batch_idx}: {e}")
            if "out of memory" in str(e):
                print("GPU out of memory, clearing cache...")
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
    """
    Validate the model.
    
    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on (cuda or cpu)
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / (progress_bar.n + 1),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Banana Ripeness Model')
    parser.add_argument('--train-dir', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--val-dir', type=str, required=True,
                       help='Path to validation data directory')
    parser.add_argument('--num-classes', type=int, default=5,
                       help='Number of ripeness classes')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (sweet spot for GPU utilization vs speed)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (4 is usually optimal)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Use all available GPUs (DataParallel)')
    parser.add_argument('--gpu-ids', type=str, default=None,
                       help='Comma-separated GPU IDs to use (e.g., "0,1"). Default: all GPUs if --multi-gpu')
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device - prefer discrete GPU (GPU 0) unless forced to CPU or multi-GPU
    if args.force_cpu:
        device = torch.device('cpu')
        print("Forced to use CPU (--force-cpu flag)")
    elif torch.cuda.is_available():
        try:
            # Test GPU with a simple operation
            test_tensor = torch.rand(10, device='cuda:0')
            del test_tensor
            torch.cuda.empty_cache()
            
            # Check if multi-GPU is requested
            if args.multi_gpu and torch.cuda.device_count() > 1:
                # Multi-GPU setup - don't restrict HIP_VISIBLE_DEVICES
                if args.gpu_ids:
                    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
                    # Set HIP_VISIBLE_DEVICES to the requested GPUs
                    os.environ['HIP_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
                else:
                    gpu_ids = list(range(torch.cuda.device_count()))
                    # Use all GPUs - don't restrict
                    if 'HIP_VISIBLE_DEVICES' in os.environ:
                        del os.environ['HIP_VISIBLE_DEVICES']
                
                print(f"Multi-GPU mode: Using GPUs {gpu_ids}")
                for gpu_id in gpu_ids:
                    print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                device = torch.device(f'cuda:{gpu_ids[0]}')  # Primary device
            else:
                # Single GPU - use GPU 0 (discrete GPU)
                os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')
                device = torch.device('cuda:0')
                torch.cuda.set_device(0)
                print(f"GPU detected! Using device: {device}")
                print(f"GPU name: {torch.cuda.get_device_name(0)}")
                print(f"GPU count: {torch.cuda.device_count()}")
                print(f"Selected GPU 0 (discrete GPU) for training")
        except Exception as e:
            print(f"GPU test failed: {e}")
            print("Falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print(f"No GPU detected. Using CPU: {device}")
        print("Note: For AMD GPU support, install ROCm and PyTorch with ROCm support")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(
        args.train_dir,
        args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Initialize model
    print("Initializing model...")
    
    # Disable some optimizations that might cause issues with ROCm
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    
    model = BananaRipenessModel(num_classes=args.num_classes).to(device)
    
    # Wrap model for multi-GPU if requested
    use_multi_gpu = args.multi_gpu and torch.cuda.device_count() > 1 and device.type == 'cuda'
    if use_multi_gpu:
        if args.gpu_ids:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
        
        print(f"\nWrapping model for multi-GPU training on GPUs: {gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        print("✓ Model wrapped with DataParallel")
    
    # Test model with a dummy input first
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
            model = model.module if hasattr(model, 'module') else model
            model = model.cpu()
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5
    )
    
    # Training loop
    best_val_acc = 0.0
    
    print(f"\nStarting training for {args.epochs} epochs...")
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
                f'best_model.pth'
            )
            # Save model state dict (unwrap DataParallel if needed)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
