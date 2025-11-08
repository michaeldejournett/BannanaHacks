"""
Dataset class for loading banana images.
"""

import os
from typing import Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class BananaDataset(Dataset):
    """
    Dataset class for banana ripeness images.
    
    Expected directory structure:
    data/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            image2.jpg
            ...
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Path to the data directory
            transform (callable, optional): Optional transform to be applied on images
            target_size (tuple): Target size for resizing images (height, width)
        """
        self.data_dir = data_dir
        self.target_size = target_size
        
        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        self.class_names = []
        
        # This will be populated when you add your data
        # For now, it's a template
        
    def __len__(self) -> int:
        """Return the total number of images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an image and its label.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            tuple: (image, label) where image is a tensor and label is an integer
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create data loaders for training and validation.
    
    Args:
        train_dir (str): Path to training data directory
        val_dir (str): Path to validation data directory
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Define transforms for training (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Define transforms for validation (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = BananaDataset(train_dir, transform=train_transform)
    val_dataset = BananaDataset(val_dir, transform=val_transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader
