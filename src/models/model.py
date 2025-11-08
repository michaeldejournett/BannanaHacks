"""
Basic CNN model template for banana ripeness classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BananaRipenessModel(nn.Module):
    """
    A simple CNN model for classifying banana ripeness.
    
    This is a template that can be modified based on your specific needs.
    """
    
    def __init__(self, num_classes=5):
        """
        Initialize the model.
        
        Args:
            num_classes (int): Number of ripeness classes (e.g., 5 stages from unripe to overripe)
        """
        super(BananaRipenessModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # Note: Input size depends on your image dimensions
        # This assumes 224x224 input images
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Convolutional blocks with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
