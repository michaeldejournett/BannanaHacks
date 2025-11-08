"""
Basic tests for the BannanaHacks project.
"""

import unittest
import torch
from src.models.model import BananaRipenessModel


class TestBananaRipenessModel(unittest.TestCase):
    """Test cases for the BananaRipenessModel."""
    
    def test_model_initialization(self):
        """Test that the model can be initialized."""
        num_classes = 5
        model = BananaRipenessModel(num_classes=num_classes)
        self.assertIsInstance(model, torch.nn.Module)
    
    def test_model_forward_pass(self):
        """Test that the model can perform a forward pass."""
        num_classes = 5
        batch_size = 4
        model = BananaRipenessModel(num_classes=num_classes)
        
        # Create dummy input (batch_size, channels, height, width)
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        
        # Forward pass
        output = model(dummy_input)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, num_classes))
    
    def test_model_different_num_classes(self):
        """Test model with different number of classes."""
        for num_classes in [3, 5, 7, 10]:
            model = BananaRipenessModel(num_classes=num_classes)
            dummy_input = torch.randn(2, 3, 224, 224)
            output = model(dummy_input)
            self.assertEqual(output.shape[1], num_classes)


if __name__ == '__main__':
    unittest.main()
