#!/usr/bin/env python3
"""
Simple script to predict if an image is a banana or not.
Run from project root: python predict_binary_banana.py <image_path>
"""

import os
import sys

# Add Model directory to path
model_dir = os.path.join(os.path.dirname(__file__), 'Model')
sys.path.insert(0, model_dir)

# Change to Model directory for relative imports
os.chdir(model_dir)

# Now import and run
from predict_binary import main

if __name__ == '__main__':
    main()

