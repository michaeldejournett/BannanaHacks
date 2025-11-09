#!/usr/bin/env python3
"""
Simple script to predict banana ripeness from an image.
Run from project root: python Scripts/predict_banana.py <image_path>
Or from Scripts folder: python predict_banana.py <image_path>
"""

import os
import sys

# Get the script's directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from Scripts to BannanaHacks
model_dir = os.path.join(project_root, 'Model')

# Add Model directory to path
sys.path.insert(0, model_dir)

# Change to Model directory for relative imports
os.chdir(model_dir)

# Now import and run
from predict import main

if __name__ == '__main__':
    main()
