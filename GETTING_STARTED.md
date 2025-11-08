# Getting Started with BannanaHacks

This guide will help you get started with the banana ripeness detection project.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/michaeldejournett/BannanaHacks.git
cd BannanaHacks
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
BannanaHacks/
├── src/
│   ├── models/
│   │   └── model.py          # CNN model for banana ripeness
│   ├── data/
│   │   └── dataset.py        # Dataset class and data loaders
│   └── utils/                # Utility functions (empty template)
├── tests/
│   └── test_model.py         # Unit tests
├── train.py                  # Training script
├── predict.py                # Inference script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Preparing Your Data

Organize your banana images in the following structure:

```
data/
├── train/
│   ├── unripe/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── slightly_ripe/
│   ├── ripe/
│   ├── very_ripe/
│   └── overripe/
└── val/
    ├── unripe/
    ├── slightly_ripe/
    ├── ripe/
    ├── very_ripe/
    └── overripe/
```

Each subdirectory represents a ripeness class.

## Training a Model

Once you have your data organized, you can train the model:

```bash
python train.py \
    --train-dir data/train \
    --val-dir data/val \
    --num-classes 5 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --checkpoint-dir checkpoints
```

### Training Parameters

- `--train-dir`: Path to training data directory
- `--val-dir`: Path to validation data directory
- `--num-classes`: Number of ripeness classes (default: 5)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--checkpoint-dir`: Directory to save model checkpoints (default: checkpoints)

## Making Predictions

After training, you can use the model to predict the ripeness of new banana images:

```bash
python predict.py \
    --image path/to/banana.jpg \
    --checkpoint checkpoints/best_model.pth \
    --num-classes 5
```

### Prediction Parameters

- `--image`: Path to the banana image
- `--checkpoint`: Path to the trained model checkpoint
- `--num-classes`: Number of classes (must match training)

## Running Tests

To verify the installation and setup:

```bash
python -m unittest discover tests -v
```

## Model Architecture

The default model is a simple CNN with three convolutional blocks:
- Conv1: 3 → 32 channels
- Conv2: 32 → 64 channels
- Conv3: 64 → 128 channels
- Fully connected layers for classification

You can modify `src/models/model.py` to experiment with different architectures.

## Data Augmentation

The training pipeline includes data augmentation:
- Random horizontal flip
- Random rotation (±10 degrees)
- Color jitter (brightness, contrast, saturation)
- Normalization using ImageNet statistics

## Next Steps

1. **Collect Data**: Gather banana images at different ripeness stages
2. **Organize Data**: Structure your data as shown above
3. **Train Model**: Run the training script with your data
4. **Evaluate**: Check validation accuracy and adjust hyperparameters
5. **Deploy**: Use the inference script to classify new images

## Customization

### Custom Model

Edit `src/models/model.py` to implement your own architecture:

```python
class MyCustomModel(nn.Module):
    def __init__(self, num_classes=5):
        super(MyCustomModel, self).__init__()
        # Your layers here
        
    def forward(self, x):
        # Your forward pass here
        return x
```

### Custom Dataset

Edit `src/data/dataset.py` to customize data loading:

```python
class BananaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # Your initialization here
        
    def __getitem__(self, idx):
        # Your data loading logic here
        return image, label
```

## Tips

- Start with a smaller learning rate (e.g., 0.0001) if training is unstable
- Increase epochs if the model hasn't converged
- Use data augmentation to improve generalization
- Monitor both training and validation accuracy to detect overfitting
- Save checkpoints regularly to avoid losing progress

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size (`--batch-size`)
- Use a smaller model
- Use CPU instead of GPU

### Poor Accuracy
- Collect more training data
- Increase data augmentation
- Train for more epochs
- Try different learning rates
- Use a pre-trained model (transfer learning)

## License

This project is open source and available for educational purposes.
