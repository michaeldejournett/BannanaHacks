# BannanaHacks

A PyTorch-based project for detecting banana ripeness using deep learning.

## Project Structure

```
BannanaHacks/
├── src/
│   ├── models/          # Model architectures
│   ├── data/            # Data loading and preprocessing
│   └── utils/           # Utility functions
├── tests/               # Unit tests
├── data/                # Dataset directory (gitignored)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

This is a blank PyTorch project template. Add your banana ripeness detection model code in the `src/` directory.

## Dependencies

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- NumPy >= 1.24.0
- Pillow >= 10.0.0
- Matplotlib >= 3.7.0
- tqdm >= 4.65.0

## HOW TO RUN

### FastAPI
```console
python -m uvicorn main:app --reload
```
### WebDev
```console
npm run dev
```