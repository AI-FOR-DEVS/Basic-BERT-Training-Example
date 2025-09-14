# ML Text Classification Example

A simple machine learning project that demonstrates text classification using DistilBERT on the AG News dataset.

## Overview

This project trains a DistilBERT model to classify news articles into 4 categories:
- World
- Sports  
- Business
- Science/Technology

## Files

- `basic.py` - Training script that fine-tunes DistilBERT on AG News dataset
- `inference.py` - Simple inference script to test the trained model
- `models/` - Directory containing the trained model files

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install torch transformers datasets scikit-learn
```

## Usage

### Training

Run the training script to fine-tune the model:
```bash
python basic.py
```

This will:
- Load the AG News dataset
- Train DistilBERT for 1 epoch on 5,000 samples
- Save the model to `models/distilbert-base-uncased-ag_news/`

### Inference

Test the trained model:
```bash
python inference.py
```

Or use it programmatically:
```python
from inference import predict
result = predict("Apple stock rises after earnings report")
print(result)  # Should output: Business
```

## Model Details

- **Base Model**: DistilBERT (distilbert-base-cased)
- **Task**: Multi-class text classification
- **Dataset**: AG News (4 categories)
- **Training**: 1 epoch, 5,000 samples
- **Max Sequence Length**: 128 tokens
