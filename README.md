# Neural Network Classification Assignment

## Description
This repository contains implementations of fully connected neural networks for binary and multi-class classification tasks. The assignment explores different network architectures, compares performance against traditional machine learning models, and analyzes overfitting behavior.

## Datasets
- **Diabetes/Cancer Dataset**: Binary classification dataset for medical diagnosis
- **CIFAR-10**: Multi-class image classification dataset with 10 classes (50,000 training images, 10,000 test images)

## Setup

### Prerequisites
- Python 3.7+
- Google Colab (recommended) or local Python environment

### Required Libraries
```bash
pip install tensorflow
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn
```

### Google Colab Setup
1. Mount Google Drive to access the dataset:
```python
from google.colab import drive
drive.mount('/content/drive')
```
2. Upload `cancer.csv` to your Google Drive
3. Update the file path in the code to match your Drive location

## Problems

### Problem 1: Neural Network for Diabetes Dataset
**File**: `problem1_diabetes_nn.py`

**Description**: 
Builds a fully connected neural network with 4 hidden layers (64, 32, 16, 8 neurons) for diabetes classification. Compares performance against Logistic Regression and Support Vector Machine models.

**Architecture**:
- Input Layer: 8 features
- Hidden Layers: 64 → 32 → 16 → 8 neurons (ReLU activation)
- Dropout: 0.3, 0.3, 0.2 after first three hidden layers
- Output Layer: 1 neuron (Sigmoid activation)
- Train/Validation Split: 80/20

**Expected Output**:
- Training and validation loss/accuracy plots
- Model performance metrics (Accuracy, Precision, Recall, F1 Score)
- Comparison bar chart between Neural Network, Logistic Regression, and SVM
- Typical accuracy: ~72-75%

### Problem 2: Neural Network for Cancer Dataset
**File**: `problem2_cancer_nn.py`

**Description**: 
Builds a deeper fully connected neural network with 5 hidden layers (128, 64, 32, 16, 8 neurons) for cancer classification. Compares performance against baseline models.

**Architecture**:
- Input Layer: 8 features
- Hidden Layers: 128 → 64 → 32 → 16 → 8 neurons (ReLU activation)
- Dropout: 0.4, 0.3, 0.3, 0.2 after first four hidden layers
- Output Layer: 1 neuron (Sigmoid activation)
- Train/Validation Split: 80/20

**Expected Output**:
- Training and validation loss/accuracy plots
- Model performance metrics (Accuracy, Precision, Recall, F1 Score)
- Comparison bar chart between all three models
- Typical accuracy: ~75-76%
- Higher recall than SVM, making it suitable for medical diagnosis

### Problem 3a: Baseline CIFAR-10 Neural Network
**File**: `problem3a_cifar10_baseline.py`

**Description**: 
Implements a simple fully connected neural network with ONE hidden layer of 512 neurons for CIFAR-10 classification (10 classes).

**Architecture**:
- Input Layer: 3072 features (32×32×3 flattened images)
- Hidden Layer: 512 neurons (ReLU activation)
- Output Layer: 10 neurons (Softmax activation)
- Train/Test Split: 50,000 training / 10,000 test samples

**Expected Output**:
- Epoch-by-epoch training results (50 epochs)
- Training time: ~12-15 minutes
- Training and validation loss/accuracy plots
- Final test accuracy: ~50-51%
- Training summary with best accuracies

**Key Metrics**:
- Total training time
- Loss and accuracy after each epoch
- Final test accuracy: 0.5068
- Best validation accuracy: 0.5145 at epoch 38

### Problem 3b: Extended CIFAR-10 Neural Network
**File**: `problem3b_cifar10_extended.py`

**Description**: 
Extends the baseline network with TWO additional hidden layers, creating a 3-hidden-layer architecture. Trains for 300 epochs to analyze overfitting behavior.

**Architecture**:
- Input Layer: 3072 features
- Hidden Layers: 512 → 256 → 128 neurons (ReLU activation)
- Output Layer: 10 neurons (Softmax activation)
- Train/Test Split: 50,000 training / 10,000 test samples

**Expected Output**:
- Epoch-by-epoch training results at key intervals (1, 10, 50, 100, 150, 200, 250, 300)
- Training time: ~60-80 minutes
- Training and validation loss/accuracy plots over 300 epochs
- Final test accuracy: ~52-55%
- Overfitting analysis showing training-validation gap
- Model size comparison with baseline

**Key Metrics**:
- Total parameters comparison (baseline vs deeper model)
- Training-validation accuracy gap
- Loss gap analysis
- Performance improvement over baseline

## Usage

### Running Problem 1 & 2 (Diabetes/Cancer Classification)
```python
# In Google Colab
python problem1_diabetes_nn.py
python problem2_cancer_nn.py
```

### Running Problem 3 (CIFAR-10 Classification)
```python
# Baseline model (50 epochs)
python problem3a_cifar10_baseline.py

# Extended model (300 epochs)
python problem3b_cifar10_extended.py
```

## Results Summary

### Binary Classification (Problems 1 & 2)
| Model | Dataset | Accuracy | Precision | Recall | F1 Score |
|-------|---------|----------|-----------|--------|----------|
| Neural Network | Diabetes | 0.7208 | 0.6038 | 0.5926 | 0.5981 |
| Neural Network | Cancer | 0.7532 | 0.6333 | 0.7037 | 0.6667 |
| SVM | Diabetes | 0.7532 | 0.6600 | 0.6111 | 0.6346 |
| SVM | Cancer | 0.7532 | 0.6600 | 0.6111 | 0.6346 |

### CIFAR-10 Classification (Problem 3)
| Model | Hidden Layers | Parameters | Epochs | Test Accuracy | Training Time |
|-------|---------------|------------|--------|---------------|---------------|
| Baseline | 1 (512) | ~1.6M | 50 | 0.5068 | ~12 min |
| Extended | 3 (512→256→128) | ~1.9M | 300 | 0.52-0.55 | ~70 min |

## Analysis

### Overfitting Observations
- **Problem 3a**: Minimal overfitting with 1 hidden layer after 50 epochs
- **Problem 3b**: Significant overfitting observed with 3 hidden layers over 300 epochs
  - Training accuracy continues increasing while validation accuracy plateaus
  - Training-validation gap increases substantially after epoch 100
  - Suggests need for regularization techniques (dropout, L2 regularization, early stopping)

### Model Comparison
- Deeper models (3 layers) have ~20% more parameters than baseline
- Performance improvement is marginal (~2-4% accuracy gain)
- Training time increases 5-6x for extended model
- Fully connected networks are limited for image classification compared to CNNs
