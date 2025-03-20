# Deepfake Detection System

## Overview
This project implements a deep learning system for detecting deepfake images using the Xception architecture with transfer learning. The model is trained to classify images as either real or artificially generated (deepfakes).

## Dataset
The system is trained on a dataset containing:
- 140,002 training images
- 10,905 test images
- 39,428 validation images

The dataset is organized with two classes: real and deepfake images.

## Model Architecture
- Base model: Xception (pre-trained on ImageNet)
- Custom top layers for classification:
  - Global Average Pooling
  - Dense layer (32 neurons)
  - Dense layer (16 neurons)
  - Output layer with sigmoid activation for binary classification

## Features
- Transfer learning using Xception architecture
- Fine-tuning of the last 20 layers of the base model
- Data augmentation for training
- Binary classification (real vs fake)
- Evaluation metrics including ROC curve, precision-recall curve, and confusion matrix

## Requirements
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV

## Usage
The entire workflow is documented in the `Deepfake-Detection-System.ipynb` notebook, which includes:
1. Data loading and preprocessing
2. Model architecture definition
3. Model training
4. Evaluation and visualization of results

## License
[Insert License Information]

## Acknowledgements
- [Insert any acknowledgements or credits]
