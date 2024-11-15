# Breast Cancer Classification with Neural Network

This project implements a neural network model in Python to classify breast cancer cases as malignant or benign based on medical imaging data features. Using deep learning techniques, this project demonstrates an end-to-end workflow from data preprocessing to model training, evaluation, and analysis.

## Table of Contents

- [Breast Cancer Classification with Neural Network](#breast-cancer-classification-with-neural-network)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [Methods](#methods)
    - [1. Data Preprocessing](#1-data-preprocessing)
    - [2. Model Architecture](#2-model-architecture)
    - [3. Model Training](#3-model-training)
    - [4. Model Evaluation](#4-model-evaluation)
    - [5. Visualization and Interpretation](#5-visualization-and-interpretation)
  - [Installation and Usage](#installation-and-usage)
    - [Prerequisites](#prerequisites)

## Project Overview

This repository aims to provide a comprehensive deep learning pipeline for breast cancer classification using neural networks. The model developed here is capable of predicting the likelihood of a tumor being malignant or benign, helping to support diagnostic processes in clinical settings.

## Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) Data Set**, which includes 569 samples with 30 features. These features are computed from digital images of fine needle aspirate (FNA) biopsies and describe characteristics like radius, texture, perimeter, area, and smoothness. The target variable classifies tumors as **benign (0)** or **malignant (1)**.

## Methods

### 1. Data Preprocessing

- **Data Loading and Cleaning**: Load the dataset and handle missing values or anomalies if present.
- **Feature Selection**: Review feature correlations and select relevant ones to enhance model performance.
- **Data Normalization**: Normalize the dataset using MinMaxScaler to ensure efficient model convergence.
- **Train-Test Split**: Split the dataset into training and testing sets (e.g., 80% training, 20% testing) to evaluate model performance.

### 2. Model Architecture

The neural network architecture for this project is designed as a multi-layer perceptron (MLP) with the following components:

- **Input Layer**: Corresponds to the 30 input features.
- **Hidden Layers**: Two dense layers with ReLU activation for nonlinear transformation of features.
- **Output Layer**: A single neuron with sigmoid activation for binary classification (malignant vs. benign).

**Hyperparameters**:
- Optimizer: Adam
- Loss function: Binary cross-entropy
- Metrics: Accuracy

### 3. Model Training

- **Batch Processing**: Use mini-batches for efficient training and better generalization.
- **Early Stopping**: Monitor the validation loss to prevent overfitting by stopping training when the loss does not improve over a certain number of epochs.
- **Epochs**: Train for a maximum of 50 epochs with early stopping to monitor performance on the validation set.

### 4. Model Evaluation

Evaluation metrics include:
- **Accuracy**: Overall percentage of correct predictions.
- **Precision and Recall**: Measure of correctness for malignant predictions (important for healthcare applications).
- **F1-Score**: Harmonic mean of precision and recall for balanced performance assessment.
- **Confusion Matrix**: Visual representation of true positives, false positives, true negatives, and false negatives.

### 5. Visualization and Interpretation

Visualize the model’s performance and data distribution through:
- **Training/Validation Curves**: Plot loss and accuracy over epochs to assess training dynamics.
- **Confusion Matrix**: Highlight the classification accuracy for benign and malignant cases.
- **ROC Curve and AUC Score**: Analyze model discrimination power between classes.

## Installation and Usage

### Prerequisites

Ensure Python 3.8+ and the following libraries are installed:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`

Install dependencies using:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```