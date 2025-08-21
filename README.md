# Real-or-Spam SMS Detection 📱🛡️

An AI-powered spam detection system developed for Turkish SMS messages. This project uses a BERT-based deep learning model to classify SMS messages as "ham" (normal) or "spam" (unwanted).

## 🎯 Project Overview

This project is a machine learning application developed to automatically classify Turkish SMS messages. It has been fine-tuned using the `dbmdz/bert-base-turkish-uncased` model and achieved high accuracy rates.

## 🚀 Features

- **Turkish Language Support**: BERT model optimized for Turkish texts
- **High Accuracy**: 98%+ accuracy rate
- **Real-time Prediction**: Instant classification for new messages
- **Visualization**: Detailed analysis and charts of the training process
- **Model Saving**: Automatic saving of the best performing model

## 📊 Model Performansı

### Test Results (Best Model - Epoch 1)

| Metric | Ham | Spam | Overall |
|--------|-----|------|-------|
| **Precision** | 0.98 | 0.97 | 0.98 |
| **Recall** | 0.97 | 0.98 | 0.98 |
| **F1-Score** | 0.98 | 0.98 | 0.98 |
| **Support** | 153 | 150 | 303 |

### Validation Results

| Metric | Ham | Spam | Overall |
|--------|-----|------|-------|
| **Precision** | 0.99 | 0.99 | 0.99 |
| **Recall** | 0.99 | 0.99 | 0.99 |
| **F1-Score** | 0.99 | 0.99 | 0.99 |
| **Support** | 152 | 151 | 303 |

## 📈 Training Charts

### Training and Validation Results
![Training Results](https://raw.githubusercontent.com/ByErenOzer/Real-or-Spam-SMS-Detection/main/training_results.png)

### Test Classification Report
![Test Classification Report](https://raw.githubusercontent.com/ByErenOzer/Real-or-Spam-SMS-Detection/main/test_classification_report.png)

### Validation Classification Report
![Validation Classification Report](https://raw.githubusercontent.com/ByErenOzer/Real-or-Spam-SMS-Detection/main/validation_classification_report.png)

## 🛠️ Installation

### Requirements

```bash
pip install pandas numpy scikit-learn torch transformers tqdm seaborn matplotlib openpyxl
```

### Required Libraries

- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning metrics
- **torch**: PyTorch deep learning framework
- **transformers**: Hugging Face transformers library
- **tqdm**: Progress bar
- **seaborn & matplotlib**: Data visualization
- **openpyxl**: Excel file reading

## 📁 Project Structure

```
Real-or-Spam-SMS-Detection/
├── sms_spam_detection.py          # Main training script
├── karistirilmis_sms_dataset.xlsx  # Training dataset
├── sms_spam_detection/             # Model outputs
│   ├── best_sms_model/            # Best model
│   ├── best_sms_tokenizer/        # Tokenizer
│   ├── training_history.json      # Training history
│   └── training_results.png       # Result charts
└── README.md                      # This file

## 🔧 Model Details

### Used Model
- **Base Model**: `dbmdz/bert-base-turkish-uncased`
- **Task**: Binary Classification (Ham vs Spam)
- **Max Length**: 128 tokens
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Epochs**: 2
- **Optimizer**: AdamW

### Data Preprocessing
- Converting to lowercase
- Cleaning extra spaces
- Filtering empty messages
- Label encoding (ham: 0, spam: 1)

### Data Split
- **Training**: 70%
- **Validation**: 15%
- **Test**: 15%

## 🎯 Results

- **Test Accuracy**: 98%
- **F1-Score**: 0.98
- **Precision**: 0.98
- **Recall**: 0.98
