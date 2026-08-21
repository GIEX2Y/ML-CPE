# ML-05 - Support Vector Machine (SVM)

Build a Support Vector Machine (SVM) classification pipeline using Python for the Breast Cancer Wisconsin Diagnostic Dataset. This project covers data loading, preprocessing, feature scaling, model training, evaluation, and prediction using multiple SVM kernels.

---

# Data

Breast Cancer Wisconsin (Diagnostic) Dataset

Source:
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

Dataset Information

- Samples: 569
- Features: 30 numerical features
- Target:
  - B = Benign
  - M = Malignant

The dataset is stored as:

```
data.csv
```

---

# Structure

```
ML-05-SVM/
│
├── data.csv
│
├── data_load.py
├── preprocess.py
├── split_data.py
├── svm_model.py
├── evaluate.py
├── main.py
│
├── outputs/
│   ├── linear_svm.pkl
│   ├── poly_svm.pkl
│   ├── rbf_svm.pkl
│   ├── linear_metrics.json
│   ├── poly_metrics.json
│   └── rbf_metrics.json
│
├── requirements.txt
└── README.md
```

---

# Features

- Load Breast Cancer Dataset
- Remove unnecessary columns
- Encode diagnosis labels
- Split dataset into training and testing sets
- Standardize input features using StandardScaler
- Train three SVM models
  - Linear Kernel
  - Polynomial Kernel
  - RBF Kernel
- Evaluate model performance
- Save trained models
- Save evaluation results

---

# Workflow

```
Load Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Train/Test Split
        │
        ▼
Feature Standardization
        │
        ▼
Train SVM
 ├── Linear
 ├── Polynomial
 └── RBF
        │
        ▼
Prediction
        │
        ▼
Evaluation
        │
        ▼
Save Models & Metrics
```

---

# Requirements

Install dependencies

```bash
pip install -r requirements.txt
```

Required packages

```
pandas
numpy
scikit-learn
joblib
```

---

# Run

```bash
python main.py
```

---

# Output

After running the program, the following files will be generated inside the `outputs` folder.

```
outputs/
├── linear_svm.pkl
├── poly_svm.pkl
├── rbf_svm.pkl
├── linear_metrics.json
├── poly_metrics.json
└── rbf_metrics.json
```

The console displays

- Accuracy
- Classification Report
- Confusion Matrix
- Accuracy comparison of all kernels
- Best performing kernel

---

# SVM Kernels

### Linear Kernel

Suitable for linearly separable data.

### Polynomial Kernel

Captures nonlinear relationships by mapping data into higher-dimensional space.

### RBF Kernel

The most commonly used kernel for nonlinear classification and usually provides the highest accuracy.

---

# Evaluation Metrics

Each model is evaluated using

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The evaluation results are saved as JSON files inside the `outputs` directory.

---

# Summary

This project demonstrates how to build a complete Support Vector Machine (SVM) classification pipeline using the Breast Cancer Wisconsin Diagnostic Dataset. The workflow includes data loading, preprocessing, feature scaling, model training, prediction, and evaluation. Three SVM kernels (Linear, Polynomial, and RBF) are compared to determine the best-performing model based on classification accuracy and other evaluation metrics.