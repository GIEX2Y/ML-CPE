# LAB-07: Convolutional Neural Network for Cat and Dog Classification

## Course Information

* **Course Code:** 04-624-201
* **Course:** Machine Learning
* **Laboratory:** LAB-07 Convolutional Neural Network (CNN)

---

## Objective

The objective of this laboratory is to apply a Convolutional Neural Network (CNN) to classify cat and dog images. The project compares different CNN architectures and training epochs to evaluate their classification performance.

---

## Dataset

This project uses a custom image dataset stored locally.

Dataset Structure

```
images/
├── Cat/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
└── Dog/
    ├── dog1.jpg
    ├── dog2.jpg
    └── ...
```

The dataset is automatically divided into:

* Training Set (80%)
* Validation Set (20%)

---

## Project Structure

```
LAB-07/
│
├── images/
│   ├── Cat/
│   └── Dog/
│
├── outputs/
│   ├── accuracy/
│   ├── loss/
│   ├── confusion_matrix/
│   ├── models/
│   └── comparison.csv
│
├── preprocess.py
├── cnn_model.py
├── train.py
├── evaluate.py
├── predict.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Features

* Image preprocessing
* Data normalization
* Data augmentation
* Three CNN configurations
* Comparison using different epochs
* Accuracy and Loss visualization
* Confusion Matrix
* Classification Report
* Prediction on new images
* Automatic model saving

---

## CNN Configurations

### Configuration 1

* 2 Convolution Layers
* MaxPooling
* Dense Layer

### Configuration 2

* 3 Convolution Layers
* MaxPooling
* Dense Layer

### Configuration 3

* Batch Normalization
* Dropout
* Deeper CNN Architecture

---

## Training Experiments

The project evaluates the following settings.

| Configuration | Epochs |
| ------------- | ------ |
| Config 1      | 10     |
| Config 1      | 20     |
| Config 1      | 30     |
| Config 2      | 10     |
| Config 2      | 20     |
| Config 2      | 30     |
| Config 3      | 10     |
| Config 3      | 20     |
| Config 3      | 30     |

---

## Requirements

Install all required libraries.

```bash
pip install -r requirements.txt
```

---

## Run Project

```bash
python main.py
```

---

## Output

The program automatically generates:

* Trained CNN models (.keras)
* Accuracy graphs
* Loss graphs
* Confusion Matrix
* Comparison results (CSV)

Example:

```
outputs/
├── accuracy/
├── loss/
├── confusion_matrix/
├── models/
└── comparison.csv
```

---

## Prediction

Predict a new image.

```bash
python predict.py
```

Example Output

```
Prediction : Cat
Confidence : 99.42%
```

or

```
Prediction : Dog
Confidence : 98.75%
```

---

## Libraries

* TensorFlow
* NumPy
* Matplotlib
* Scikit-learn
* Pillow

---