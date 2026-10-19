# LAB-08: Deep Convolutional Neural Network (DCNN) for Cat vs Dog Classification

## Course

**Machine Learning (04-624-201)**

Faculty of Engineering

Rajamangala University of Technology Thanyaburi

---

## Project Overview

This project implements a Deep Convolutional Neural Network (DCNN) to classify images of cats and dogs. The objective is to compare different DCNN architectures and training epochs to evaluate their classification performance.

Three different DCNN configurations are developed and trained using three epoch settings (10, 20, and 30 epochs). The performance of each model is evaluated using classification accuracy.

---

## Objectives

- Apply Deep Convolutional Neural Networks (DCNN) for image classification.
- Compare different DCNN architectures.
- Compare different numbers of training epochs.
- Evaluate model performance using accuracy.
- Predict new images using the trained model.

---

## Dataset

**Dataset:** Cats vs Dogs

The dataset contains two image classes:

- Cat
- Dog

Directory structure:

```text
images/
├── Cat/
└── Dog/
```

Dataset Source:

https://www.kaggle.com/datasets/salader/dogs-vs-cats

---

## Project Structure

```text
LAB-08/
│
├── images/
│   ├── Cat/
│   └── Dog/
│
├── outputs/
│
├── data_loader.py
├── model.py
├── train.py
├── evaluate.py
├── predict.py
├── main.py
├── requirements.txt
└── README.md
```

---

## DCNN Configurations

### Configuration 1