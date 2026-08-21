import os

import joblib

from data_load import load_data
from preprocess import standardize_data
from split_data import split_dataset
from svm_model import train_svm, predict_svm
from evaluate import evaluate_model

DATA_PATH = "data.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
X, y = load_data(DATA_PATH)

# Split dataset
X_train, X_test, y_train, y_test = split_dataset(
    X,
    y,
    test_size=0.2
)

# Standardize data
X_train, X_test, scaler = standardize_data(
    X_train,
    X_test
)

kernels = [
    "linear",
    "poly",
    "rbf"
]

scores = {}

for kernel in kernels:

    print(f"\nTraining SVM ({kernel})...\n")

    model = train_svm(
        X_train,
        y_train,
        kernel=kernel
    )

    prediction = predict_svm(
        model,
        X_test
    )

    accuracy = evaluate_model(
        y_test,
        prediction,
        kernel,
        OUTPUT_DIR
    )

    scores[kernel] = accuracy

    joblib.dump(
        model,
        os.path.join(
            OUTPUT_DIR,
            f"{kernel}_svm.pkl"
        )
    )

print("\n==============================")
print("Accuracy Comparison")
print("==============================")

for kernel, score in scores.items():
    print(f"{kernel:10s}: {score:.4f}")

best = max(scores, key=scores.get)

print("\nBest Kernel :", best)
print("Accuracy    :", round(scores[best], 4))