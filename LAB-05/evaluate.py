import json
import os

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_true, y_pred, kernel, output_dir="outputs"):
    """
    Evaluate SVM model
    """

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    print("=" * 50)
    print(f"SVM Kernel : {kernel}")
    print(f"Accuracy  : {accuracy:.4f}")
    print("=" * 50)
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix")
    print(matrix)

    os.makedirs(output_dir, exist_ok=True)

    results = {
        "kernel": kernel,
        "accuracy": accuracy,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report
    }

    with open(
        os.path.join(output_dir, f"{kernel}_metrics.json"),
        "w"
    ) as file:
        json.dump(results, file, indent=4)

    return accuracy