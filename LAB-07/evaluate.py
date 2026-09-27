# evaluate.py

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


def evaluate_model(
    model,
    validation_generator,
    config_name="Config1",
    epochs=10
):
    """
    Evaluate CNN model
    """

    # Evaluate
    loss, accuracy = model.evaluate(
        validation_generator,
        verbose=0
    )

    print("=" * 50)
    print(f"Configuration : {config_name}")
    print(f"Epochs        : {epochs}")
    print(f"Loss          : {loss:.4f}")
    print(f"Accuracy      : {accuracy:.4f}")
    print("=" * 50)

    # Reset Generator
    validation_generator.reset()

    # Prediction
    predictions = model.predict(
        validation_generator,
        verbose=0
    )

    predicted_class = (predictions > 0.5).astype(int)

    true_class = validation_generator.classes

    # Classification Report
    print("\nClassification Report\n")

    print(
        classification_report(
            true_class,
            predicted_class,
            target_names=["Cat", "Dog"]
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(
        true_class,
        predicted_class
    )

    os.makedirs("outputs", exist_ok=True)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Cat", "Dog"]
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    disp.plot(ax=ax)

    plt.title(
        f"{config_name} ({epochs} Epochs)"
    )

    plt.savefig(
        f"outputs/{config_name}_{epochs}_confusion_matrix.png"
    )

    plt.close()

    return accuracy