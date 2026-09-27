# train.py

import os
import csv
import matplotlib.pyplot as plt


def train_model(
    model,
    train_generator,
    validation_generator,
    epochs=10,
    config_name="Config1"
):
    """
    Train CNN model
    """

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        verbose=1
    )

    os.makedirs("outputs", exist_ok=True)

    # Save Model
    model.save(f"outputs/{config_name}_{epochs}epoch.keras")

    # Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{config_name} Accuracy ({epochs} Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"outputs/{config_name}_{epochs}_accuracy.png")
    plt.close()

    # Loss
    plt.figure(figsize=(8,5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{config_name} Loss ({epochs} Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"outputs/{config_name}_{epochs}_loss.png")
    plt.close()

    return history


def save_result(config_name, epochs, accuracy):
    """
    Save comparison result
    """

    os.makedirs("outputs", exist_ok=True)

    file_path = "outputs/comparison.csv"

    write_header = not os.path.exists(file_path)

    with open(file_path, "a", newline="") as file:

        writer = csv.writer(file)

        if write_header:
            writer.writerow([
                "Configuration",
                "Epochs",
                "Validation Accuracy"
            ])

        writer.writerow([
            config_name,
            epochs,
            round(accuracy, 4)
        ])