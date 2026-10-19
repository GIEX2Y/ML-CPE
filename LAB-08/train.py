import os
import matplotlib.pyplot as plt

from model import build_model


def train_model(train_generator,
                validation_generator,
                config,
                epochs):

    model = build_model(config)

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs
    )

    os.makedirs("outputs", exist_ok=True)

    # Save model
    model.save(f"outputs/Config{config}_{epochs}epoch.keras")

    # Accuracy Graph
    plt.figure(figsize=(8,5))
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title(f"Accuracy Config {config} ({epochs} Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"outputs/Accuracy_Config{config}_{epochs}.png")
    plt.close()

    # Loss Graph
    plt.figure(figsize=(8,5))
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title(f"Loss Config {config} ({epochs} Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"outputs/Loss_Config{config}_{epochs}.png")
    plt.close()

    return model, history