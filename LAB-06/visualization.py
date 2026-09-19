import matplotlib.pyplot as plt


def plot_loss(model):

    plt.figure(figsize=(8, 5))

    plt.plot(
        model.loss_curve_,
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.grid()
    plt.show()


def plot_epoch_results(results):

    epochs = [
        result["Epochs"]
        for result in results
    ]

    train_accuracy = [
        result["Train Accuracy"]
        for result in results
    ]

    test_accuracy = [
        result["Test Accuracy"]
        for result in results
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_accuracy,
        marker="o",
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        test_accuracy,
        marker="o",
        label="Test Accuracy"
    )

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Epochs")

    plt.legend()
    plt.grid()

    plt.show()


def plot_configuration_results(results):

    names = [
        result["Configuration"]
        for result in results
    ]

    train_accuracy = [
        result["Train Accuracy"]
        for result in results
    ]

    test_accuracy = [
        result["Test Accuracy"]
        for result in results
    ]

    x = range(len(names))

    plt.figure(figsize=(9, 5))

    plt.bar(
        [i - 0.2 for i in x],
        train_accuracy,
        width=0.4,
        label="Train Accuracy"
    )

    plt.bar(
        [i + 0.2 for i in x],
        test_accuracy,
        width=0.4,
        label="Test Accuracy"
    )

    plt.xticks(
        list(x),
        names
    )

    plt.xlabel("Neural Network Configuration")
    plt.ylabel("Accuracy (%)")

    plt.title(
        "Comparison of Neural Network Configurations"
    )

    plt.legend()
    plt.grid(axis="y")

    plt.show()