from neural_network import (
    load_dataset,
    prepare_data,
    train_model,
    evaluate_model,
    compare_epochs,
    compare_configurations
)

from visualization import (
    plot_loss,
    plot_epoch_results,
    plot_configuration_results
)


def main():

    print("=" * 50)
    print("        ML LAB 06 - NEURAL NETWORK")
    print("=" * 50)

    # ==========================================
    # 1. Load Dataset
    # ==========================================

    X, y, data = load_dataset()

    print("\nDataset Information")
    print("-" * 30)

    print("Number of samples :", X.shape[0])
    print("Number of features:", X.shape[1])
    print("Classes           :", data.target_names)

    # ==========================================
    # 2. Prepare Data
    # ==========================================

    X_train, X_test, y_train, y_test = prepare_data(
        X,
        y
    )

    print("\nData Split")
    print("-" * 30)

    print("Training data:", X_train.shape)
    print("Testing data :", X_test.shape)

    # ==========================================
    # 3. Train Main Model
    # ==========================================

    print("\nTraining Neural Network...")
    print("-" * 30)

    model = train_model(
        X_train,
        y_train,
        hidden_layers=(50, 25),
        epochs=50
    )

    train_acc, test_acc = evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("Train Accuracy:",
          round(train_acc * 100, 2), "%")

    print("Test Accuracy :",
          round(test_acc * 100, 2), "%")

    # ==========================================
    # 4. Prediction
    # ==========================================

    print("\nPrediction")
    print("-" * 30)

    predictions = model.predict(X_test)

    for i in range(10):

        actual = data.target_names[y_test[i]]
        predicted = data.target_names[predictions[i]]

        print(
            f"{i + 1}. Actual: {actual:10s}"
            f" | Predicted: {predicted}"
        )

    # ==========================================
    # 5. Loss Graph
    # ==========================================

    plot_loss(model)

    # ==========================================
    # 6. Compare Epochs
    # ==========================================

    print("\nEpoch Comparison")
    print("-" * 50)

    epoch_results = compare_epochs(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print(
        f"{'Epochs':<10}"
        f"{'Train Accuracy':<20}"
        f"{'Test Accuracy':<20}"
    )

    for result in epoch_results:

        print(
            f"{result['Epochs']:<10}"
            f"{result['Train Accuracy']:.2f}%"
            f"{'':<10}"
            f"{result['Test Accuracy']:.2f}%"
        )

    # Graph Accuracy vs Epochs
    plot_epoch_results(
        epoch_results
    )

    # ==========================================
    # 7. Compare Configurations
    # ==========================================

    print("\nNeural Network Configuration Comparison")
    print("-" * 60)

    configuration_results = compare_configurations(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print(
        f"{'Configuration':<20}"
        f"{'Train Accuracy':<20}"
        f"{'Test Accuracy':<20}"
    )

    for result in configuration_results:

        print(
            f"{result['Configuration']:<20}"
            f"{result['Train Accuracy']:.2f}%"
            f"{'':<10}"
            f"{result['Test Accuracy']:.2f}%"
        )

    # Graph Configuration Comparison
    plot_configuration_results(
        configuration_results
    )

    print("\n" + "=" * 50)
    print("             LAB 06 COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()