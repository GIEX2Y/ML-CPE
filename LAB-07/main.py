# main.py

from preprocess import create_data_generators
from cnn_model import build_cnn_model
from train import train_model, save_result
from evaluate import evaluate_model


def main():

    # ==========================
    # Load Dataset
    # ==========================
    train_generator, validation_generator = create_data_generators(
        data_dir="images",
        img_size=(128, 128),
        batch_size=32
    )

    # ==========================
    # Experiment Settings
    # ==========================
    configs = [1, 2, 3]
    epochs_list = [10, 20, 30]

    # ==========================
    # Run Experiments
    # ==========================
    for config in configs:

        print("\n" + "=" * 60)
        print(f"Training CNN Configuration {config}")
        print("=" * 60)

        for epochs in epochs_list:

            print(f"\nEpochs : {epochs}")

            # Build Model
            model = build_cnn_model(config=config)

            # Train Model
            train_model(
                model=model,
                train_generator=train_generator,
                validation_generator=validation_generator,
                epochs=epochs,
                config_name=f"Config{config}"
            )

            # Evaluate Model
            accuracy = evaluate_model(
                model=model,
                validation_generator=validation_generator,
                config_name=f"Config{config}",
                epochs=epochs
            )

            # Save Result
            save_result(
                config_name=f"Config{config}",
                epochs=epochs,
                accuracy=accuracy
            )

    print("\n" + "=" * 60)
    print("All Experiments Completed Successfully")
    print("Results are saved in the outputs folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()