from data_loader import create_generators
from train import train_model
from evaluate import evaluate_models
from predict import predict_image

DATA_PATH = "images"

train_generator, validation_generator = create_generators(DATA_PATH)

epochs_list = [10, 20, 30]
configs = [1, 2, 3]

for config in configs:

    for epochs in epochs_list:

        print("=" * 60)
        print(f"Training Config {config} ({epochs} Epochs)")
        print("=" * 60)

        train_model(
            train_generator,
            validation_generator,
            config,
            epochs
        )

print("\nTraining Finished")

results = evaluate_models(validation_generator)

print(results)

predict_image(
    "outputs/Config3_30epoch.keras",
    "images/Cat/1.jpg"
)