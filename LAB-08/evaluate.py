import os
import pandas as pd
from tensorflow.keras.models import load_model


def evaluate_models(validation_generator):

    results = []

    os.makedirs("outputs", exist_ok=True)

    for config in [1, 2, 3]:
        for epoch in [10, 20, 30]:

            model_path = f"outputs/Config{config}_{epoch}epoch.keras"

            if not os.path.exists(model_path):
                print(f"{model_path} not found")
                continue

            print("=" * 50)
            print(f"Evaluating Config {config} ({epoch} Epochs)")

            model = load_model(model_path)

            loss, accuracy = model.evaluate(
                validation_generator,
                verbose=0
            )

            print(f"Loss     : {loss:.4f}")
            print(f"Accuracy : {accuracy:.4f}")

            results.append({
                "Configuration": config,
                "Epoch": epoch,
                "Loss": round(loss, 4),
                "Accuracy": round(accuracy, 4)
            })

    df = pd.DataFrame(results)

    print("\n========== SUMMARY ==========")
    print(df)

    df.to_csv(
        "outputs/accuracy_results.csv",
        index=False
    )
