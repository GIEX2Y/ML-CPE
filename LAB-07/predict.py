# predict.py

import numpy as np
from tensorflow.keras.preprocessing import image


def predict_image(model, image_path, img_size=(128, 128)):
    """
    Predict Cat or Dog from a single image.
    """

    # Load image
    img = image.load_img(
        image_path,
        target_size=img_size
    )

    # Convert to array
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)

    probability = prediction[0][0]

    if probability < 0.5:
        label = "Cat"
        confidence = (1 - probability) * 100
    else:
        label = "Dog"
        confidence = probability * 100

    print("=" * 40)
    print("Prediction Result")
    print("=" * 40)
    print(f"Image      : {image_path}")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.2f}%")
    print("=" * 40)

    return label, confidence


if __name__ == "__main__":

    from tensorflow.keras.models import load_model

    model = load_model("outputs/Config3_30epoch.keras")

    predict_image(
        model,
        "images/Cat/cat1.jpg"
    )