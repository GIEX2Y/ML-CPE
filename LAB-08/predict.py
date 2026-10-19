import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def predict_image(model_path, image_path):

    model = load_model(model_path)

    img = image.load_img(
        image_path,
        target_size=(128, 128)
    )

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array, verbose=0)[0][0]

    if prediction >= 0.5:
        label = "Dog"
        confidence = prediction
    else:
        label = "Cat"
        confidence = 1 - prediction

    print("=" * 40)
    print(f"Image      : {image_path}")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence * 100:.2f}%")