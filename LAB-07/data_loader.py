# data_loader.py

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_dataset(
    data_dir="images",
    img_size=(128, 128),
    batch_size=32,
    validation_split=0.2
):
    """
    Load Cat and Dog dataset from images folder.

    Folder structure:
    images/
        Cat/
        Dog/
    """

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Folder '{data_dir}' not found.")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=validation_split
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False
    )

    return train_generator, validation_generator