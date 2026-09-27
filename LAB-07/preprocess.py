# preprocess.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generators(
    data_dir="images",
    img_size=(128, 128),
    batch_size=32,
    validation_split=0.2
):
    """
    Create training and validation data generators.

    Folder structure:
    images/
    ├── Cat/
    └── Dog/
    """

    # Image augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=validation_split
    )

    # Validation data (only normalize)
    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=validation_split
    )

    # Training Generator
    train_generator = train_datagen.flow_from_directory(
        directory=data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True
    )

    # Validation Generator
    validation_generator = validation_datagen.flow_from_directory(
        directory=data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False
    )

    return train_generator, validation_generator


if __name__ == "__main__":

    train_gen, val_gen = create_data_generators()

    print("\nDataset Information")
    print("-" * 40)
    print("Training Samples   :", train_gen.samples)
    print("Validation Samples :", val_gen.samples)
    print("Classes            :", train_gen.class_indices)
    print("Image Size         :", train_gen.target_size)
    print("Batch Size         :", train_gen.batch_size)