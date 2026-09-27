# cnn_model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)


def build_cnn_model(config=1, input_shape=(128, 128, 3)):
    """
    Build CNN model with different configurations.

    config = 1 -> Basic CNN
    config = 2 -> Deeper CNN
    config = 3 -> CNN + BatchNorm + Dropout
    """

    model = Sequential()

    # ==========================
    # Configuration 1
    # ==========================
    if config == 1:

        model.add(Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=input_shape
        ))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(
            64,
            (3, 3),
            activation="relu"
        ))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        model.add(Dense(
            128,
            activation="relu"
        ))

        model.add(Dense(
            1,
            activation="sigmoid"
        ))

    # ==========================
    # Configuration 2
    # ==========================
    elif config == 2:

        model.add(Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=input_shape
        ))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(
            64,
            (3, 3),
            activation="relu"
        ))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(
            128,
            (3, 3),
            activation="relu"
        ))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        model.add(Dense(
            256,
            activation="relu"
        ))

        model.add(Dense(
            1,
            activation="sigmoid"
        ))

    # ==========================
    # Configuration 3
    # ==========================
    else:

        model.add(Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=input_shape
        ))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(
            64,
            (3, 3),
            activation="relu"
        ))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(
            128,
            (3, 3),
            activation="relu"
        ))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())

        model.add(Dense(
            256,
            activation="relu"
        ))

        model.add(Dropout(0.5))

        model.add(Dense(
            1,
            activation="sigmoid"
        ))

    # ==========================
    # Compile Model
    # ==========================

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    cnn = build_cnn_model(config=1)

    cnn.summary()