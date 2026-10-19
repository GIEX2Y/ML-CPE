from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout


def build_model(config=1):

    model = Sequential()

    # ===== Config 1 =====
    if config == 1:

        model.add(Conv2D(32, (3,3), activation="relu",
                         input_shape=(128,128,3)))
        model.add(MaxPooling2D())

        model.add(Conv2D(64, (3,3), activation="relu"))
        model.add(MaxPooling2D())

        model.add(Flatten())

        model.add(Dense(128, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

    # ===== Config 2 =====
    elif config == 2:

        model.add(Conv2D(32,(3,3),activation="relu",
                         input_shape=(128,128,3)))
        model.add(MaxPooling2D())

        model.add(Conv2D(64,(3,3),activation="relu"))
        model.add(MaxPooling2D())

        model.add(Conv2D(128,(3,3),activation="relu"))
        model.add(MaxPooling2D())

        model.add(Flatten())

        model.add(Dense(256,activation="relu"))
        model.add(Dropout(0.5))

        model.add(Dense(1,activation="sigmoid"))

    # ===== Config 3 =====
    else:

        model.add(Conv2D(32,(3,3),activation="relu",
                         input_shape=(128,128,3)))
        model.add(MaxPooling2D())

        model.add(Conv2D(64,(3,3),activation="relu"))
        model.add(MaxPooling2D())

        model.add(Conv2D(128,(3,3),activation="relu"))
        model.add(MaxPooling2D())

        model.add(Conv2D(256,(3,3),activation="relu"))
        model.add(MaxPooling2D())

        model.add(Flatten())

        model.add(Dense(512,activation="relu"))
        model.add(Dropout(0.5))

        model.add(Dense(1,activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model