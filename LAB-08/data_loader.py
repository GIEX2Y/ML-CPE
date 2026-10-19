import tensorflow as tf

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
LIMIT = 500   # ใช้ทั้งหมด 500 รูป (250 Cat + 250 Dog)

def create_generators(data_dir):

    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.30,
        subset="both",
        seed=42,
        label_mode="binary"
    )

    train_dataset, validation_dataset = dataset

    # จำกัดจำนวน Batch
    train_batches = int((LIMIT * 0.7) / BATCH_SIZE)
    val_batches = int((LIMIT * 0.3) / BATCH_SIZE)

    train_dataset = train_dataset.take(train_batches)
    validation_dataset = validation_dataset.take(val_batches)

    # Normalize
    normalization = tf.keras.layers.Rescaling(1./255)

    train_dataset = train_dataset.map(
        lambda x, y: (normalization(x), y)
    )

    validation_dataset = validation_dataset.map(
        lambda x, y: (normalization(x), y)
    )

    return train_dataset, validation_dataset