import os

os.environ["TF_CPP_MINLOG_LEVEL"] = "2"
import numpy as np
import cv2
import tensorflow as tf
from model.unet import build_unet
from dataloader.data import load_data, tf_dataset
from tensorflow.keras.callbacks import *

if __name__ == "__main__":
    """Hyperparameters"""
    dataset_path = "dataset"
    input_shape = (256, 256, 3)
    batch_size = 8
    epochs = 10
    lr = 1e-4
    model_path = "saved_model/unet.h5"
    csv_path = "data.csv"

    """Load the Dataset"""
    (train_x, train_y), (test_x, test_y) = load_data(dataset_path)

    print("Training Images:")
    print(f"Training: {len(train_x)} - {len(train_y)}")
    print(f"Testing: {len(test_x)} - {len(test_y)}")

    train_ds = tf_dataset(train_x, train_y, batch_size=batch_size)
    val_ds = tf_dataset(test_x, test_y, batch_size=batch_size)

    """Build the Model"""
    model = build_unet(input_shape)

    """Compile the Model"""
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=[
            tf.keras.metrics.MeanIoU(num_classes=2),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(),
        ],
    )

    """Summary of the Model"""
    model.summary()

    """Setting up Callback"""
    callbacks = [
        ModelCheckpoint(model_path, monitor="val_loss", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.1, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor="val_loss", patience=5),
    ]

    train_steps = len(train_x) // batch_size
    if len(train_x) % batch_size != 0:
        train_steps += 1

    test_steps = len(test_x) // batch_size
    if len(test_x) % batch_size != 0:
        test_steps += 1

    """Train the Model"""
    print("Training Started!")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=test_steps,
        callbacks=callbacks,
    )
