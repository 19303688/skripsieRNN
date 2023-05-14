"""Assignment 2, task 1 for computer vision."""
import time
import matplotlib.pyplot as plt
import tensorflow as tf

import sys

#import tensorflow.keras
import pandas as pd
import numpy as np
import sklearn as sk
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
import platform
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# CIFAR-10 image size (50000, 32, 32, 3)
print(f"CIFAR-10 image size: {x_train.shape}")


# Build the transfer learning model
def transfer_model() -> tf.keras.Model:
    """Build a ResNet model with a 10 class classifier for images of shape (32, 32, 3).

    Returns:
        tf.keras.Model: Compiled model.
    """
    # Inputs
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    # Resize and rescale
    layer = tf.keras.layers.Resizing(224, 224, interpolation="bilinear")(inputs)
    # layer = tf.keras.layers.Rescaling(scale=1.0 / 255)(layer)
    layer = tf.keras.applications.resnet_v2.preprocess_input(layer)
    # Augment
    # layer = tf.keras.layers.RandomFlip("horizontal")(layer)
    # layer = tf.keras.layers.RandomRotation(factor=(-0.2, 0.2))(layer)
    # Load pre-trained resnet50v2 encoder
    resnet_encoder = tf.keras.applications.ResNet50V2(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    # Freeze the ResNet encoder
    resnet_encoder.trainable = False
    # Add this to model
    layer = resnet_encoder(layer)
    # Add the classifier
    layer = tf.keras.layers.Flatten()(layer)
    layer = tf.keras.layers.Dense(512, activation="relu")(layer)
    layer = tf.keras.layers.Dense(128, activation="relu")(layer)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(layer)
    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Compile
    model.compile(
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        optimizer=tf.keras.optimizers.Adam(),
    )
    return model


# Build the model
trans_model = transfer_model()
trans_model.summary()

# Train the model
es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
# csv_callback = tf.keras.callbacks.CSVLogger("logging/model_fit_finetuning.csv")
# Time the training
start_time = time.time()
history = trans_model.fit(
    x_train,
    y_train,
    epochs=100,
    validation_split=0.2,
    batch_size=64,
    callbacks=[es_callback],#, csv_callback],
    use_multiprocessing=True,
)
print(f"Time to train: {time.time() - start_time} seconds")
