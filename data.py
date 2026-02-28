import os
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR    = "./flower_photos"
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    validation_split=0.2,
)


val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
)
