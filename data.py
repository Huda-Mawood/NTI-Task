import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================
#  FUNCTION 1 — Train Generator
# ============================================================
def get_train_generator():
    train_generator = augmentation.flow_from_directory(
        directory   = DATASET_DIR,
        target_size = IMG_SIZE,
        batch_size  = BATCH_SIZE,
        class_mode  = "binary",
        subset      = "training",
        seed        = SEED,
        shuffle     = True
    )

    print(f"   Classes     : {train_generator.class_indices}")
    print(f"   Samples     : {train_generator.samples}")
    print(f"   Steps/epoch : {len(train_generator)}")

    return train_generator


# ============================================================
#  FUNCTION 2 — Validation Generator
# ============================================================
def get_validation_generator():
    val_datagen = ImageDataGenerator(
        rescale          = 1.0 / 255,
        validation_split = VAL_SPLIT
    )

    val_generator = val_datagen.flow_from_directory(
        directory   = DATASET_DIR,
        target_size = IMG_SIZE,
        batch_size  = BATCH_SIZE,
        class_mode  = "binary",
        subset      = "validation",
        seed        = SEED,
        shuffle     = False
    )

    print(f"   Classes  : {val_generator.class_indices}")
    print(f"   Samples  : {val_generator.samples}")
    print(f"   Steps    : {len(val_generator)}")

    return val_generator


train_gen = get_train_generator()
val_gen   = get_validation_generator()


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
