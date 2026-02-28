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

