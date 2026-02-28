import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

def build_flower_model(img_size=(224,224),
                       batch_size=32,
                       training_type="transfer",
                       num_classes=5,
                       train_path="data/train",
                       val_path="data/val"):
    
    # =======================
    # 1️⃣ DATA LOADING
    # =======================
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        image_size=img_size,
        batch_size=batch_size
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_path,
        image_size=img_size,
        batch_size=batch_size
    )
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    
    # =======================
    # 2️⃣ BASE MODEL
    # =======================
    
    base_model = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size[0], img_size[1], 3)
    )
    
    # =======================
    # 3️⃣ TRAINING TYPE
    # =======================
    
    if training_type == "transfer":
        base_model.trainable = False
        learning_rate = 1e-3
        
    elif training_type == "finetune":
        base_model.trainable = True
        
        # Freeze early layers, unfreeze last 30
        for layer in base_model.layers[:-30]:
            layer.trainable = False
            
        learning_rate = 1e-5
        
    else:
        raise ValueError("training_type must be 'transfer' or 'finetune'")
    
    # =======================
    # 4️⃣ MODEL HEAD
    # =======================
    
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))
    
    x = keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    
    # =======================
    # 5️⃣ COMPILE
    # =======================
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
      # =======================
    # 4️⃣ PREPARE SAVE PATH
    # =======================
    save_dir = "flower_model"
    os.makedirs(save_dir, exist_ok=True)
    
    model_name = f"flower_model_{training_type}_{img_size[0]}x{img_size[1]}.keras"
    model_path = os.path.join(save_dir, model_name)

    return model, train_ds, val_ds, model_path

model, train_ds, val_ds, model_path = build_flower_model(
    img_size=(224,224),
    batch_size=32,
    training_type="transfer"
)

history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# Save after training
model.save(model_path)

print("Model saved at:", model_path)