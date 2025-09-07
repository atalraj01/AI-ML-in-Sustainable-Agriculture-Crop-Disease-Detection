
# -------------------------
# Step 1: Import libraries
# -------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
# -------------------------
# Step 2: Dataset loading
# -------------------------
# ⚠️ IMPORTANT:
# Download PlantVillage dataset from Kaggle:
# https://www.kaggle.com/datasets/emmarex/plantdisease
# Unzip and place in "dataset/PlantVillage"
# Folder structure should be:
# dataset/PlantVillage/<class_name>/<image_files>.jpg

dataset_dir = "/Users/atalraj/Downloads/PlantVillage"
img_size = (128,128)
batch_size = 32

# Data augmentation + normalization
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    subset="training",
    class_mode="categorical"
)

val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    subset="validation",
    class_mode="categorical"
)
# -------------------------
# Step 3: Baseline CNN Model
# -------------------------
num_classes = len(train_gen.class_indices)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# -------------------------
# Step 4: Train model
# -------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# -------------------------
# Step 5: Plot results
# -------------------------
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy (Baseline CNN)")
plt.show()

# -------------------------
# Step 6: Plot Training Results
# -------------------------

# Accuracy Plot
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("accuracy_plot.png")   # saves image
plt.show()

# Loss Plot
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")   # saves image
plt.show()

print("✅ Accuracy and loss plots saved as 'accuracy_plot.png' and 'loss_plot.png'")