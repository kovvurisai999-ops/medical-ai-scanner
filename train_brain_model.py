"""
Brain Tumor Classification - High Accuracy Training (95% Target)
Architecture: DenseNet121 (State-of-the-Art for Medical Imaging)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import kagglehub

# -- Paths ------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Using your local dataset path
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "masoudnickparvar", "brain-tumor-mri-dataset", "versions", "2", "Training")

MODEL_OUT    = os.path.join(PROJECT_ROOT, "models", "brain_model.h5")

# -- Config -----------------------------------------------------------------
IMG_SIZE   = 224
BATCH_SIZE = 32 # Increased for stability
EPOCHS     = 40
CLASSES    = ['glioma', 'meningioma', 'notumor', 'pituitary']

# -- Data Loading with Heavy Augmentation -----------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.15
)

test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

print(f"Loading data from: {DATASET_DIR}")
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    subset='training'
)

val_generator = test_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES,
    subset='validation'
)

# -- Model Building (DenseNet121) -------------------------------------------
print("\nBuilding DenseNet121 Pipeline...")
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(CLASSES), activation='softmax')
])

# Use Label Smoothing to help with Meningioma/Pituitary confusion
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# -- Advanced Callbacks -----------------------------------------------------
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-7),
    ModelCheckpoint(MODEL_OUT, monitor='val_accuracy', save_best_only=True, verbose=1)
]

# -- Phase 1: Training Head -------------------------------------------------
print("\n[Phase 1] Training Classifier Head...")
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks
)

# -- Phase 2: Fine-Tuning ---------------------------------------------------
print("\n[Phase 2] Fine-tuning late layers of DenseNet...")
base_model.trainable = True
# Unfreeze the last 50 layers
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5), # Tiny learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=callbacks
)

print(f"\n[SUCCESS] High-accuracy model saved to {MODEL_OUT}")
