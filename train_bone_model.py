"""
Bone Fracture Classification Model Training
Dataset: Human Bone Fractures Multi-modal Image Dataset (HBFMID)
Classes: 10 fracture types
Architecture: MobileNetV2 (Transfer Learning)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
DATASET_BASE = (
    r"D:\Softwares\ai brain and bone scan detection\datasets"
    r"\jockeroika\human-bone-fractures-image-dataset\versions\1"
    r"\Human Bone Fractures Multi-modal Image Dataset (HBFMID)"
    r"\Bone Fractures Detection"
)

TRAIN_DIR  = os.path.join(DATASET_BASE, "train", "images")
VALID_DIR  = os.path.join(DATASET_BASE, "valid", "images")
TEST_DIR   = os.path.join(DATASET_BASE, "test",  "images")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_OUT    = os.path.join(PROJECT_ROOT, "models", "bone_model.h5")

# ── Config ─────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 30
NUM_CLASSES = 10

CLASS_NAMES = [
    "Comminuted", "Greenstick", "Healthy", "Linear",
    "Oblique Displaced", "Oblique", "Segmental",
    "Spiral", "Transverse Displaced", "Transverse"
]

print("=" * 60)
print("   Bone Fracture Model Training")
print("=" * 60)
print(f"Train dir : {TRAIN_DIR}")
print(f"Valid dir : {VALID_DIR}")
print(f"Output    : {MODEL_OUT}")
print(f"Classes   : {NUM_CLASSES}")
print()

# ── Dataset ────────────────────────────────────────────────────────────────
# NOTE: The dataset uses YOLO label format (labels/ folder).
# Images are NOT in class subfolders, so we need a custom approach.
# We'll use the label .txt files to map images → class index.

def load_yolo_dataset(img_dir, label_dir, img_size=IMG_SIZE):
    """Load YOLO-format dataset, mapping images to their dominant class."""
    images, labels = [], []
    label_dir_path = label_dir

    img_files = [f for f in os.listdir(img_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    missing, loaded = 0, 0
    for img_file in img_files:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(label_dir_path, label_file)

        if not os.path.exists(label_path):
            missing += 1
            continue

        with open(label_path) as f:
            lines = f.read().strip().split('\n')

        if not lines or lines[0] == '':
            missing += 1
            continue

        # Take the first annotation's class (dominant object)
        try:
            cls = int(lines[0].split()[0])
        except (ValueError, IndexError):
            missing += 1
            continue

        img_path = os.path.join(img_dir, img_file)
        img = tf.keras.preprocessing.image.load_img(
            img_path, target_size=(img_size, img_size)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(cls)
        loaded += 1

    print(f"  Loaded {loaded} images | Skipped {missing} (no label)")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


print("Loading training data ...")
X_train, y_train = load_yolo_dataset(
    TRAIN_DIR,
    os.path.join(DATASET_BASE, "train", "labels")
)

print("Loading validation data ...")
X_val, y_val = load_yolo_dataset(
    VALID_DIR,
    os.path.join(DATASET_BASE, "valid", "labels")
)

print("Loading test data ...")
X_test, y_test = load_yolo_dataset(
    TEST_DIR,
    os.path.join(DATASET_BASE, "test", "labels")
)

print(f"\nDataset sizes  ->  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# One-hot encode
y_train_oh = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_oh   = tf.keras.utils.to_categorical(y_val,   NUM_CLASSES)
y_test_oh  = tf.keras.utils.to_categorical(y_test,  NUM_CLASSES)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# ── Model ──────────────────────────────────────────────────────────────────
print("\nBuilding model (MobileNetV2 transfer learning) ...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze base initially

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# -- Phase 1: Train classifier head ----------------------------------------
print("\n-- Phase 1: Training classifier head (frozen base) --")
callbacks_p1 = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(MODEL_OUT, save_best_only=True, verbose=1)
]

history1 = model.fit(
    datagen.flow(X_train, y_train_oh, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=15,
    validation_data=(X_val, y_val_oh),
    callbacks=callbacks_p1
)

# -- Phase 2: Fine-tune top layers -----------------------------------------
print("\n-- Phase 2: Fine-tuning top 30 layers of base --")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_p2 = [
    EarlyStopping(patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.3, patience=3, verbose=1),
    ModelCheckpoint(MODEL_OUT, save_best_only=True, verbose=1)
]

history2 = model.fit(
    datagen.flow(X_train, y_train_oh, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val_oh),
    callbacks=callbacks_p2
)

# -- Evaluation ------------------------------------------------------------
print("\n-- Evaluating on test set --")
loss, acc = model.evaluate(X_test, y_test_oh, verbose=1)
print(f"\nTest Accuracy : {acc*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

# -- Update ml_model.py labels ---------------------------------------------
print("\n[OK] bone_model.h5 saved to:", MODEL_OUT)
print("\nClass mapping for ml_model.py:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {i}: {name}")

print("\nDone! The bone model is ready for inference.")
