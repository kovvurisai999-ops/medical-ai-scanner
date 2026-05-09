"""
Fast Modality Type Classifier (Brain vs Bone vs Chest)
This ensures cross-verification works correctly.
Since we don't have Chest data readily, we'll synthesize a 3-class classifier 
mostly focused on rejecting Brain-Bone mixups.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# -- Config --
IMG_SIZE = 224
NUM_CLASSES = 3

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BRAIN_DIR = os.path.join(PROJECT_ROOT, "datasets", "navoneel", "brain-mri-images-for-brain-tumor-detection", "versions", "1", "brain_tumor_dataset", "yes")
BONE_DIR = os.path.join(PROJECT_ROOT, "datasets", "jockeroika", "human-bone-fractures-image-dataset", "versions", "1", "Human Bone Fractures Multi-modal Image Dataset (HBFMID)", "Bone Fractures Detection", "train", "images")
MODEL_OUT = os.path.join(PROJECT_ROOT, "models", "image_type_model.h5")

def load_images(folder, label, max_imgs=300):
    images, labels = [], []
    if not os.path.exists(folder):
        return images, labels
    files = os.listdir(folder)[:max_imgs]
    for idx, f in enumerate(files):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        try:
            img_path = os.path.join(folder, f)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img / 255.0)
            labels.append(label)
        except:
            pass
    return images, labels

CHEST_DIR = os.path.join(PROJECT_ROOT, "datasets", "paultimothymooney", "chest-xray-pneumonia", "versions", "2", "chest_xray", "chest_xray", "train", "NORMAL")

print("Loading Brain Dataset...")
X_brain, y_brain = load_images(BRAIN_DIR, 0, 300)

print("Loading Bone Dataset...")
X_bone, y_bone = load_images(BONE_DIR, 1, 300)

print("Loading Chest Dataset...")
X_chest, y_chest = load_images(CHEST_DIR, 2, 300)

X = np.array(X_brain + X_bone + X_chest, dtype=np.float32)
y = np.array(y_brain + y_bone + y_chest, dtype=np.int32)

# Shuffle
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = tf.keras.utils.to_categorical(y[indices], NUM_CLASSES)

print(f"Total images: {len(X)}")

# Fast MobileNet model
base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base.trainable = False

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dense(128, activation='relu')(x)
out = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(base.input, out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nTraining Modality Router (Epochs=5)...")
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

model.save(MODEL_OUT)
print(f"Saved robust routing model to {MODEL_OUT}")
