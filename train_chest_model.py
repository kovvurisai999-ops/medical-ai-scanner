"""
Pneumonia Detection CNN Training Script
Inspired by Kaggle Kernel: madz2000/pneumonia-detection-using-cnn-92-6-accuracy
Dataset: paultimothymooney/chest-xray-pneumonia
"""

import os
import kagglehub
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16
import warnings
warnings.filterwarnings('ignore')

import time
import requests

print("Fetching Chest X-Ray Pneumonia Dataset using Kagglehub (No API Key Required)...")

def reliable_kaggle_download(dataset_name, max_retries=15):
    for attempt in range(max_retries):
        try:
            print(f"Download Attempt {attempt + 1}/{max_retries}...")
            return kagglehub.dataset_download(dataset_name)
        except Exception as e:
            print(f"Network Interruption: {e}")
            if attempt < max_retries - 1:
                print("Re-establishing connection in 3 seconds...")
                time.sleep(3)
            else:
                raise e

dataset_path = reliable_kaggle_download("paultimothymooney/chest-xray-pneumonia")
base_dir = os.path.join(dataset_path, "chest_xray")

if not os.path.exists(base_dir):
    # Sometimes it's nested differently
    if os.path.exists(os.path.join(dataset_path, "chest_xray", "chest_xray")):
        base_dir = os.path.join(dataset_path, "chest_xray", "chest_xray")
    else:
        print(f"Warning: Unexpected dataset structure. Contents of {dataset_path}:", os.listdir(dataset_path))

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

IMG_SIZE = 150
BATCH_SIZE = 32

print(f"Training Directory: {train_dir}")
print("Configuring Data Generators with Augmentation...")

# Data augmentation mimicking the notebook
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)

test_val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb'
)

val_generator = test_val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb'
)

print("Building VGG16 Transfer Learning Architecture...")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False 

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

checkpoint = ModelCheckpoint(r'models\chest_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.000001)

print("Starting Transfer Learning Training...")
EPOCHS = 10 

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[learning_rate_reduction, checkpoint]
)

print("Training Complete! The best weights are saved in models/chest_model.h5.")
