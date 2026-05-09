import tensorflow as tf
from tensorflow.keras import layers, models
import os

def create_model(output_classes, input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(output_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if not os.path.exists("models"):
    os.makedirs("models")

# Image Type Classifier (3 classes: Brain, Bone, Chest)
type_model = create_model(3)
type_model.save("models/image_type_model.h5")
print("Saved models/image_type_model.h5")

# Brain Model (2 classes: Tumor, Normal)
brain_model = create_model(2)
brain_model.save("models/brain_model.h5")
print("Saved models/brain_model.h5")

# Bone Model (2 classes: Fracture, Normal)
bone_model = create_model(2)
bone_model.save("models/bone_model.h5")
print("Saved models/bone_model.h5")

# Chest Model (2 classes: Pneumonia, Normal)
chest_model = create_model(2)
chest_model.save("models/chest_model.h5")
print("Saved models/chest_model.h5")
