"""
Quick save script: Retrains and saves the model.
Run this if the main training completed but saving failed.
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os

tf.keras.backend.clear_session()

TRAIN_DIR = r'C:\Users\anike\Desktop\Med-Chain-Project\chest_xray\train'
TEST_DIR  = r'C:\Users\anike\Desktop\Med-Chain-Project\chest_xray\test'

# Data
train_datagen = ImageDataGenerator(
    rescale=1./255, validation_split=0.2,
    rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
    zoom_range=0.15, horizontal_flip=True, brightness_range=[0.8, 1.2], fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224), batch_size=32, class_mode='categorical', subset='training', shuffle=True)
val_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224), batch_size=32, class_mode='categorical', subset='validation', shuffle=False)

# Class weights
cw = compute_class_weight('balanced', classes=np.unique(train_gen.classes), y=train_gen.classes)
class_weights = dict(enumerate(cw))
print(f"Class weights: {class_weights}")

# Model
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base.trainable = False
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
pred = Dense(2, activation='softmax')(x)
model = Model(inputs=base.input, outputs=pred)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

# Phase 1
print("\n=== Phase 1: Top layers ===")
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=20, validation_data=val_gen, class_weight=class_weights, callbacks=callbacks)

# Phase 2
print("\n=== Phase 2: Fine-tuning ===")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=15, validation_data=val_gen, class_weight=class_weights, callbacks=callbacks)

# Test
if os.path.exists(TEST_DIR):
    test_gen = test_datagen.flow_from_directory(TEST_DIR, target_size=(224,224), batch_size=32, class_mode='categorical', shuffle=False)
    results = model.evaluate(test_gen)
    print(f"\nTest Accuracy: {results[1]*100:.2f}%")

# SAVE (fixed filenames)
model.save_weights('local_model_weights.weights.h5')
model.save('medchain_model.keras')
print("\n=== SAVED SUCCESSFULLY ===")
print("Files: local_model_weights.weights.h5, medchain_model.keras")
