"""
Med-Chain: Improved Training Script for Chest X-Ray Classification
Fixes: frozen base model, proper fine-tuning, early stopping, full model save.
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

tf.keras.backend.clear_session()

# =================================================================
# CONFIGURATION — Change these paths to match your system
# =================================================================
TRAIN_DIR = r'C:\Users\anike\Desktop\Med-Chain-Project\chest_xray\train'
TEST_DIR  = r'C:\Users\anike\Desktop\Med-Chain-Project\chest_xray\test'
OUTPUT_WEIGHTS = 'local_model_weights.weights.h5'
OUTPUT_MODEL   = 'medchain_model.keras'  # Full model save (recommended)

# =================================================================
# STEP 1: Data Generators
# =================================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\nClass indices: {train_generator.class_indices}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# =================================================================
# STEP 2: Build Model with FROZEN base
# =================================================================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# CRITICAL FIX: Freeze the base model so pre-trained weights are not destroyed
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)  # Prevent overfitting
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nTrainable parameters (Phase 1): {sum(p.numpy().size for p in model.trainable_weights):,}")

# =================================================================
# STEP 3: Compute class weights from actual data
# =================================================================
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(train_generator.classes)
class_weights_array = compute_class_weight('balanced', classes=classes, y=train_generator.classes)
class_weights = dict(zip(classes, class_weights_array))
print(f"Computed class weights: {class_weights}")

# =================================================================
# STEP 4: Callbacks
# =================================================================
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# =================================================================
# STEP 5: Phase 1 — Train only the top layers (base frozen)
# =================================================================
print("\n===== PHASE 1: Training top layers (base frozen) =====")
history1 = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks
)

# =================================================================
# STEP 6: Phase 2 — Fine-tune the last 30 layers of base model
# =================================================================
print("\n===== PHASE 2: Fine-tuning last 30 layers =====")
base_model.trainable = True

# Freeze all except the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with a MUCH lower learning rate (so we don't destroy weights)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Trainable parameters (Phase 2): {sum(p.numpy().size for p in model.trainable_weights):,}")

history2 = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks
)

# =================================================================
# STEP 7: Evaluate on test set (if available)
# =================================================================
if os.path.exists(TEST_DIR):
    print("\n===== Evaluating on Test Set =====")
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    results = model.evaluate(test_generator)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]*100:.2f}%")

# =================================================================
# STEP 8: Save
# =================================================================
# Save weights (for backward compatibility with your existing code)
model.save_weights(OUTPUT_WEIGHTS)
print(f"\nWeights saved: {OUTPUT_WEIGHTS}")

# Save full model (recommended — includes architecture + weights + optimizer)
model.save(OUTPUT_MODEL)
print(f"Full model saved: {OUTPUT_MODEL}")

print("\n===== TRAINING COMPLETE =====")
