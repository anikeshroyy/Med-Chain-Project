import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.keras.backend.clear_session()

# 1. Path Setup
train_dir = r'C:\Users\prave\Desktop\project 8\chest_xray\chest_xray\train'

# 2. Data Generators (With Augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

# --- YE DO BLOCKS ZAROORI HAIN (Inke bina error aayega) ---
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)
# -------------------------------------------------------

# 3. Model Setup
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Class Weights (Normal images ko zyada weight diya taaki bias khatam ho)
class_weights = {0: 2.5, 1: 1.0}

# 5. Training
print("\n--- Training Starting on GPU (15 Epochs) ---")
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    class_weight=class_weights
)

# 6. Save the Weights
model.save_weights("local_model_weights.h5")
print("\n--- Success! Weights saved as local_model_weights.h5 ---")