import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Replace this with your actual path after extracting the zip
train_dir = r'C:\Users\prave\Desktop\project 8\chest_xray\chest_xray\train'

# Preprocessing: Rescaling pixel values from [0, 255] to [0, 1]
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print(f"Success! Found {train_generator.samples} images belonging to {train_generator.num_class} classes.")