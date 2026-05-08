import tensorflow as tf
import numpy as np
from PIL import Image


def build_model():
    """Builds and returns the MobileNetV2-based model architecture."""
    base_model = tf.keras.applications.MobileNetV2(
        weights=None, include_top=False, input_shape=(224, 224, 3)
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    return model


def load_model(weights_path="ml/model/local_model_weights.h5"):
    """Loads the trained model weights and returns the model."""
    model = build_model()
    model.load_weights(weights_path)
    return model


def preprocess_image(img: Image.Image):
    """
    Accepts a PIL Image, resizes it to 224x224,
    normalizes pixel values, and returns a batch array.
    """
    img = img.convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img_resized


def predict(model, img_array):
    """
    Runs inference on a preprocessed image array.
    Returns: (label, confidence_percent)
    """
    classes = ['NORMAL', 'PNEUMONIA']
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    label = classes[class_idx]
    confidence = float(np.max(prediction)) * 100
    return label, confidence
