"""
Med-Chain AI Backend — FastAPI server for X-ray diagnosis.
Deploy this to HuggingFace Spaces (Docker SDK) for free hosting.
"""

import io
import base64
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# ===== APP SETUP =====
app = FastAPI(title="Med-Chain AI API")

# Allow frontend to call this API from any origin (Vercel, localhost, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MODEL LOADING (runs once at startup) =====
def load_model():
    """Loads the full saved model (.keras format)."""
    return tf.keras.models.load_model("medchain_model.keras")

print("Loading AI model...")
model = load_model()
print("Model loaded successfully.")

# ===== GRAD-CAM =====
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, img_resized):
    img_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    return superimposed_rgb

def numpy_to_base64(img_array):
    """Converts a numpy RGB array to a base64-encoded PNG string."""
    img = Image.fromarray(img_array.astype('uint8'))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ===== API ENDPOINTS =====

@app.get("/health")
def health_check():
    """Health check endpoint — frontend uses this to show connection status."""
    return {"status": "ok", "model": "MobileNetV2"}

@app.post("/predict")
async def predict_xray(file: UploadFile = File(...)):
    """
    Accepts a chest X-ray image, returns:
    - diagnosis: NORMAL or PNEUMONIA
    - confidence: percentage (0-100)
    - heatmap_image: base64-encoded Grad-CAM overlay PNG
    """
    # 1. Read and preprocess
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 2. Predict
    classes = ["NORMAL", "PNEUMONIA"]
    prediction = model.predict(img_array)
    class_idx = int(np.argmax(prediction))
    label = classes[class_idx]
    confidence = float(np.max(prediction)) * 100

    # 3. Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model)
    superimposed = overlay_heatmap(heatmap, img_resized)
    heatmap_b64 = numpy_to_base64(superimposed)

    return {
        "diagnosis": label,
        "confidence": round(confidence, 2),
        "heatmap_image": heatmap_b64,
    }
