import tensorflow as tf
import numpy as np
import cv2


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1", pred_index=None):
    """
    Generates a Grad-CAM heatmap for the given image array.
    Returns a normalized numpy heatmap (values 0.0 to 1.0).
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(heatmap, original_img_resized):
    """
    Overlays the Grad-CAM heatmap on the original image.
    Accepts a numpy heatmap and a PIL Image (224x224).
    Returns an RGB numpy array with the heatmap superimposed.
    """
    img_bgr = cv2.cvtColor(np.array(original_img_resized), cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    return superimposed_rgb
