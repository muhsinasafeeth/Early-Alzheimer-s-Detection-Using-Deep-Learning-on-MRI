"""
gradcam.py
----------
Reusable Grad-CAM implementation for the Alzheimer's Detection CNN.

The model is a plain tf.keras Sequential CNN (Conv2D -> MaxPool x3 ->
GlobalAveragePooling2D -> Dense -> Dropout -> Dense(1, sigmoid)) trained for
binary classification (0 = NonDemented, 1 = VeryMildDemented / Alzheimer's).

This mirrors the Grad-CAM logic from the training notebook (cells 51-53),
with the last convolutional layer auto-detected so the code keeps working
even if the model is retrained/re-architected later.
"""

import numpy as np
import tensorflow as tf
import cv2


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """
    Walk the model's layers in reverse and return the name of the last
    Conv2D layer. Raises if no Conv2D layer is found.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str | None = None,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for a single preprocessed image batch.

    Parameters
    ----------
    img_array : np.ndarray
        Preprocessed input of shape (1, 224, 224, 3), values in [0, 1].
    model : tf.keras.Model
        The trained Keras model.
    last_conv_layer_name : str, optional
        Name of the conv layer to explain. Auto-detected if not given.

    Returns
    -------
    np.ndarray
        2D heatmap (values in [0, 1]) sized to the last conv layer's
        spatial dimensions.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # NOTE: We deliberately do a manual layer-by-layer forward pass here
    # instead of building a `tf.keras.models.Model(inputs=..., outputs=...)`
    # sub-model. With Keras 3, rebuilding a functional sub-model around a
    # loaded Sequential model's symbolic KerasTensors can silently
    # disconnect the graph from a freshly-fed input tensor, which makes
    # gradients come back as None. Running each layer explicitly inside
    # the tape guarantees the conv layer's output is actually connected
    # to (and differentiable w.r.t.) the input we care about.
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        x = img_tensor
        conv_outputs = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                conv_outputs = x
        if conv_outputs is None:
            raise ValueError(f"Layer '{last_conv_layer_name}' not found while tracing the model.")
        predictions = x
        # Single sigmoid output neuron -> use it directly as the "class score"
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Guard against a degenerate all-zero heatmap (avoid div-by-zero)
    max_val = tf.math.reduce_max(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (max_val + 1e-8)

    return heatmap.numpy()


def overlay_heatmap(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Resize the heatmap to the original image size, colorize it, and
    superimpose it on the original image.

    Parameters
    ----------
    original_img : np.ndarray
        RGB image, uint8, shape (H, W, 3), values in [0, 255].
    heatmap : np.ndarray
        2D Grad-CAM heatmap, values in [0, 1].
    alpha : float
        Heatmap opacity in the overlay.

    Returns
    -------
    np.ndarray
        uint8 RGB image (H, W, 3) with the heatmap overlaid.
    """
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # cv2.applyColorMap expects BGR-ish colormaps; it returns BGR order.
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

    superimposed = heatmap_color_rgb * alpha + original_img * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return superimposed


def get_heatmap_only(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """
    Return a standalone colorized heatmap (no overlay), resized to `size`
    (width, height), as an RGB uint8 array. Useful for a side-by-side
    "original | heatmap | overlay" display.
    """
    heatmap_resized = cv2.resize(heatmap, size)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_color_rgb
