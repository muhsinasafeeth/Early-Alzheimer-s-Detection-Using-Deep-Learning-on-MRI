"""
app.py
------
Early-Stage Alzheimer's Detection — Streamlit App

Upload a brain MRI image and the app will:
  1. Preprocess it exactly as done during training (224x224, RGB, /255.0)
  2. Predict NonDemented vs VeryMildDemented (Alzheimer's) with a confidence score
  3. Generate a Grad-CAM heatmap explaining which regions drove the prediction
  4. Show original image, heatmap, and overlay side by side
  5. Let the user download the Grad-CAM overlay image
"""

import io

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from preprocess import load_image, preprocess_image, prepare_input
from gradcam import make_gradcam_heatmap, overlay_heatmap, find_last_conv_layer

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Early-Stage Alzheimer's Detection",
    page_icon="🧠",
    layout="wide",
)

MODEL_PATH = "alzheimer_model_.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = {0: "Non-Demented", 1: "Very Mild Demented (Alzheimer's)"}
THRESHOLD = 0.5


# ----------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    last_conv = find_last_conv_layer(model)
    return model, last_conv


# ----------------------------------------------------------------------------
# Sidebar — model information
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("🧠 About This Model")
    st.markdown(
        """
        **Task:** Binary classification of brain MRI scans

        **Classes:**
        - `0` → Non-Demented
        - `1` → Very Mild Demented (early-stage Alzheimer's)

        **Architecture:** Custom CNN
        - 3× (Conv2D → MaxPooling2D)
        - GlobalAveragePooling2D
        - Dense(128, ReLU) → Dropout(0.5)
        - Dense(1, Sigmoid)

        **Input size:** 224 × 224 × 3 (RGB)

        **Preprocessing:** Pixel values scaled to [0, 1]

        **Explainability:** Grad-CAM highlights the regions of the
        scan that most influenced the model's prediction.
        """
    )
    st.divider()
    st.warning(
        "⚠️ **Disclaimer:** This tool is for educational/research "
        "demonstration purposes only and is **not** a medical diagnostic "
        "device. Always consult a qualified healthcare professional for "
        "medical advice."
    )

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.title("🧠 Early-Stage Alzheimer's Detection")
st.markdown(
    "Upload a brain MRI scan (JPG/PNG) to classify it as **Non-Demented** "
    "or **Very Mild Demented**, with a Grad-CAM visualization explaining "
    "the model's decision."
)

uploaded_file = st.file_uploader(
    "Upload an MRI image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # --- Validate & load image ---
        raw_bytes = uploaded_file.read()
        pil_img = load_image(io.BytesIO(raw_bytes))
    except Exception:
        st.error(
            "⚠️ Could not read this file as an image. Please upload a "
            "valid JPG or PNG file."
        )
        st.stop()

    # --- Load model ---
    try:
        model, last_conv_layer_name = get_model()
    except Exception as e:
        st.error(f"⚠️ Failed to load the model: {e}")
        st.stop()

    # --- Preprocess ---
    resized_for_display = pil_img.resize(IMG_SIZE)
    original_uint8 = np.array(resized_for_display).astype(np.uint8)
    input_batch = prepare_input(pil_img)  # (1, 224, 224, 3), float32 in [0,1]

    # --- Predict ---
    with st.spinner("Running inference..."):
        prob = float(model.predict(input_batch, verbose=0)[0][0])
        pred_class = int(prob >= THRESHOLD)
        confidence = prob if pred_class == 1 else (1 - prob)

    # --- Grad-CAM ---
    with st.spinner("Generating Grad-CAM explanation..."):
        heatmap = make_gradcam_heatmap(input_batch, model, last_conv_layer_name)
        overlay_img = overlay_heatmap(original_uint8, heatmap, alpha=0.4)
        heatmap_resized = (
            np.uint8(255 * heatmap)
        )

    st.divider()

    # --- Prediction summary ---
    col_result, col_conf = st.columns([2, 1])
    with col_result:
        if pred_class == 1:
            st.error(f"### Prediction: {CLASS_NAMES[1]}")
        else:
            st.success(f"### Prediction: {CLASS_NAMES[0]}")
    with col_conf:
        st.metric("Confidence", f"{confidence * 100:.2f}%")

    st.progress(confidence)

    # --- Confidence bar chart ---
    st.subheader("📊 Prediction Probability")
    prob_data = {
        CLASS_NAMES[0]: 1 - prob,
        CLASS_NAMES[1]: prob,
    }
    st.bar_chart(prob_data)

    st.divider()

    # --- Image displays: original / heatmap / overlay ---
    st.subheader("🔍 Grad-CAM Visualization")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(original_uint8, caption="Original (Resized 224×224)", use_container_width=True)
    with col2:
        st.image(heatmap, caption="Grad-CAM Heatmap (raw)", use_container_width=True, clamp=True)
    with col3:
        st.image(overlay_img, caption="Overlay", use_container_width=True)

    with st.expander("ℹ️ What is Grad-CAM?"):
        st.markdown(
            """
            **Grad-CAM (Gradient-weighted Class Activation Mapping)** shows
            which regions of the MRI scan most strongly influenced the
            model's prediction. Warmer colors (red/yellow) indicate regions
            that pushed the model toward its predicted class, while cooler
            colors (blue) had less influence.

            This is an explainability tool — it does not confirm the
            prediction is medically correct, but helps visualize the
            model's reasoning.
            """
        )

    # --- Download button for the Grad-CAM overlay ---
    overlay_pil = Image.fromarray(overlay_img)
    buf = io.BytesIO()
    overlay_pil.save(buf, format="PNG")
    st.download_button(
        label="💾 Download Grad-CAM Overlay",
        data=buf.getvalue(),
        file_name="gradcam_overlay.png",
        mime="image/png",
    )

else:
    st.info("👆 Upload an MRI image above to get started.")
