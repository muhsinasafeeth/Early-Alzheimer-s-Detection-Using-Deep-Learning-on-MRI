import streamlit as st
import tensorflow as tf

from config import MODEL_PATH


@st.cache_resource
def load_model():
    """
    Load the trained CNN model.

    The model is cached so it is loaded only once,
    improving Streamlit performance.
    """

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
