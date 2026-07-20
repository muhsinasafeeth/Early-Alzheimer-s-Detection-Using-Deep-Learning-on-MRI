"""
Application configuration
"""

# App Information
APP_TITLE = "Early Stage Alzheimer's Detection"
APP_ICON = "🧠"

# Model
MODEL_PATH = "alzheimer_model.keras"

# Image Settings
IMG_SIZE = (224, 224)

# Binary Classification Threshold
THRESHOLD = 0.5

# Class Labels
CLASS_NAMES = {
    0: "Non Demented",
    1: "Very Mild Demented"
}

# Model Information
MODEL_INFO = {
    "Architecture": "Custom CNN",
    "Input Size": "224 × 224 RGB",
    "Output": "Binary Classification",
    "Activation": "Sigmoid",
    "Framework": "TensorFlow / Keras"
}
