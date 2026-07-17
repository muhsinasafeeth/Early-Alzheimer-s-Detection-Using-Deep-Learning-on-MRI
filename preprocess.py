"""
preprocess.py
-------------
Image loading and preprocessing utilities for the Alzheimer's Detection model.

IMPORTANT: This mirrors the EXACT preprocessing used during training
(see notebook cells 24 & 27):
    - Convert to RGB
    - Resize to 224 x 224
    - Scale pixel values to [0, 1]  (divide by 255.0)
    - No mean/std normalization was used during training, so none is applied here.
"""

import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)  # (width, height) as used by PIL.Image.resize


def load_image(file) -> Image.Image:
    """
    Load an image from a file path or a file-like object (e.g. Streamlit's
    UploadedFile) and force it to RGB (matches training, which used
    Image.open(path).convert("RGB")).
    """
    img = Image.open(file)
    img = img.convert("RGB")
    return img


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Resize + normalize a PIL image to match the training pipeline.

    Returns a float32 array of shape (224, 224, 3) in range [0, 1].
    """
    img_resized = img.resize(IMG_SIZE)
    arr = np.array(img_resized).astype(np.float32) / 255.0
    return arr


def prepare_input(img: Image.Image) -> np.ndarray:
    """
    Full pipeline: resize/normalize + add batch dimension.

    Returns a float32 array of shape (1, 224, 224, 3), ready to feed
    directly into model.predict(...).
    """
    arr = preprocess_image(img)
    batched = np.expand_dims(arr, axis=0)
    return batched
