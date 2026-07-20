import numpy as np
from PIL import Image

from config import IMG_SIZE


def preprocess_image(uploaded_file):
    """
    Preprocess uploaded MRI image.

    Steps:
    1. Convert to RGB
    2. Resize to 224x224
    3. Normalize to [0,1]
    4. Add batch dimension

    Returns
    -------
    image_array : np.ndarray
        Shape: (1, 224, 224, 3)

    display_image : PIL.Image
        Original image for Streamlit display.
    """

    image = Image.open(uploaded_file).convert("RGB")

    display_image = image.copy()

    image = image.resize(IMG_SIZE)

    image = np.array(image, dtype=np.float32)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image, display_image
