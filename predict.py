import numpy as np

from config import CLASS_NAMES, THRESHOLD


def predict_image(model, image):
    """
    Predict Alzheimer's stage from MRI image.

    Parameters
    ----------
    model : keras.Model

    image : np.ndarray
        Shape = (1,224,224,3)

    Returns
    -------
    dict
        prediction
        confidence
        probability
    """

    probability = float(model.predict(image, verbose=0)[0][0])

    if probability >= THRESHOLD:
        class_id = 1
    else:
        class_id = 0

    confidence = probability if class_id == 1 else 1 - probability

    return {
        "class_id": class_id,
        "prediction": CLASS_NAMES[class_id],
        "confidence": confidence,
        "probability": probability
    }
