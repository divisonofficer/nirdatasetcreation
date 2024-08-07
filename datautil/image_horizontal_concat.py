from altair import value
import cv2
import numpy as np
from typing import Optional


def image_horizontal_concat(image_dict: dict):
    image: Optional[np.ndarray] = None
    for key, value in image_dict.items():
        if not isinstance(value, np.ndarray):
            value = value.cpu().numpy()
        if len(value.shape) != 3 or value.shape[2] != 3:
            value = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
        if value.dtype != np.uint8:
            value = (value * 255 / value.max()).astype(np.uint8)
        if image is None:
            image = value
        else:
            image = np.concatenate((image, value), axis=1)
    return image
