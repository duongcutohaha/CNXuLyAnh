from __future__ import annotations

import cv2


def preprocess_image(image, blur_kernel: int = 5, scale: float = 1.0):
    if scale != 1.0:
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    if blur_kernel % 2 == 0:
        blur_kernel += 1

    blurred = cv2.GaussianBlur(enhanced, (blur_kernel, blur_kernel), 0)
    return blurred
