import base64

import cv2
import numpy as np


def encode_frame(image: np.ndarray, max_side: int) -> str:
    """Downscale so the longest side is <= max_side, then JPEG + base64 (cheap to send and to hash)."""
    scale = max_side / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("Could not JPEG-encode frame")
    return base64.b64encode(buffer.tobytes()).decode("ascii")
