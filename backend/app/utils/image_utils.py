"""Image loading, orientation normalization and basic helpers."""
import base64
import io
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Standard RU/exif orientation tags
EXIF_ORIENTATION = 274


def load_image_bytes(data: bytes, filename: str = "upload") -> np.ndarray:
    """Load an image from raw bytes into a BGR numpy array, normalizing EXIF orientation.

    Falls back to PIL for formats OpenCV cannot decode directly (e.g. PDF first page).
    """
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("OpenCV could not decode image")
        img = _normalize_exif(img, data)
        return img
    except Exception as exc:  # noqa: BLE001
        logger.info("OpenCV decode failed (%s), trying PIL fallback", exc)
        pil = Image.open(io.BytesIO(data))
        pil = ImageOps.exif_transpose(pil)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return img


def _normalize_exif(img: np.ndarray, data: bytes) -> np.ndarray:
    """Normalize image orientation based on EXIF metadata."""
    try:
        pil = Image.open(io.BytesIO(data))
        exif = pil.getexif()
        orientation = exif.get(EXIF_ORIENTATION, 1)
        if orientation == 3:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif orientation == 6:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 8:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:  # noqa: BLE001
        pass
    return img


def to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to BGR."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def resize_to_max(img: np.ndarray, max_side: int = 2000) -> np.ndarray:
    """Resize the image so its longest side is at most max_side (preserve aspect)."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_bgr(img: np.ndarray) -> str:
    """Encode a BGR image to a base64 JPEG string."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def save_image(img: np.ndarray, path: Path) -> Path:
    """Persist an image to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path

