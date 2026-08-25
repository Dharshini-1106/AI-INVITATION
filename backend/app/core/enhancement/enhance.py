"""Automatic image enhancement pipeline.

Steps applied based on quality analysis:
- CLAHE for contrast (dark/bright)
- Deblurring (Wiener/Unsharp) for blur
- Denoising (NLM) for noise
- Perspective correction for skew/rotation
- Super-resolution (OpenCV FSRCNN) for low resolution
"""
import logging
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_denoise(img: np.ndarray) -> np.ndarray:
    """Apply Non-Local Means denoising."""
    return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)


def apply_deblur(img: np.ndarray) -> np.ndarray:
    """Deblur using unsharp masking (sharpening convolution)."""
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp = cv2.filter2D(img, -1, kernel)
    # Blend slightly to avoid over-sharpening
    return cv2.addWeighted(img, 0.6, sharp, 0.4, 0)


def apply_super_resolution(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """Upscale using OpenCV FSRCNN if available; else simple bicubic resize."""
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        model_path = None
        # Try common cached FSRCNN path
        import os
        candidates = [
            "FSRCNN_x2.pb",
            "models/FSRCNN_x2.pb",
            os.path.join(os.path.dirname(__file__), "weights", "FSRCNN_x2.pb"),
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break
        if model_path is None:
            raise FileNotFoundError("FSRCNN model not present")
        sr.readModel(model_path)
        sr.setModel("fsrcnn", scale)
        return sr.upsample(img)
    except Exception as exc:  # noqa: BLE001
        logger.info("Super-resolution model unavailable, using bicubic resize (%s)", exc)
        h, w = img.shape[:2]
        return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def _find_largest_quadrilateral(img: np.ndarray) -> np.ndarray:
    """Detect the largest contour approximating a quadrilateral (the invitation card)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > img.shape[0] * img.shape[1] * 0.1:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def apply_perspective_correction(img: np.ndarray) -> np.ndarray:
    """Correct perspective/skew by detecting the card quadrilateral and warping."""
    quad = _find_largest_quadrilateral(img)
    if quad is None:
        logger.info("No quadrilateral detected; skipping perspective correction")
        return img

    # Order points: top-left, top-right, bottom-right, bottom-left
    def order(pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    rect = order(quad)
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, matrix, (max_width, max_height))


def enhance_image(img: np.ndarray, quality: dict) -> tuple[np.ndarray, List[str]]:
    """Apply enhancements based on the quality analysis dict.

    The transformation order follows the reference pipeline:
    CLAHE -> Perspective Correction -> Super Resolution, with optional
    denoise/deblur applied earliest. Every step is gated by the detected
    quality condition so it adapts to each image rather than applying a
    fixed recipe.

    Returns (enhanced_image, applied_enhancements_list).
    """
    result = img
    applied: List[str] = []

    # Lightest, most general corrections first (fix acquisition defects).
    if quality.get("is_noisy"):
        result = apply_denoise(result)
        applied.append("Denoise")

    if quality.get("is_blurred"):
        result = apply_deblur(result)
        applied.append("Deblur")

    # 1) CLAHE: normalize contrast so later geometric/upscale steps see a
    #    consistent intensity range.
    if quality.get("is_dark") or quality.get("is_bright"):
        result = apply_clahe(result)
        applied.append("CLAHE")

    # 2) Perspective correction: remove skew/rotation before resampling.
    if quality.get("is_rotated") or quality.get("is_skewed"):
        corrected = apply_perspective_correction(result)
        # Only keep if correction produced a meaningful change
        if corrected.shape[0] > 0 and corrected.shape[1] > 0:
            result = corrected
            applied.append("Perspective Correction")

    # 3) Super Resolution: upscale last so OCR sees the sharpest possible
    #    text pixels.
    if quality.get("low_resolution"):
        result = apply_super_resolution(result)
        applied.append("Super Resolution")

    return result, applied

