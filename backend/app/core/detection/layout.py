"""Invitation layout analysis.

Primary: DocLayout-YOLO (if weights available) to identify regions like:
title, bride name, groom name, event name, reception, wedding, date, time,
venue, address, contact number, footer.

Fallback: rule-based region segmentation using image processing heuristics.
"""
import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Expected layout regions (sections the parser understands)
LAYOUT_REGIONS = [
    "title",
    "bride_name",
    "groom_name",
    "event_name",
    "reception",
    "wedding",
    "date",
    "time",
    "venue",
    "address",
    "contact_number",
    "footer",
]

_doclayout_model = None
_doclayout_available = False


def _get_doclayout():
    """Lazily load DocLayout-YOLO model if available."""
    global _doclayout_model, _doclayout_available
    if _doclayout_model is not None:
        return _doclayout_model
    try:
        from doclayout_yolo import YOLOv10

        # Attempt to load a cached model; if missing, fall back to heuristics.
        import os
        model_path = os.environ.get("DOCLAYOUT_MODEL", "doclayout_yolo_docstructbench_imgsz1024.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError("DocLayout-YOLO weights not present")
        _doclayout_model = YOLOv10(model_path)
        _doclayout_available = True
        logger.info("DocLayout-YOLO loaded")
    except Exception as exc:  # noqa: BLE001
        logger.warning("DocLayout-YOLO unavailable (%s); using rule-based layout", exc)
        _doclayout_available = False
        _doclayout_model = None
    return _doclayout_model


def _rule_based_layout(img: np.ndarray) -> List[Dict]:
    """Segment the image into horizontal bands and label them heuristically."""
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 10))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bh < 20 or bw < w * 0.2:
            continue
        # Label by vertical position (top = title/header, bottom = footer)
        rel_y = y / float(h)
        if rel_y < 0.15:
            label = "title"
        elif rel_y > 0.85:
            label = "footer"
        else:
            label = "text_region"
        regions.append({
            "label": label,
            "confidence": 0.5,
            "box": [x, y, bw, bh],
        })
    if not regions:
        regions.append({"label": "text_region", "confidence": 0.3,
                        "box": [0, 0, w, h]})
    return regions


def analyze_layout(img: np.ndarray) -> List[Dict]:
    """Analyze invitation layout and return labeled regions."""
    model = _get_doclayout()
    if model is not None:
        try:
            # Convert BGR->RGB for model
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb, conf=0.25, verbose=False)
            regions = []
            for res in results:
                boxes = res.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = _map_doclayout_class(cls)
                    regions.append({
                        "label": label,
                        "confidence": conf,
                        "box": [int(xyxy[0]), int(xyxy[1]),
                                int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])],
                    })
            if regions:
                return regions
        except Exception as exc:  # noqa: BLE001
            logger.warning("DocLayout prediction failed (%s); using rule-based", exc)
    return _rule_based_layout(img)


def _map_doclayout_class(cls: int) -> str:
    """Map DocLayout-YOLO class index to our region label."""
    mapping = {
        0: "title",
        1: "text_region",
        2: "footer",
        3: "image",
        4: "table",
        5: "author",
        6: "date",
    }
    return mapping.get(cls, "text_region")

