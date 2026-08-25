"""Multilingual OCR engine.

Primary: PaddleOCR 3.7.0 (PP-OCRv6, multilingual, supports English + Indic scripts).
Fallback 1: RapidOCR (ONNX, PP-OCR-v4/v5 style models) — multilingual, lightweight.
Fallback 2: rule-based text extraction for environments where no OCR model is
installed.

Architecture:
    PaddleOCR -> usable result -> FINAL RESULT = PaddleOCR
    PaddleOCR -> exception/unusable/empty -> RapidOCR -> FINAL RESULT
"""
import logging
import os
import pathlib
import time
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Disable oneDNN BEFORE any paddle import to avoid Windows CPU inference crashes
# and to ensure stable inference on all machines.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("MKLDNN_CACHE_CAPACITY", "0")

# Location of the Tamil-capable recognition model (PP-OCRv3 Tamil + dictionary).
# Placed outside the venv so it survives dependency changes.
_MODELS_ROOT = pathlib.Path(__file__).resolve().parents[3] / "models" / "tamil"

# Unicode block for Tamil script.
_TAMIL_LO = 0x0B80
_TAMIL_HI = 0x0BFF

# Thresholds for lightweight script detection.
_DETECT_MAX_SIDE = 320
_DETECT_TAMIL_THRESHOLD = 3
_DETECT_ENGLISH_THRESHOLD = 5


def _has_tamil(text: str) -> bool:
    """Return True if *text* contains at least one Tamil Unicode codepoint."""
    return any(_TAMIL_LO <= ord(ch) <= _TAMIL_HI for ch in (text or ""))


def _tamil_char_count(text: str) -> int:
    return sum(1 for ch in (text or "") if _TAMIL_LO <= ord(ch) <= _TAMIL_HI)


def _latin_char_count(text: str) -> int:
    return sum(1 for ch in (text or "") if ch.isalpha() and ch.isascii())


def _resize_for_detection(image_path: str) -> str:
    """Downscale *image_path* to a small preview and return the temp path.

    The preview preserves aspect ratio and is used ONLY for lightweight
    script detection, never for final OCR.
    """
    import tempfile
    from PIL import Image

    img = Image.open(image_path)
    w, h = img.size
    scale = min(1.0, _DETECT_MAX_SIDE / max(w, h))
    if scale >= 1.0:
        return image_path
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.convert("RGB")
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(tmp_path, "PNG")
    return tmp_path


def _detect_script(image_path: str) -> tuple[str, float]:
    """Lightweight script detection using a downscaled image + Tamil PaddleOCR.

    Returns:
        (script, confidence) where script is one of:
        "tamil", "english", "mixed", "unknown".
        confidence is a float between 0.0 and 1.0.
    """
    tmp_path = None
    try:
        tmp_path = _resize_for_detection(image_path)
        ocr = _get_paddleocr(
            "ta",
            det_model_name="PP-OCRv5_mobile_det",
            rec_model_name="ta_PP-OCRv5_mobile_rec",
        )
        if ocr is None:
            return "unknown", 0.0
        result = list(ocr.predict(tmp_path))
        all_text = ""
        for page in result:
            try:
                payload = page.json if hasattr(page, "json") else page
                if isinstance(payload, str):
                    import json
                    payload = json.loads(payload)
                payload = payload.get("res", payload)
                texts = payload.get("rec_texts", [])
                all_text += " ".join(str(t) for t in texts)
            except Exception:
                continue
        tamil_chars = _tamil_char_count(all_text)
        latin_chars = _latin_char_count(all_text)
        total_chars = tamil_chars + latin_chars
        if total_chars == 0:
            return "unknown", 0.0
        if tamil_chars >= _DETECT_TAMIL_THRESHOLD and latin_chars >= _DETECT_ENGLISH_THRESHOLD:
            conf = max(tamil_chars, latin_chars) / total_chars
            return "mixed", round(conf, 2)
        if tamil_chars >= _DETECT_TAMIL_THRESHOLD:
            return "tamil", round(tamil_chars / total_chars, 2)
        if latin_chars >= _DETECT_ENGLISH_THRESHOLD:
            return "english", round(latin_chars / total_chars, 2)
        return "unknown", 0.0
    except Exception:
        return "unknown", 0.0
    finally:
        if tmp_path and tmp_path != image_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _maybe_repair_encoding(text: str) -> str:
    """Repair accidental latin-1 decoding of UTF-8 Tamil bytes.

    Some OCR/post-processing paths can double-encode Tamil text
    (UTF-8 bytes interpreted as latin-1).  If re-decoding the latin-1
    representation yields valid Tamil, use the repaired string.
    """
    if text is None:
        return text
    if _has_tamil(text):
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    # Only accept the repair when it actually introduces Tamil glyphs and
    # does not look like a UTF-8 decoding failure (replacement chars).
    if _has_tamil(repaired) and "\ufffd" not in repaired:
        return repaired
    return text

# Language codes mapped to Paddle OCR language codes.
PPOCR_LANGS = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "Malayalam": "ml",
    "Telugu": "te",
}

SUPPORTED_LANGS = ["en", "ta", "hi", "ml", "te"]


class OCRResult:
    """Container for OCR output: text lines with bounding boxes.

    Extended with debug fields so downstream code can access raw per-engine
    results, merged candidate groups, and an optionally corrected text view.
    """

    def __init__(self, text: str, lines: List[Tuple[str, np.ndarray, float]]):
        self.text = text
        self.lines = lines  # (text, box, confidence)
        # Debug-preserving fields (may be populated by the higher-level
        # extract_text implementation):
        # raw_results: list of {text, confidence, bbox, source}
        # merged: list of {bbox, candidates: [{text, confidence, source}], selected}
        # corrected_text: a corrected/normalized text view (string)
        # ocr_diagnostics: per-request OCR diagnostics dict
        self.raw_results = []
        self.merged = []
        # ``lines`` is intentionally kept backward compatible.  This parallel
        # list retains the engine selected for each line so callers do not have
        # to guess that every non-Tamil result came from RapidOCR.
        self.selected_lines = []
        self.corrected_text = text
        self.ocr_diagnostics = {}


# ---------------------------------------------------------------------------
# PaddleOCR (PP-OCRv5) — primary engine
# ---------------------------------------------------------------------------
_paddleocr = {}
_paddleocr_available = False


def _get_paddleocr(lang: str = "en", ocr_version: str = "PP-OCRv5", det_model_name: str = None, rec_model_name: str = None):
    """Lazily initialize PaddleOCR 3.7.0 with Windows CPU-safe settings.

    PaddleOCR 3.x uses ``predict`` rather than the removed 2.x ``ocr`` API.
    Model-host checks and oneDNN are disabled here because they respectively
    block first startup and fail during Windows CPU inference.  The settings
    are process-local and preserve RapidOCR as the independent fallback.
    """
    global _paddleocr, _paddleocr_available
    cache_key = f"{lang}:{ocr_version}:{det_model_name or 'default'}:{rec_model_name or 'default'}"
    if cache_key in _paddleocr:
        return _paddleocr[cache_key]
    try:
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", "D:/MINI/.ml-cache/paddle")
        os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "modelscope")
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("PADDLE_PDX_EAGER_INIT", "False")
        os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        os.environ.setdefault("MKLDNN_CACHE_CAPACITY", "0")
        from paddleocr import PaddleOCR

        options = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            text_det_limit_side_len=960,
        )
        if det_model_name or rec_model_name:
            if det_model_name:
                options["text_detection_model_name"] = det_model_name
            if rec_model_name:
                options["text_recognition_model_name"] = rec_model_name
        else:
            options["lang"] = lang
            options["ocr_version"] = ocr_version
        _paddleocr[cache_key] = PaddleOCR(**options)
        _paddleocr_available = True
        logger.info("PaddleOCR 3.7.0 initialized for %s (version=%s, det=%s, rec=%s)", lang, ocr_version, det_model_name or "default", rec_model_name or "default")
        return _paddleocr[cache_key]
    except Exception as exc:  # noqa: BLE001
        logger.warning("PaddleOCR unavailable (%s)", exc)
        _paddleocr_available = False
        _paddleocr.pop(cache_key, None)
        return None


def _ocr_with_paddleocr(image_path: str, lang: str = "en", det_model_name: str = None, rec_model_name: str = None) -> OCRResult:
    """Run PaddleOCR 3.7.0 and return backward-compatible structured lines."""
    if det_model_name is None and rec_model_name is None:
        if lang == "en":
            det_model_name = "PP-OCRv5_mobile_det"
            rec_model_name = "en_PP-OCRv5_mobile_rec"
        elif lang == "ta":
            det_model_name = "PP-OCRv5_mobile_det"
            rec_model_name = "ta_PP-OCRv5_mobile_rec"
    ocr = _get_paddleocr(lang, ocr_version="PP-OCRv5", det_model_name=det_model_name, rec_model_name=rec_model_name)
    if ocr is None:
        raise RuntimeError("PaddleOCR not available")
    result = list(ocr.predict(image_path))
    lines = []
    text_parts = []
    for page in result:
        try:
            payload = page.json if hasattr(page, "json") else page
            if isinstance(payload, str):
                import json
                payload = json.loads(payload)
            payload = payload.get("res", payload)
            texts = payload.get("rec_texts", [])
            boxes = payload.get("rec_polys", [])
            scores = payload.get("rec_scores", [])
            for text, box, confidence in zip(texts, boxes, scores):
                if not text:
                    continue
                box_arr = np.array(box, dtype=np.float32)
                lines.append((str(text), box_arr, float(confidence)))
                text_parts.append(str(text))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unable to parse PaddleOCR 3.7.0 output (%s)", exc)
    if not lines:
        raise RuntimeError("PaddleOCR returned no text")
    return OCRResult("\n".join(text_parts), lines)


# ---------------------------------------------------------------------------
# RapidOCR (ONNX) — fallback 1
# ---------------------------------------------------------------------------
_rapidocr = None
_rapidocr_available = False


def _get_rapidocr():
    """Lazily initialize the RapidOCR engine.

    RapidOCR uses bundled ONNX detection/recognition models that handle a broad
    set of scripts (including English, Tamil, Hindi, Malayalam, Telugu) with a
    single engine — there is no per-language engine to instantiate.
    """
    global _rapidocr, _rapidocr_available
    if _rapidocr is not None:
        return _rapidocr
    try:
        from rapidocr_onnxruntime import RapidOCR

        _rapidocr = RapidOCR()
        _rapidocr_available = True
        logger.info("RapidOCR initialized (multilingual ONNX)")
        return _rapidocr
    except Exception as exc:  # noqa: BLE001
        logger.warning("RapidOCR unavailable (%s); using rule-based OCR", exc)
        _rapidocr_available = False
        _rapidocr = None
        return None


def _ocr_with_rapidocr(image_path: str) -> OCRResult:
    """Run RapidOCR on an image file and return structured lines.

    RapidOCR returns: list of [ [box(4 points)], text, confidence ].
    """
    ocr = _get_rapidocr()
    if ocr is None:
        raise RuntimeError("RapidOCR not available")
    result, _elapsed = ocr(image_path)
    lines = []
    text_parts = []
    if result and isinstance(result, (list, tuple)):
        for item in result:
            try:
                box, text, conf = item[0], item[1], item[2]
                if not text:
                    continue
                box_arr = np.array(box, dtype=np.float32)
                lines.append((str(text), box_arr, float(conf)))
                text_parts.append(str(text))
            except Exception:  # noqa: BLE001
                continue
    if not lines:
        raise RuntimeError("RapidOCR returned no text")
    return OCRResult("\n".join(text_parts), lines)


# ---------------------------------------------------------------------------
# pytesseract fallback (lightweight OCR using Tesseract if installed)
# ---------------------------------------------------------------------------
_pytesseract = None
_pytesseract_available = False


def _get_pytesseract():
    """Lazily import pytesseract and confirm Tesseract is available on PATH."""
    global _pytesseract, _pytesseract_available
    if _pytesseract is not None:
        return _pytesseract
    try:
        import pytesseract
        from pytesseract import Output  # noqa: F401

        # Quick runtime check: call get_tesseract_version if available
        try:
            _ = pytesseract.get_tesseract_version()
        except Exception:
            # If Tesseract binary is missing, treat as unavailable
            raise
        _pytesseract = pytesseract
        _pytesseract_available = True
        logger.info("pytesseract/Tesseract OCR available")
        return _pytesseract
    except Exception as exc:  # noqa: BLE001
        logger.warning("pytesseract/Tesseract unavailable (%s)", exc)
        _pytesseract_available = False
        _pytesseract = None
        return None


def _ocr_with_pytesseract(image_path: str) -> OCRResult:
    """Run Tesseract via pytesseract and return structured lines.

    Uses image_to_data to obtain per-word bounding boxes and confidences and
    aggregates words into heuristic lines by their top coordinate.
    """
    pyt = _get_pytesseract()
    if pyt is None:
        raise RuntimeError("pytesseract not available")
    try:
        from PIL import Image
        from pytesseract import Output
        import numpy as _np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"pytesseract runtime error: {exc}")

    img = Image.open(image_path)
    # Use a conservative page segmentation mode
    config = "--psm 6"
    data = pyt.image_to_data(img, output_type=Output.DICT, config=config)
    n = len(data.get("text", []))
    lines = []
    text_lines_map = {}

    for i in range(n):
        txt = (data.get("text", [])[i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data.get("conf", [])[i])
        except Exception:
            conf = -1.0
        # Normalize confidence to 0.0-1.0 when possible (tesseract gives -1 for non-detected)
        conf_norm = max(0.0, conf) / 100.0 if conf >= 0 else 0.0
        left = int(data.get("left", [])[i] or 0)
        top = int(data.get("top", [])[i] or 0)
        width = int(data.get("width", [])[i] or 0)
        height = int(data.get("height", [])[i] or 0)
        # Group words by their 'top' coordinate rounded to 10 pixels to form lines
        top_key = int(round(top / 10.0) * 10)
        entry = {
            "text": txt,
            "conf": conf_norm,
            "box": _np.array([[left, top], [left + width, top], [left + width, top + height], [left, top + height]], dtype=_np.float32),
        }
        text_lines_map.setdefault(top_key, []).append(entry)

    text_parts = []
    for top_key in sorted(text_lines_map.keys()):
        parts = text_lines_map[top_key]
        # sort by left coordinate
        parts.sort(key=lambda e: int(e["box"][0][0]))
        line_text = " ".join(p["text"] for p in parts)
        # approximate line box by min/max of word boxes
        xs = [int(p["box"][0][0]) for p in parts] + [int(p["box"][1][0]) for p in parts]
        ys = [int(p["box"][0][1]) for p in parts] + [int(p["box"][2][1]) for p in parts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        box_arr = _np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=_np.float32)
        avg_conf = float(sum(p["conf"] for p in parts) / len(parts))
        lines.append((line_text, box_arr, avg_conf))
        text_parts.append(line_text)

    if not lines:
        raise RuntimeError("pytesseract returned no text")
    return OCRResult("\n".join(text_parts), lines)


# ---------------------------------------------------------------------------
# Tamil-capable OCR (PP-OCRv3 Tamil recognition + multilingual det/cls)
# ---------------------------------------------------------------------------
# The bundled RapidOCR recognition model is Chinese+English only and cannot
# emit Tamil glyphs, which is why Tamil invitations used to come back as Latin
# garbage ("Ootij...", "Bsumflcor Buygonn").  This engine reuses RapidOCR's
# language-agnostic text *detection* and *classification* models but swaps in a
# Tamil recognition model so the pipeline produces real Tamil Unicode.
_tamil_rapidocr = None
_tamil_rapidocr_available = False


def _get_tamil_rapidocr():
    """Lazily build a RapidOCR instance whose recognizer uses the Tamil model.

    Detection (text localization) and angle classification are script-agnostic,
    so we keep RapidOCR's default det/cls models and only replace the
    recognition head with the Tamil PP-OCRv3 model + its character dictionary.
    """
    global _tamil_rapidocr, _tamil_rapidocr_available
    if _tamil_rapidocr is not None:
        return _tamil_rapidocr
    model_path = _MODELS_ROOT / "model.onnx"
    dict_path = _MODELS_ROOT / "dict.txt"
    if not model_path.exists() or not dict_path.exists():
        logger.warning(
            "Tamil OCR model not found at %s; Tamil OCR disabled", _MODELS_ROOT
        )
        _tamil_rapidocr_available = False
        _tamil_rapidocr = None
        return None
    try:
        from rapidocr_onnxruntime import RapidOCR
        from rapidocr_onnxruntime.ch_ppocr_v3_rec.text_recognize import (
            TextRecognizer,
        )

        ocr = RapidOCR()
        tamil_cfg = {
            "model_path": str(model_path),
            "keys_path": str(dict_path),
            "rec_img_shape": [3, 48, 320],
            "rec_batch_num": 6,
            "use_cuda": False,
        }
        ocr.text_recognizer = TextRecognizer(tamil_cfg)
        _tamil_rapidocr = ocr
        _tamil_rapidocr_available = True
        logger.info("Tamil OCR engine initialized (Tamil PP-OCRv3 recognition)")
        return _tamil_rapidocr
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tamil OCR unavailable (%s)", exc)
        _tamil_rapidocr_available = False
        _tamil_rapidocr = None
        return None


def _ocr_with_tamilocr(image_path: str) -> OCRResult:
    """Run the Tamil-capable OCR engine and return structured lines.

    Only lines that actually contain Tamil script are kept so that English (or
    other-script) invitations are never polluted by Tamil-model misreads.
    """
    ocr = _get_tamil_rapidocr()
    if ocr is None:
        raise RuntimeError("Tamil OCR not available")
    result, _elapsed = ocr(image_path)
    lines = []
    text_parts = []
    if result and isinstance(result, (list, tuple)):
        for item in result:
            try:
                box, text, conf = item[0], item[1], item[2]
                text = _maybe_repair_encoding(str(text))
                if not text:
                    continue
                # Keep only genuine Tamil content; a stray Tamil char in an
                # otherwise-Latin misread is not useful and would otherwise
                # suppress the correct English reading during merge.
                if _tamil_char_count(text) < 2:
                    continue
                box_arr = np.array(box, dtype=np.float32)
                lines.append((text, box_arr, float(conf)))
                text_parts.append(text)
            except Exception:  # noqa: BLE001
                continue
    if not lines:
        raise RuntimeError("Tamil OCR returned no Tamil text")
    return OCRResult("\n".join(text_parts), lines)


# ---------------------------------------------------------------------------
# Rule-based fallback (simple contours)
# ---------------------------------------------------------------------------
def _rule_based_fallback(image_path: str) -> OCRResult:
    """Fallback OCR using simple heuristics (works offline, low accuracy).

    Detects text blocks via contours and returns them as pseudo-lines.

    """
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        return OCRResult("", [])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold to separate text
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 15)
    # Find text-line contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    text_parts = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < 8 or w < 20:
            continue
        # Region crop -> placeholder text (real OCR requires model)
        box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        lines.append(("[text-block]", box, 0.0))
        text_parts.append("[text-block]")
    if not text_parts:
        text_parts = ["[no text detected]"]
    return OCRResult("\n".join(text_parts), lines)


def _box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Approximate IoU between two 4-point boxes (assumes axis-aligned rects)."""
    # Convert to x0,y0,x1,y1
    x01, y01 = float(min(box1[:, 0])), float(min(box1[:, 1]))
    x11, y11 = float(max(box1[:, 0])), float(max(box1[:, 1]))
    x02, y02 = float(min(box2[:, 0])), float(min(box2[:, 1]))
    x12, y12 = float(max(box2[:, 0])), float(max(box2[:, 1]))
    xi0, yi0 = max(x01, x02), max(y01, y02)
    xi1, yi1 = min(x11, x12), min(y11, y12)
    inter_w = max(0.0, xi1 - xi0)
    inter_h = max(0.0, yi1 - yi0)
    inter = inter_w * inter_h
    area1 = max(0.0, (x11 - x01) * (y11 - y01))
    area2 = max(0.0, (x12 - x02) * (y12 - y02))
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def _merge_boxes(boxes: List[np.ndarray]) -> np.ndarray:
    xs = []
    ys = []
    for b in boxes:
        xs.extend([float(min(b[:, 0])), float(max(b[:, 0]))])
        ys.extend([float(min(b[:, 1])), float(max(b[:, 1]))])
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _normalize_text_for_agreement(s: str) -> str:
    return ''.join(ch for ch in s.lower() if ch.isalnum())


def extract_text(image_path: str, use_ppocr: bool = True, lang: str = "en",
                 use_rapidocr: bool = True, use_tamil_ocr: bool = True) -> OCRResult:
    """Extract text using PaddleOCR PRIMARY with Tamil support and RapidOCR FALLBACK.

    The engine tries PaddleOCR first.  If PaddleOCR:
      - cannot import,
      - cannot initialize,
      - throws an inference exception,
      - or returns genuinely unusable/empty OCR output,
    then RapidOCR is attempted.

    Tamil handling:
      - If PaddleOCR succeeds but produces zero Tamil characters, and Tamil OCR
        is enabled, PaddleOCR is re-run with lang='ta' to get proper Tamil Unicode.
      - If Tamil PaddleOCR produces more Tamil characters, its result is used.
      - RapidOCR is only used as fallback when PaddleOCR completely fails.

    Detailed per-request diagnostics are returned via the OCRResult object
    (``ocr_diagnostics`` dict) so the pipeline can log and expose:
      - PRIMARY ENGINE
      - PaddleOCR import / init / inference status
      - PaddleOCR line count and average confidence
      - Tamil Unicode character count
      - RapidOCR fallback trigger and exact reason
      - FINAL OCR ENGINE
    """
    raw_results = []  # list of {text, confidence, bbox, source}
    diagnostics = {
        "primary_engine": "PaddleOCR" if use_ppocr else "RapidOCR",
        "paddleocr_import_ok": False,
        "paddleocr_version": "N/A",
        "paddlepaddle_version": "N/A",
        "paddleocr_init_ok": False,
        "paddleocr_inference_ok": False,
        "paddleocr_lines": 0,
        "paddleocr_avg_confidence": 0.0,
        "paddleocr_character_count": 0,
        "tamil_character_count": 0,
        "has_tamil": False,
        "has_latin": False,
        "english_character_count": 0,
        "rapidocr_available": False,
        "rapidocr_fallback_used": False,
        "rapidocr_fallback_reason": "",
        "final_engine": "unknown",
        "tamil_model_name": "N/A",
        "tamil_dictionary_path": str(_MODELS_ROOT / "dict.txt") if (_MODELS_ROOT / "dict.txt").exists() else "N/A",
        "tamil_dictionary_has_unicode": False,
        "tamil_inference_ms": 0,
        "detector_model_name": "N/A",
        "script_detection": "unknown",
        "script_confidence": 0.0,
        "ocr_route": "unknown",
        "english_ocr_executed": False,
        "tamil_ocr_executed": False,
    }

    # Capture version info
    try:
        import paddle
        diagnostics["paddlepaddle_version"] = getattr(paddle, "__version__", "N/A")
    except Exception:
        pass
    try:
        import paddleocr
        diagnostics["paddleocr_version"] = getattr(paddleocr, "__version__", "N/A")
    except Exception:
        pass
    try:
        from rapidocr_onnxruntime import RapidOCR
        diagnostics["rapidocr_available"] = True
    except Exception:
        pass

    # Check Tamil dictionary Unicode presence (RapidOCR fallback dict)
    _tamil_dict_path = _MODELS_ROOT / "dict.txt"
    if _tamil_dict_path.exists():
        diagnostics["tamil_dictionary_path"] = str(_tamil_dict_path)
        try:
            with open(_tamil_dict_path, "r", encoding="utf-8") as _df:
                _dict_chars = _df.read()
            diagnostics["tamil_dictionary_has_unicode"] = any(
                0x0B80 <= ord(ch) <= 0x0BFF for ch in _dict_chars
            )
        except Exception:
            pass

    def _is_unusable(result: OCRResult) -> bool:
        """Return True if the OCR result is genuinely unusable."""
        if result is None:
            return True
        text = (result.text or "").strip()
        if not text:
            return True
        if not result.lines:
            return True
        alnum_chars = sum(1 for ch in text if ch.isalnum())
        if alnum_chars < 3:
            return True
        return False

    def _count_tamil(text: str) -> int:
        return sum(1 for ch in text if 0x0B80 <= ord(ch) <= 0x0BFF)

    def _count_latin(text: str) -> int:
        return sum(1 for ch in text if ch.isalpha() and ch.isascii())

    paddle_result = None
    rapid_result = None
    tamil_result = None
    fallback_used = False
    final_engine = "unknown"
    script_detected = "unknown"
    script_confidence = 0.0
    ocr_route = "unknown"

    # ------------------------------------------------------------------
    # Step 0: Lightweight script detection (runs on downscaled image).
    # ------------------------------------------------------------------
    if use_ppocr and use_tamil_ocr:
        try:
            script_detected, script_confidence = _detect_script(image_path)
            diagnostics["script_detection"] = script_detected
            diagnostics["script_confidence"] = script_confidence
            logger.info("SCRIPT DETECTION: %s (confidence=%.2f)", script_detected, script_confidence)
        except Exception as exc:
            logger.warning("Script detection failed (%s); using safe fallback", exc)
            script_detected = "unknown"
            script_confidence = 0.0
            diagnostics["script_detection"] = "unknown"

    if script_detected == "tamil":
        ocr_route = "Tamil"
        diagnostics["ocr_route"] = "Tamil"
        logger.info("OCR ROUTE: Tamil")
    elif script_detected == "english":
        ocr_route = "English"
        diagnostics["ocr_route"] = "English"
        logger.info("OCR ROUTE: English")
    elif script_detected == "mixed":
        ocr_route = "Mixed"
        diagnostics["ocr_route"] = "Mixed"
        logger.info("OCR ROUTE: Mixed")
    else:
        ocr_route = "Safe-Fallback"
        diagnostics["ocr_route"] = "Safe-Fallback"
        logger.info("OCR ROUTE: Safe-Fallback (unknown script)")

    # ------------------------------------------------------------------
    # Step 1: PaddleOCR PRIMARY (routed by script detection)
    # ------------------------------------------------------------------
    _tamil_ocr_start = None
    _tamil_ocr_end = None
    _english_ocr_start = None
    _english_ocr_end = None

    if use_ppocr:
        # --- Tamil-only path: skip English OCR entirely ---
        if script_detected == "tamil":
            logger.info("English OCR: SKIPPED")
            logger.info("Tamil OCR: EXECUTED")
            diagnostics["english_ocr_executed"] = False
            diagnostics["tamil_ocr_executed"] = True
            try:
                _tamil_ocr_start = time.time()
                ocr_ta = _ocr_with_paddleocr(
                    image_path,
                    lang="ta",
                    det_model_name="PP-OCRv5_mobile_det",
                    rec_model_name="ta_PP-OCRv5_mobile_rec",
                )
                _tamil_ocr_end = time.time()
                if not _is_unusable(ocr_ta):
                    paddle_result = ocr_ta
                    final_engine = "paddleocr-ta"
                    diagnostics["paddleocr_import_ok"] = True
                    diagnostics["paddleocr_init_ok"] = True
                    diagnostics["paddleocr_inference_ok"] = True
                    diagnostics["paddleocr_lines"] = len(ocr_ta.lines)
                    confs = [float(c) for _, _, c in ocr_ta.lines]
                    diagnostics["paddleocr_avg_confidence"] = (
                        round(sum(confs) / len(confs), 4) if confs else 0.0
                    )
                    diagnostics["tamil_model_name"] = "ta_PP-OCRv5_mobile_rec"
                    try:
                        rec_name = ocr_ta.paddlex_pipeline.text_rec_model.model_name
                        diagnostics["tamil_model_name"] = str(rec_name)
                    except Exception:
                        pass
                    try:
                        det_name = ocr_ta.paddlex_pipeline.text_det_model.model_name
                        diagnostics["detector_model_name"] = str(det_name)
                    except Exception:
                        pass
                    for (txt, box, conf) in ocr_ta.lines:
                        repaired = _maybe_repair_encoding(str(txt))
                        raw_results.append({
                            "text": repaired,
                            "confidence": float(conf),
                            "bbox": box.tolist(),
                            "source": "paddleocr-ta",
                        })
                    if _tamil_ocr_start and _tamil_ocr_end:
                        diagnostics["tamil_inference_ms"] = int(
                            (_tamil_ocr_end - _tamil_ocr_start) * 1000
                        )
                else:
                    diagnostics["rapidocr_fallback_reason"] = (
                        "Tamil PaddleOCR returned unusable/empty text"
                    )
            except Exception as exc:
                diagnostics["rapidocr_fallback_reason"] = f"Tamil PaddleOCR failed: {exc}"
                logger.warning("Tamil PaddleOCR failed (%s)", exc)

        # --- English-only path: skip Tamil OCR entirely ---
        elif script_detected == "english":
            logger.info("English OCR: EXECUTED")
            logger.info("Tamil OCR: SKIPPED")
            diagnostics["english_ocr_executed"] = True
            diagnostics["tamil_ocr_executed"] = False
            try:
                _english_ocr_start = time.time()
                ocr_en = _ocr_with_paddleocr(image_path, lang="en")
                _english_ocr_end = time.time()
                if ocr_en is not None and hasattr(ocr_en, "paddlex_pipeline"):
                    try:
                        det_name = ocr_en.paddlex_pipeline.text_det_model.model_name
                        diagnostics["detector_model_name"] = str(det_name)
                    except Exception:
                        pass
                diagnostics["paddleocr_import_ok"] = True
                diagnostics["paddleocr_init_ok"] = True
                diagnostics["paddleocr_inference_ok"] = True
                diagnostics["paddleocr_lines"] = len(ocr_en.lines)
                confs = [float(c) for _, _, c in ocr_en.lines]
                diagnostics["paddleocr_avg_confidence"] = (
                    round(sum(confs) / len(confs), 4) if confs else 0.0
                )
                if not _is_unusable(ocr_en):
                    paddle_result = ocr_en
                    final_engine = "paddleocr"
                    for (txt, box, conf) in ocr_en.lines:
                        raw_results.append({
                            "text": str(txt),
                            "confidence": float(conf),
                            "bbox": box.tolist(),
                            "source": "paddleocr",
                        })
                else:
                    diagnostics["rapidocr_fallback_reason"] = (
                        "English PaddleOCR returned unusable/empty text"
                    )
            except Exception as exc:
                diagnostics["rapidocr_fallback_reason"] = f"English PaddleOCR failed: {exc}"
                logger.warning("English PaddleOCR failed (%s)", exc)

        # --- Mixed path: run Tamil first (faster), then English ---
        elif script_detected == "mixed":
            logger.info("Tamil OCR: EXECUTED")
            logger.info("English OCR: EXECUTED")
            diagnostics["english_ocr_executed"] = True
            diagnostics["tamil_ocr_executed"] = True
            try:
                _tamil_ocr_start = time.time()
                ocr_ta = _ocr_with_paddleocr(
                    image_path,
                    lang="ta",
                    det_model_name="PP-OCRv5_mobile_det",
                    rec_model_name="ta_PP-OCRv5_mobile_rec",
                )
                _tamil_ocr_end = time.time()
                if not _is_unusable(ocr_ta):
                    paddle_result = ocr_ta
                    final_engine = "paddleocr-ta"
                    diagnostics["paddleocr_import_ok"] = True
                    diagnostics["paddleocr_init_ok"] = True
                    diagnostics["paddleocr_inference_ok"] = True
                    diagnostics["paddleocr_lines"] = len(ocr_ta.lines)
                    confs = [float(c) for _, _, c in ocr_ta.lines]
                    diagnostics["paddleocr_avg_confidence"] = (
                        round(sum(confs) / len(confs), 4) if confs else 0.0
                    )
                    diagnostics["tamil_model_name"] = "ta_PP-OCRv5_mobile_rec"
                    try:
                        rec_name = ocr_ta.paddlex_pipeline.text_rec_model.model_name
                        diagnostics["tamil_model_name"] = str(rec_name)
                    except Exception:
                        pass
                    try:
                        det_name = ocr_ta.paddlex_pipeline.text_det_model.model_name
                        diagnostics["detector_model_name"] = str(det_name)
                    except Exception:
                        pass
                    for (txt, box, conf) in ocr_ta.lines:
                        repaired = _maybe_repair_encoding(str(txt))
                        raw_results.append({
                            "text": repaired,
                            "confidence": float(conf),
                            "bbox": box.tolist(),
                            "source": "paddleocr-ta",
                        })
                    if _tamil_ocr_start and _tamil_ocr_end:
                        diagnostics["tamil_inference_ms"] = int(
                            (_tamil_ocr_end - _tamil_ocr_start) * 1000
                        )
            except Exception as exc:
                logger.warning("Tamil PaddleOCR failed (%s)", exc)
            # English OCR follows for mixed content
            try:
                _english_ocr_start = time.time()
                ocr_en = _ocr_with_paddleocr(image_path, lang="en")
                _english_ocr_end = time.time()
                if not _is_unusable(ocr_en):
                    for (txt, box, conf) in ocr_en.lines:
                        raw_results.append({
                            "text": str(txt),
                            "confidence": float(conf),
                            "bbox": box.tolist(),
                            "source": "paddleocr",
                        })
            except Exception as exc:
                logger.warning("English PaddleOCR failed (%s)", exc)

        # --- Unknown/uncertain: safe fallback (current behavior) ---
        else:
            logger.info("English OCR: EXECUTED (safe fallback)")
            logger.info("Tamil OCR: CONDITIONAL")
            diagnostics["english_ocr_executed"] = True
            diagnostics["tamil_ocr_executed"] = True
            try:
                _english_ocr_start = time.time()
                ocr_en = _ocr_with_paddleocr(image_path, lang=lang)
                _english_ocr_end = time.time()
                if ocr_en is not None and hasattr(ocr_en, "paddlex_pipeline"):
                    try:
                        det_name = ocr_en.paddlex_pipeline.text_det_model.model_name
                        diagnostics["detector_model_name"] = str(det_name)
                    except Exception:
                        pass
                diagnostics["paddleocr_import_ok"] = True
                diagnostics["paddleocr_init_ok"] = True
                diagnostics["paddleocr_inference_ok"] = True
                diagnostics["paddleocr_lines"] = len(ocr_en.lines)
                confs = [float(c) for _, _, c in ocr_en.lines]
                diagnostics["paddleocr_avg_confidence"] = (
                    round(sum(confs) / len(confs), 4) if confs else 0.0
                )
                if not _is_unusable(ocr_en):
                    paddle_result = ocr_en
                    final_engine = "paddleocr"
                    for (txt, box, conf) in ocr_en.lines:
                        raw_results.append({
                            "text": str(txt),
                            "confidence": float(conf),
                            "bbox": box.tolist(),
                            "source": "paddleocr",
                        })
                else:
                    diagnostics["rapidocr_fallback_reason"] = (
                        "PaddleOCR returned unusable/empty text"
                    )
            except Exception as exc:
                diagnostics["rapidocr_fallback_reason"] = f"PaddleOCR inference failed: {exc}"
                logger.warning("PaddleOCR failed (%s)", exc)
            # Conditional Tamil OCR for uncertain cases
            if paddle_result is not None and use_tamil_ocr and lang != "ta":
                en_text = paddle_result.text or ""
                en_tamil_count = _count_tamil(en_text)
                if en_tamil_count == 0:
                    try:
                        _tamil_ocr_start = time.time()
                        _ta_ocr_obj = _get_paddleocr(
                            lang="ta",
                            det_model_name="PP-OCRv5_mobile_det",
                            rec_model_name="ta_PP-OCRv5_mobile_rec",
                        )
                        _tamil_ocr_end = time.time()
                        if _ta_ocr_obj is not None:
                            diagnostics["tamil_model_name"] = "ta_PP-OCRv5_mobile_rec"
                            try:
                                rec_name = _ta_ocr_obj.paddlex_pipeline.text_rec_model.model_name
                                diagnostics["tamil_model_name"] = str(rec_name)
                            except Exception:
                                pass
                            try:
                                det_name = _ta_ocr_obj.paddlex_pipeline.text_det_model.model_name
                                diagnostics["detector_model_name"] = str(det_name)
                            except Exception:
                                pass
                        if _ta_ocr_obj is not None:
                            ocr_ta = _ocr_with_paddleocr(
                                image_path,
                                lang="ta",
                                det_model_name="PP-OCRv5_mobile_det",
                                rec_model_name="ta_PP-OCRv5_mobile_rec",
                            )
                        else:
                            ocr_ta = None
                        if not _is_unusable(ocr_ta):
                            ta_text = ocr_ta.text or ""
                            ta_tamil_count = _count_tamil(ta_text)
                            if ta_tamil_count > en_tamil_count:
                                paddle_result = ocr_ta
                                final_engine = "paddleocr-ta"
                                raw_results = []
                                for (txt, box, conf) in ocr_ta.lines:
                                    repaired = _maybe_repair_encoding(str(txt))
                                    tc = _count_tamil(repaired)
                                    if tc >= 2 or (float(conf) >= 0.85 and len(repaired) > 2):
                                        raw_results.append({
                                            "text": repaired,
                                            "confidence": float(conf),
                                            "bbox": box.tolist(),
                                            "source": "paddleocr-ta",
                                        })
                                diagnostics["paddleocr_lines"] = len(ocr_ta.lines)
                                confs = [float(c) for _, _, c in ocr_ta.lines]
                                diagnostics["paddleocr_avg_confidence"] = (
                                    round(sum(confs) / len(confs), 4) if confs else 0.0
                                )
                                if _tamil_ocr_start and _tamil_ocr_end:
                                    diagnostics["tamil_inference_ms"] = int(
                                        (_tamil_ocr_end - _tamil_ocr_start) * 1000
                                    )
                    except Exception as exc:
                        logger.warning("PaddleOCR Tamil failed (%s)", exc)

    # ------------------------------------------------------------------
    # Step 3: RapidOCR FALLBACK ONLY (only if PaddleOCR failed completely)
    # ------------------------------------------------------------------
    if paddle_result is None and use_rapidocr:
        diagnostics["rapidocr_fallback_used"] = True
        try:
            rapid_result = _ocr_with_rapidocr(image_path)
            final_engine = "rapidocr"
            for (txt, box, conf) in rapid_result.lines:
                raw_results.append({
                    "text": str(txt),
                    "confidence": float(conf),
                    "bbox": box.tolist(),
                    "source": "rapidocr",
                })
        except Exception as exc:  # noqa: BLE001
            logger.warning("RapidOCR fallback failed (%s)", exc)

    # ------------------------------------------------------------------
    # Step 4: Tamil-capable RapidOCR (only if everything else failed)
    # ------------------------------------------------------------------
    if not raw_results and use_tamil_ocr:
        try:
            tamil_result = _ocr_with_tamilocr(image_path)
            final_engine = "tamilocr"
            for (txt, box, conf) in tamil_result.lines:
                raw_results.append({
                    "text": _maybe_repair_encoding(str(txt)),
                    "confidence": float(conf),
                    "bbox": box.tolist(),
                    "source": "tamilocr",
                })
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tamil OCR fallback failed (%s)", exc)

    # ------------------------------------------------------------------
    # Step 5: Rule-based fallback when every OCR engine failed
    # ------------------------------------------------------------------
    if not raw_results:
        fallback = _rule_based_fallback(image_path)
        for (txt, box, conf) in fallback.lines:
            raw_results.append({
                "text": str(txt),
                "confidence": float(conf),
                "bbox": box.tolist(),
                "source": "fallback",
            })
        if final_engine == "unknown":
            final_engine = "fallback"

    # Compute character counts across all collected raw results
    all_text = " ".join(item["text"] for item in raw_results)
    diagnostics["tamil_character_count"] = _count_tamil(all_text)
    diagnostics["english_character_count"] = _count_latin(all_text)
    diagnostics["has_tamil"] = diagnostics["tamil_character_count"] > 0
    diagnostics["has_latin"] = diagnostics["english_character_count"] > 0
    diagnostics["paddleocr_character_count"] = sum(len(item.get("text", "")) for item in raw_results)

    # Determine whether RapidOCR was actually used as fallback
    fallback_used = any(item.get("source", "").startswith("rapidocr") for item in raw_results)
    diagnostics["rapidocr_fallback_used"] = fallback_used
    diagnostics["final_engine"] = final_engine

    # Log the critical diagnostics
    logger.info("OCR PRIMARY ENGINE: %s", diagnostics["primary_engine"])
    logger.info("PaddleOCR VERSION: %s", diagnostics["paddleocr_version"])
    logger.info("PaddlePaddle VERSION: %s", diagnostics["paddlepaddle_version"])
    logger.info("SCRIPT DETECTION: %s", diagnostics["script_detection"])
    logger.info("SCRIPT CONFIDENCE: %.2f", diagnostics["script_confidence"])
    logger.info("OCR ROUTE: %s", diagnostics["ocr_route"])
    logger.info("ENGLISH OCR: %s", "EXECUTED" if diagnostics["english_ocr_executed"] else "SKIPPED")
    logger.info("TAMIL OCR: %s", "EXECUTED" if diagnostics["tamil_ocr_executed"] else "SKIPPED")
    logger.info("PaddleOCR import: %s", diagnostics["paddleocr_import_ok"])
    logger.info("PaddleOCR init: %s", diagnostics["paddleocr_init_ok"])
    logger.info("PaddleOCR inference: %s", diagnostics["paddleocr_inference_ok"])
    logger.info("PaddleOCR lines: %d", diagnostics["paddleocr_lines"])
    logger.info("PaddleOCR avg confidence: %.4f", diagnostics["paddleocr_avg_confidence"])
    logger.info("PaddleOCR character count: %d", diagnostics["paddleocr_character_count"])
    logger.info("Tamil Unicode count: %d", diagnostics["tamil_character_count"])
    logger.info("English character count: %d", diagnostics["english_character_count"])
    logger.info("Has Tamil: %s", diagnostics["has_tamil"])
    logger.info("Has Latin: %s", diagnostics["has_latin"])
    logger.info("RapidOCR available: %s", diagnostics["rapidocr_available"])
    logger.info("RapidOCR fallback used: %s", diagnostics["rapidocr_fallback_used"])
    if diagnostics["rapidocr_fallback_reason"]:
        logger.info("RapidOCR fallback reason: %s", diagnostics["rapidocr_fallback_reason"])
    logger.info("FINAL OCR ENGINE: %s", diagnostics["final_engine"])
    logger.info("Tamil model name: %s", diagnostics["tamil_model_name"])
    logger.info("Tamil dictionary path: %s", diagnostics["tamil_dictionary_path"])
    logger.info("Tamil dictionary has Unicode: %s", diagnostics["tamil_dictionary_has_unicode"])
    logger.info("Tamil inference ms: %d", diagnostics["tamil_inference_ms"])
    logger.info("Detector model name: %s", diagnostics["detector_model_name"])

    # ------------------------------------------------------------------
    # Cluster raw_results into groups based on bbox IoU / spatial proximity.
    # ------------------------------------------------------------------
    clusters = []  # each cluster: {boxes: [...], items: [raw_item]}
    for item in raw_results:
        box = np.array(item["bbox"], dtype=np.float32)
        placed = False
        for c in clusters:
            rep_box = c["boxes"][0]
            iou = _box_iou(rep_box, box)
            if iou >= 0.2:
                c["boxes"].append(box)
                c["items"].append(item)
                placed = True
                break
            cx1 = float((rep_box[:, 0].min() + rep_box[:, 0].max()) / 2.0)
            cy1 = float((rep_box[:, 1].min() + rep_box[:, 1].max()) / 2.0)
            cx2 = float((box[:, 0].min() + box[:, 0].max()) / 2.0)
            cy2 = float((box[:, 1].min() + box[:, 1].max()) / 2.0)
            dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
            hrep = float(rep_box[:, 1].max() - rep_box[:, 1].min())
            if hrep > 0 and dist < max(20.0, hrep * 0.6):
                c["boxes"].append(box)
                c["items"].append(item)
                placed = True
                break
        if not placed:
            clusters.append({"boxes": [box], "items": [item]})

    merged = []
    for c in clusters:
        merged_box = _merge_boxes(c["boxes"]) if c["boxes"] else np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        candidates = []
        norm_map = {}
        for it in c["items"]:
            txt = it.get("text", "")
            conf = float(it.get("confidence", 0.0))
            src = it.get("source", "unknown")
            norm = _normalize_text_for_agreement(txt)
            candidates.append({"text": txt, "confidence": conf, "source": src})
            norm_map.setdefault(norm, 0)
            norm_map[norm] += 1
        best = None
        best_score = -1.0
        for cand in candidates:
            norm = _normalize_text_for_agreement(cand["text"])
            score = cand["confidence"] + 0.08 * norm_map.get(norm, 0)
            if _has_tamil(cand["text"]):
                score += 0.6
            if score > best_score:
                best_score = score
                best = cand
        merged.append({
            "bbox": merged_box.tolist(),
            "candidates": candidates,
            "selected": best or (candidates[0] if candidates else {"text": "", "confidence": 0.0, "source": ""}),
            "score": float(best_score),
        })

    def _cluster_key(m):
        b = np.array(m["bbox"], dtype=np.float32)
        top = float(b[:, 1].min())
        left = float(b[:, 0].min())
        return (top, left)

    merged.sort(key=_cluster_key)

    out_lines = []
    text_parts = []
    for m in merged:
        sel = m.get("selected", {})
        t = sel.get("text", "")
        conf = float(sel.get("confidence", 0.0))
        box_arr = np.array(m.get("bbox", [[0, 0], [0, 0], [0, 0], [0, 0]]), dtype=np.float32)
        out_lines.append((t, box_arr, conf))
        text_parts.append(t)

    merged_text = "\n".join(text_parts) if text_parts else ""

    result = OCRResult(merged_text, out_lines)
    result.raw_results = raw_results
    result.merged = merged
    result.selected_lines = [
        (
            m.get("selected", {}).get("text", ""),
            np.array(m.get("bbox", [[0, 0], [0, 0], [0, 0], [0, 0]]), dtype=np.float32),
            float(m.get("selected", {}).get("confidence", 0.0)),
            m.get("selected", {}).get("source", "unknown"),
        )
        for m in merged
    ]
    result.corrected_text = merged_text
    result.ocr_diagnostics = diagnostics

    return result
