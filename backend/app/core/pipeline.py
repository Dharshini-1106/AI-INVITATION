"""End-to-end invitation understanding pipeline orchestration.

Reference flow:
    Image -> BRISQUE -> CLAHE -> Perspective Correction -> Super Resolution
    -> DocLayout-YOLO -> PaddleOCR -> LayoutLMv3 -> NER (PERSON/DATE/TIME/
    LOCATION/ORGANIZATION) -> Sentence-BERT (match to fields) -> parser.py
    (validate & format only).

Every model stage has a graceful fallback so the pipeline works regardless of
which heavy models are installed. Heavy model stages are gated by config
toggles (``settings.use_*``) so the pipeline runs fast in lightweight/offline
mode using rule-based fallbacks.
"""
import logging
import tempfile
import time
from pathlib import Path
from typing import List

import cv2

from ..config import settings
from .quality.brisque import analyze_quality
from .enhancement.enhance import enhance_image
from .detection.layout import analyze_layout
from .ocr.ocreader import extract_text
from .language import detect_language
from .postprocess.correction import correct_text
from .entity import extract_entities
from .matching import match_entities_to_fields
from .understanding.parser import parse_invitation

logger = logging.getLogger(__name__)

# Pipeline stage descriptions (used by the frontend progress display)
PIPELINE_STAGES = [
    "Analyzing image quality (BRISQUE)",
    "Enhancing image (CLAHE/Perspective/Super-Resolution)",
    "Detecting invitation layout (DocLayout-YOLO)",
    "Performing multilingual OCR",
    "Extracting entities (PERSON/DATE/TIME/LOCATION/ORGANIZATION)",
    "Matching entities to fields (Sentence-BERT)",
    "Validating & formatting output",
]


def _entities_to_fields(entities) -> dict:
    """Convert extracted entity objects to a simple field-mapping dict."""
    return match_entities_to_fields(entities)


def run_pipeline(image_bytes: bytes, filename: str, source: str = "gallery") -> dict:
    """Run the full invitation understanding pipeline on uploaded image bytes."""
    notes: List[str] = []
    from ..utils.image_utils import load_image_bytes

    img = load_image_bytes(image_bytes, filename)
    label = "CAMERA" if source == "camera" else "GALLERY"
    height, width = img.shape[:2]
    logger.info("[%s IMAGE] width=%d height=%d", label, width, height)

    # ------------------------------------------------------------------
    # Step 1: Quality analysis (BRISQUE)
    # ------------------------------------------------------------------
    quality = analyze_quality(img)
    if quality["needs_enhancement"]:
        notes.append(
            "Low quality detected: "
            + ", ".join(k for k in [
                "Blur" if quality["is_blurred"] else "",
                "Noise" if quality["is_noisy"] else "",
                "Dark" if quality["is_dark"] else "",
                "Bright" if quality["is_bright"] else "",
                "Rotated" if quality["is_rotated"] else "",
                "Low Resolution" if quality["low_resolution"] else "",
            ] if k)
        )

    # ------------------------------------------------------------------
    # Step 2: Enhancement (CLAHE -> Perspective -> Super Resolution)
    # ------------------------------------------------------------------
    enhanced_img, applied = enhance_image(img, quality)
    quality["applied_enhancements"] = applied
    if applied:
        notes.append("Applied enhancements: " + ", ".join(applied))

    # ------------------------------------------------------------------
    # Step 3: Layout analysis (DocLayout-YOLO or rule-based)
    # ------------------------------------------------------------------
    layout_regions = analyze_layout(enhanced_img)

    # ------------------------------------------------------------------
    # Step 4: OCR (PaddleOCR/RapidOCR primary with fallbacks)
    # ------------------------------------------------------------------
    def _ocr_one(img_arr, use_ppocr=None, use_rapidocr=None, use_tamil_ocr=None, lang="en"):
        """Run OCR on a single image array and return OCRResult."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cv2.imwrite(tmp_path, img_arr)
            return extract_text(
                str(tmp_path),
                use_ppocr=settings.use_ocr_ppocr if use_ppocr is None else use_ppocr,
                lang=lang,
                use_rapidocr=settings.use_rapidocr if use_rapidocr is None else use_rapidocr,
                use_tamil_ocr=settings.use_tamil_ocr if use_tamil_ocr is None else use_tamil_ocr,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # Run OCR on the enhanced image (single pass by default). When
    # use_dual_ocr is enabled, also run on the original and merge, which
    # catches more text but is slower.
    enhanced_ocr = _ocr_one(enhanced_img)

    # Extract per-request OCR diagnostics from the primary OCR run.
    enhanced_diag = getattr(enhanced_ocr, "ocr_diagnostics", {})
    if enhanced_diag:
        logger.info("[%s OCR] PRIMARY ENGINE: %s", label, enhanced_diag.get("primary_engine", "unknown"))
        logger.info("[%s OCR] PaddleOCR VERSION: %s", label, enhanced_diag.get("paddleocr_version", "N/A"))
        logger.info("[%s OCR] PaddlePaddle VERSION: %s", label, enhanced_diag.get("paddlepaddle_version", "N/A"))
        logger.info("[%s OCR] PaddleOCR import: %s", label, enhanced_diag.get("paddleocr_import_ok", False))
        logger.info("[%s OCR] PaddleOCR init: %s", label, enhanced_diag.get("paddleocr_init_ok", False))
        logger.info("[%s OCR] PaddleOCR inference: %s", label, enhanced_diag.get("paddleocr_inference_ok", False))
        logger.info("[%s OCR] PaddleOCR lines: %d", label, enhanced_diag.get("paddleocr_lines", 0))
        logger.info("[%s OCR] PaddleOCR avg confidence: %.4f", label, enhanced_diag.get("paddleocr_avg_confidence", 0.0))
        logger.info("[%s OCR] PaddleOCR character count: %d", label, enhanced_diag.get("paddleocr_character_count", 0))
        logger.info("[%s OCR] Tamil Unicode count: %d", label, enhanced_diag.get("tamil_character_count", 0))
        logger.info("[%s OCR] English character count: %d", label, enhanced_diag.get("english_character_count", 0))
        logger.info("[%s OCR] Has Tamil: %s", label, enhanced_diag.get("has_tamil", False))
        logger.info("[%s OCR] Has Latin: %s", label, enhanced_diag.get("has_latin", False))
        logger.info("[%s OCR] RapidOCR available: %s", label, enhanced_diag.get("rapidocr_available", False))
        logger.info("[%s OCR] RapidOCR fallback used: %s", label, enhanced_diag.get("rapidocr_fallback_used", False))
        if enhanced_diag.get("rapidocr_fallback_reason"):
            logger.info("[%s OCR] RapidOCR fallback reason: %s", label, enhanced_diag.get("rapidocr_fallback_reason"))
        logger.info("[%s OCR] FINAL OCR ENGINE: %s", label, enhanced_diag.get("final_engine", "unknown"))
        logger.info("[%s OCR] Tamil model: %s", label, enhanced_diag.get("tamil_model_name", "N/A"))
        logger.info("[%s OCR] Tamil dict path: %s", label, enhanced_diag.get("tamil_dictionary_path", "N/A"))
        logger.info("[%s OCR] Tamil dict has Unicode: %s", label, enhanced_diag.get("tamil_dictionary_has_unicode", False))
        logger.info("[%s OCR] Tamil inference ms: %d", label, enhanced_diag.get("tamil_inference_ms", 0))
        logger.info("[%s OCR] Detector model: %s", label, enhanced_diag.get("detector_model_name", "N/A"))

    # Determine whether the enhanced OCR already used Tamil mode.
    # If so, we can skip the English->Tamil fallback on subsequent passes.
    _enhanced_used_tamil = enhanced_diag.get("final_engine", "") == "paddleocr-ta"

    def _lines_with_source(ocr_result):
        """Return OCR lines together with their actual selected engine."""
        selected = getattr(ocr_result, "selected_lines", [])
        if selected and len(selected) == len(ocr_result.lines):
            return list(selected)
        return [(text, box, conf, "unknown")
                for text, box, conf in ocr_result.lines]

    # A small second OCR view improves recall for fine invitation text without
    # changing the main enhancement pipeline or adding another model.
    # If Tamil was already detected in the enhanced pass, run the upscaled
    # pass directly in Tamil mode to avoid a redundant English->Tamil double
    # pass (which doubles inference time on CPU).
    ocr_scale = 2
    upscaled_for_ocr = cv2.resize(
        enhanced_img, None, fx=ocr_scale, fy=ocr_scale,
        interpolation=cv2.INTER_CUBIC,
    )
    upscaled_lang = "ta" if _enhanced_used_tamil else "en"
    upscaled_ocr = _ocr_one(upscaled_for_ocr, lang=upscaled_lang)
    upscaled_lines = [
        (text, box / ocr_scale, conf, source_name)
        for text, box, conf, source_name in _lines_with_source(upscaled_ocr)
    ]
    if settings.use_dual_ocr:
        original_ocr = _ocr_one(img, lang=upscaled_lang)
        merged_lines = (
            _lines_with_source(enhanced_ocr) + upscaled_lines
            + _lines_with_source(original_ocr)
        )
    else:
        merged_lines = _lines_with_source(enhanced_ocr) + upscaled_lines

    # ------------------------------------------------------------------
    # Controlled Tamil OCR: only run when preliminary OCR shows Tamil
    # script, and run on multiple preprocessing variants for robustness.
    # ------------------------------------------------------------------
    def _has_tamil_in_lines(lines):
        for text, _, _, _ in lines:
            if any(0x0B80 <= ord(ch) <= 0x0BFF for ch in str(text)):
                return True
        return False

    def _otsu_binarize(img_arr):
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY) if len(img_arr.shape) == 3 else img_arr
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR) if len(img_arr.shape) == 3 else thresh

    if _has_tamil_in_lines(merged_lines) and not _enhanced_used_tamil:
        tamil_variants = []
        for variant_img, variant_name in [
            (img, "original"),
            (enhanced_img, "enhanced"),
            (_otsu_binarize(img), "otsu"),
        ]:
            try:
                res = _ocr_one(variant_img, use_ppocr=False,
                              use_rapidocr=False, use_tamil_ocr=True, lang="ta")
                tamil_variants.extend(_lines_with_source(res))
            except Exception as exc:
                logger.warning("Tamil OCR variant %s failed (%s)", variant_name, exc)
        merged_lines = merged_lines + tamil_variants

    # Keep OCR provenance visible for real-device diagnostics. The raw text is
    # already returned in the API result; this request-scoped log makes it
    # possible to compare Gallery and Camera input without changing either
    # upload path.
    ocr_engines = sorted({
        item.get("source", "unknown")
        for ocr_result in (enhanced_ocr, upscaled_ocr)
        for item in getattr(ocr_result, "raw_results", [])
    } | {source_name for _, _, _, source_name in merged_lines}) or ["fallback"]

    seen = set()
    unique_lines = []
    for text, box, conf, source_name in merged_lines:
        key = "".join(ch for ch in str(text).lower() if ch.isalnum())
        if not key or key in seen:
            continue
        seen.add(key)
        unique_lines.append((text, box, conf, source_name))

    # Build a combined text blob (preserve order of first appearance).
    text_parts = []
    for text, _, _, _ in unique_lines:
        if str(text).strip():
            text_parts.append(str(text).strip())
    raw_text = "\n".join(text_parts)
    tamil_character_count = sum(0x0B80 <= ord(ch) <= 0x0BFF for ch in raw_text)
    english_character_count = sum(
        ("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in raw_text
    )
    ocr_confidences = [float(conf) for _, _, conf, _ in unique_lines]
    ocr_confidence = round(
        sum(ocr_confidences) / len(ocr_confidences), 4
    ) if ocr_confidences else 0.0
    ocr_layout = [
        {
            "text": str(text).strip(),
            "bbox": box.tolist(),
            "confidence": float(conf),
            "source": src,
        }
        for text, box, conf, src in unique_lines
        if str(text).strip()
    ]

    if not raw_text:
        notes.append("No text detected in image")
    notes.append("OCR engine(s): " + ", ".join(ocr_engines))
    logger.info("[%s OCR] OCR engine(s): %s", label, ", ".join(ocr_engines))
    logger.info("[%s OCR] Number of OCR lines: %d", label, len(ocr_layout))
    logger.info("[%s OCR] Raw OCR text:\n%s", label, raw_text or "[no text detected]")

    # ------------------------------------------------------------------
    # Step 5: Language detection
    # ------------------------------------------------------------------
    language = detect_language(raw_text)
    notes.append(f"Detected language: {language}")

    # ------------------------------------------------------------------
    # Step 6: OCR correction (Sentence-BERT or dictionary fallback)
    # ------------------------------------------------------------------
    corrected_text = correct_text(raw_text) if raw_text else raw_text
    logger.info("[%s NORMALIZATION] OCR text after conservative normalization:\n%s",
                label, corrected_text or "[no text detected]")

    # ------------------------------------------------------------------
    # Step 7: Entity extraction (multilingual NER + rule-based fallback)
    # ------------------------------------------------------------------
    text_lines = [l for l in corrected_text.split("\n") if l.strip()]
    entities = extract_entities(corrected_text, text_lines, layout_regions)

    # ------------------------------------------------------------------
    # Step 8: Match entities to fields (Sentence-BERT or rule-based)
    # ------------------------------------------------------------------
    matched_fields = _entities_to_fields(entities)
    logger.info("[%s PARTICIPANT DEBUG] Upstream field hints: bride=%r groom=%r",
                label, matched_fields.get("bride_name", ""),
                matched_fields.get("groom_name", ""))

    # ------------------------------------------------------------------
    # Step 9: Validate & format (parser.py)
    # ------------------------------------------------------------------
    parsed = parse_invitation(
        matched_fields,
        corrected_text,
        layout_regions,
        ocr_lines=ocr_layout,
    )
    logger.info("[%s FINAL PARSER OUTPUT] bride_name=%r groom_name=%r",
                label, parsed.get("bride_name", ""), parsed.get("groom_name", ""))
    logger.info("[%s OCR] Final parsed extraction: %s", label, {
        key: parsed.get(key, "") for key in (
            "event_name", "event_type", "bride_name", "groom_name",
            "date", "time", "venue", "address",
        )
    })

    # Final confidence score is an average of quality and parse confidence.
    parse_conf = parsed.get("confidence", 0.0)
    quality_score = quality["score"] / 100.0
    final_confidence = round(0.4 * quality_score + 0.6 * parse_conf, 2)

    # Determine the authoritative final engine and fallback status from the
    # primary (enhanced) OCR run diagnostics, not from the merged engine list.
    primary_diag = getattr(enhanced_ocr, "ocr_diagnostics", {})
    final_ocr_engine = primary_diag.get("final_engine", "unknown")
    rapidocr_fallback_used = primary_diag.get("rapidocr_fallback_used", False)

    return {
        "event_name": parsed.get("event_name", ""),
        "event_type": parsed.get("event_type", ""),
        "bride_name": parsed.get("bride_name", ""),
        "groom_name": parsed.get("groom_name", ""),
        "date": parsed.get("date", ""),
        "time": parsed.get("time", ""),
        "venue": parsed.get("venue", ""),
        "address": parsed.get("address", ""),
        "contact_number": parsed.get("contact_number", ""),
        "language": language,
        "confidence_score": final_confidence,
        "number_of_events": parsed.get("number_of_events", 1),
        "events": parsed.get("events", []),
        "quality": quality,
        "raw_text": raw_text,
        "ocr_layout": ocr_layout,
        "ocr_engine": final_ocr_engine,
        "ocr_confidence": ocr_confidence,
        "tamil_character_count": tamil_character_count,
        "english_character_count": english_character_count,
        "fallback_used": rapidocr_fallback_used,
        "processing_notes": notes,
    }
