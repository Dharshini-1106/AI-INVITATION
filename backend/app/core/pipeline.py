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

    # Run OCR on the enhanced image. extract_text already performs lightweight
    # script detection and runs the correct PaddleOCR model (Tamil or English)
    # based on that detection, so only one OCR pass is needed per image.
    enhanced_ocr = _ocr_one(enhanced_img)

    # ------------------------------------------------------------------
    # Secondary OCR pass on the original (non-enhanced) image.
    #
    # Enhancements such as CLAHE / perspective correction can occasionally
    # suppress short text lines or change their shape enough for the
    # recogniser to miss them.  Running a second pass on the original
    # image gives us a fallback source of text without the cost of a
    # full rapidocr fallback.
    # ------------------------------------------------------------------
    original_ocr = None
    if settings.use_dual_ocr and enhanced_ocr is not None:
        try:
            original_ocr = _ocr_one(img)
        except Exception as exc:
            logger.warning("[%s OCR] Original-image OCR pass failed (%s)", label, exc)

    # Extract per-request OCR diagnostics from the primary OCR run.

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
    
    def _lines_with_source(ocr_result):
        """Return OCR lines together with their actual selected engine."""
        selected = getattr(ocr_result, "selected_lines", [])
        if selected and len(selected) == len(ocr_result.lines):
            return list(selected)
        return [(text, box, conf, "unknown")
                for text, box, conf in ocr_result.lines]

    # Use enhanced OCR lines directly — extract_text already handles language
    # detection, Tamil/English model routing, and fallbacks internally.
    enhanced_lines = _lines_with_source(enhanced_ocr)

    # Merge in lines from the original-image OCR pass.  This recovers text
    # that the enhancement pipeline may have suppressed (e.g. short headings
    # like "Party" or "AT 8 PM" on bright/rotated images).
    original_lines = []
    if original_ocr is not None:
        original_lines = _lines_with_source(original_ocr)

    # Combine both passes, preserving reading order from the enhanced pass
    # first and appending any extra lines from the original pass that were
    # not already present.
    seen_keys = set()
    merged_lines = []
    for text, box, conf, source_name in enhanced_lines:
        key = "".join(ch for ch in str(text).lower() if ch.isalnum())
        if key and key not in seen_keys:
            seen_keys.add(key)
            merged_lines.append((text, box, conf, source_name))
    for text, box, conf, source_name in original_lines:
        key = "".join(ch for ch in str(text).lower() if ch.isalnum())
        if key and key not in seen_keys:
            seen_keys.add(key)
            merged_lines.append((text, box, conf, source_name + "_original"))

    # Collect OCR engine provenance from both passes.
    ocr_engines = sorted({
        item.get("source", "unknown")
        for item in getattr(enhanced_ocr, "raw_results", [])
    } | {item.get("source", "unknown")
         for item in getattr(original_ocr, "raw_results", [])
         if original_ocr is not None} | {source_name for _, _, _, source_name in merged_lines}) or ["fallback"]

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
    ) if ocr_confidences else None
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
    if not final_ocr_engine or final_ocr_engine == "unknown":
        # Fallback to the merged engine list when the primary diagnostics
        # did not record a final engine explicitly.
        if ocr_engines:
            final_ocr_engine = ocr_engines[0]
        else:
            final_ocr_engine = "unknown"
    rapidocr_fallback_used = primary_diag.get("rapidocr_fallback_used", False)

    return {
        "event_name": parsed.get("event_name", ""),
        "event_type": parsed.get("event_type", ""),
        "bride_name": parsed.get("bride_name", ""),
        "groom_name": parsed.get("groom_name", ""),
        "date": parsed.get("date", ""),
        "time": parsed.get("time", ""),
        "end_time": parsed.get("end_time", ""),
        "venue": parsed.get("venue", ""),
        "address": parsed.get("address", ""),
        "contact_number": parsed.get("contact_number", ""),
        "additional_information": parsed.get("additional_information", ""),
        "birthday_age": parsed.get("birthday_age", ""),
        "printed_weekday": parsed.get("printed_weekday", ""),
        "language": language,
        "confidence_score": final_confidence,
        "number_of_events": parsed.get("number_of_events", 1),
        "invitation_mode": parsed.get("invitation_mode", "single"),
        "people": parsed.get("people", []),
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
