"""Tamil OCR Diagnostic - Full Candidate Extraction.

Runs Tamil OCR on three preprocessing variants and saves structured results.
"""
import sys
import os
import json
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.ocr.ocreader import _ocr_with_tamilocr, _get_tamil_rapidocr
from app.core.quality.brisque import analyze_quality
from app.core.enhancement.enhance import enhance_image

IMAGE_PATH = Path(__file__).parent / "tamil" / "_invitation.png"
OUT_DIR = Path(__file__).parent / "tamil"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUT = OUT_DIR / "_ocr_candidates.json"
TXT_OUT = OUT_DIR / "_ocr_candidates.txt"
MOJIBAKE_TEST = OUT_DIR / "_mojibake_test.txt"

TAMIL_LO = 0x0B80
TAMIL_HI = 0x0BFF
LATIN_LO = 0x0041
LATIN_HI = 0x007A


def has_tamil(text: str) -> bool:
    return any(TAMIL_LO <= ord(ch) <= TAMIL_HI for ch in (text or ""))


def has_latin(text: str) -> bool:
    return any(LATIN_LO <= ord(ch) <= LATIN_HI for ch in (text or ""))


def tamil_char_count(text: str) -> int:
    return sum(1 for ch in (text or "") if TAMIL_LO <= ord(ch) <= TAMIL_HI)


def repair_mojibake(text: str) -> str:
    """Repair accidental latin-1 decoding of UTF-8 Tamil bytes.

    1. If text already contains valid Tamil Unicode, return unchanged.
    2. Only attempt latin-1 -> UTF-8 repair when text strongly resembles
       UTF-8 mojibake (contains bytes in 0x80-0xFF range).
    3. Validate the repaired result contains Tamil.
    4. If repair fails, return original.
    """
    if not text:
        return text
    if has_tamil(text):
        return text
    try:
        encoded = text.encode("latin-1")
    except UnicodeEncodeError:
        return text
    has_high_bytes = any(b >= 0x80 for b in encoded)
    if not has_high_bytes:
        return text
    try:
        repaired = encoded.decode("utf-8")
    except UnicodeDecodeError:
        return text
    if has_tamil(repaired) and "\ufffd" not in repaired:
        return repaired
    return text


def _save_temp(arr: np.ndarray) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, arr)
    tmp.close()
    return tmp.name


def _tamil_ocr_on_array(img_arr: np.ndarray):
    tmp_path = _save_temp(img_arr)
    try:
        result = _ocr_with_tamilocr(tmp_path)
        return result
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _otsu_binarize(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def run_tamil_diagnostic():
    print(f"Loading image: {IMAGE_PATH}")
    if not IMAGE_PATH.exists():
        print(f"ERROR: Image not found at {IMAGE_PATH}")
        sys.exit(1)

    original = cv2.imread(str(IMAGE_PATH))
    if original is None:
        print("ERROR: cv2.imread failed")
        sys.exit(1)

    print(f"Original size: {original.shape[1]}x{original.shape[0]}")

    tamil_ocr = _get_tamil_rapidocr()
    if tamil_ocr is None:
        print("ERROR: Tamil OCR engine not available")
        sys.exit(1)
    print("Tamil OCR engine: READY")

    quality = analyze_quality(original)
    print(f"Quality score: {quality.get('score', 'N/A')}")
    print(f"Needs enhancement: {quality.get('needs_enhancement', 'N/A')}")

    enhanced, applied = enhance_image(original, quality)
    print(f"Enhancements applied: {applied}")

    otsu = _otsu_binarize(original)

    variants = [
        ("original", original),
        ("enhanced", enhanced),
        ("otsu", otsu),
    ]

    all_candidates = []
    seen_keys = set()

    for name, img_arr in variants:
        print(f"Running Tamil OCR on: {name} ...")
        try:
            result = _tamil_ocr_on_array(img_arr)
            lines_data = []
            for text, box, conf in result.lines:
                pts = box.tolist()
                xs = [float(p[0]) for p in pts]
                ys = [float(p[1]) for p in pts]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                w_box = x_max - x_min
                h_box = y_max - y_min
                repaired = repair_mojibake(str(text))
                norm_key = "".join(ch for ch in repaired.lower() if ch.isalnum())
                if not norm_key or norm_key in seen_keys:
                    continue
                seen_keys.add(norm_key)
                lines_data.append({
                    "variant": name,
                    "text": repaired,
                    "original_text": str(text),
                    "confidence": float(conf),
                    "bbox": pts,
                    "x": round(x_min, 1),
                    "y": round(y_min, 1),
                    "width": round(w_box, 1),
                    "height": round(h_box, 1),
                    "has_tamil": has_tamil(repaired),
                    "has_latin": has_latin(repaired),
                    "tamil_char_count": tamil_char_count(repaired),
                })
            all_candidates.extend(lines_data)
            print(f"  -> {len(lines_data)} unique lines detected")
        except Exception as exc:
            print(f"  -> FAILED: {exc}")

    # Sort helpers
    by_y = sorted(all_candidates, key=lambda x: x["y"])
    by_height = sorted(all_candidates, key=lambda x: x["height"], reverse=True)
    by_conf = sorted(all_candidates, key=lambda x: x["confidence"], reverse=True)

    # Special debug candidates
    special_patterns = [r"\bA\.", r"\bS\.", r"\bM\.A\."]
    a_s_ma_candidates = [
        c for c in all_candidates
        if any(re.search(p, c["text"]) for p in special_patterns)
    ]

    # Middle/name region: y between 30% and 70% of image height
    img_h = original.shape[0]
    mid_region = [
        c for c in all_candidates
        if 0.30 * img_h <= c["y"] <= 0.70 * img_h
    ]

    # A. name and S. name detection
    a_name_detected = any(
        re.search(r"\bA\.[\s\u0B80-\u0BFF]", c["text"])
        for c in all_candidates
    )
    s_name_detected = any(
        re.search(r"\bS\.[\s\u0B80-\u0BFF]", c["text"])
        for c in all_candidates
    )

    output = {
        "image": str(IMAGE_PATH),
        "image_size": f"{original.shape[1]}x{original.shape[0]}",
        "total_unique_candidates": len(all_candidates),
        "a_name_detected": a_name_detected,
        "s_name_detected": s_name_detected,
        "candidates_by_y": by_y,
        "candidates_by_height": by_height,
        "candidates_by_confidence": by_conf,
        "a_s_ma_candidates": a_s_ma_candidates,
        "middle_region_candidates": mid_region,
    }

    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nJSON saved: {JSON_OUT}")

    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TAMIL OCR CANDIDATES REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Image: {IMAGE_PATH}\n")
        f.write(f"Size: {original.shape[1]}x{original.shape[0]}\n")
        f.write(f"Total unique candidates: {len(all_candidates)}\n\n")

        def write_section(title, candidates):
            f.write("-" * 80 + "\n")
            f.write(f"{title}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'#':<5} {'VAR':<10} {'Y':>8} {'H':>6} {'W':>6} {'CONF':>8} {'TAMIL':<6} {'LATIN':<6} {'TEXT'}\n")
            for i, c in enumerate(candidates, 1):
                text_short = c["text"][:60].replace("\n", "\\n")
                f.write(
                    f"{i:<5} {c['variant']:<10} {c['y']:>8.1f} {c['height']:>6.1f} "
                    f"{c['width']:>6.1f} {c['confidence']:>8.4f} "
                    f"{'Y' if c['has_tamil'] else 'N':<6} {'Y' if c['has_latin'] else 'N':<6} "
                    f"{text_short}\n"
                )
            f.write("\n")

        write_section("A. SORTED BY VERTICAL POSITION (top -> bottom)", by_y)
        write_section("B. SORTED BY BOUNDING-BOX HEIGHT (largest -> smallest)", by_height)
        write_section("C. SORTED BY CONFIDENCE (highest -> lowest)", by_conf)

        f.write("-" * 80 + "\n")
        f.write("SPECIAL DEBUG: A. / S. / M.A. CANDIDATES\n")
        f.write("-" * 80 + "\n")
        if a_s_ma_candidates:
            for c in a_s_ma_candidates:
                f.write(f"  y={c['y']:>8.1f} h={c['height']:>6.1f} conf={c['confidence']:.4f} text={c['text']}\n")
        else:
            f.write("  No A./S./M.A. candidates detected.\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("SPECIAL DEBUG: MIDDLE/NAME REGION CANDIDATES (30%-70% height)\n")
        f.write("-" * 80 + "\n")
        if mid_region:
            for c in mid_region:
                f.write(f"  y={c['y']:>8.1f} h={c['height']:>6.1f} conf={c['confidence']:.4f} text={c['text']}\n")
        else:
            f.write("  No candidates in middle/name region.\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("NAME REGION DETECTION\n")
        f.write("-" * 80 + "\n")
        f.write(f"A. name region detected: {a_name_detected}\n")
        f.write(f"S. name region detected: {s_name_detected}\n")
        if not a_name_detected:
            f.write("A. name region not detected by OCR.\n")
        if not s_name_detected:
            f.write("S. name region not detected by OCR.\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("MOJIBAKE REPAIR TEST\n")
        f.write("-" * 80 + "\n")
        moji_samples = [
            "à®...à¯...à®´...",
            "à®´à®±à¯à®¯à®à®®à¯à®¯à®à®®à®²à®®à¯à®²à®©à®à®©à®à®²à®¾à®²à®²à®¾à®®à¯à®²à®¾à®©à¯à®®à®°à¯à®¤à®®à®²à¯à®ªà®ªà®³à®®à¯à®±à®¿à®¯à®²à¯à®µà®¿à®°à®®à®¿à®¯à®²à¯",
        ]
        for sample in moji_samples:
            repaired = repair_mojibake(sample)
            f.write(f"BEFORE: {sample}\n")
            f.write(f"AFTER:  {repaired}\n\n")

    print(f"Text report saved: {TXT_OUT}")
    print(f"Mojibake test saved: {MOJIBAKE_TEST}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total unique candidates: {len(all_candidates)}")
    for name, _ in variants:
        count = sum(1 for c in all_candidates if c["variant"] == name)
        print(f"  {name}: {count} candidates")
    print(f"A. name detected: {a_name_detected}")
    print(f"S. name detected: {s_name_detected}")
    print(f"A./S./M.A. candidates: {len(a_s_ma_candidates)}")
    print(f"Middle region candidates: {len(mid_region)}")


if __name__ == "__main__":
    run_tamil_diagnostic()
