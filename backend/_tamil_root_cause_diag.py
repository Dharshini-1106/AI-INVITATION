"""Tamil OCR Root Cause Diagnostic.

Tasks:
1. Save existing raw OCR result as UTF-8 text
2. Analyze character counts
3. Test PaddleOCR Tamil on small synthetic image
4. Test PaddleOCR Tamil on cropped region from invitation
5. Measure timing for init vs inference
"""
import sys
import os
import json
import time
import tempfile
import re
from pathlib import Path

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

BACKEND = Path(r"D:/MINI/backend")
IMAGE_PATH = BACKEND / "tamil_invitation.png"
TAMIL_LO = 0x0B80
TAMIL_HI = 0x0BFF


def has_tamil(text: str) -> bool:
    return any(TAMIL_LO <= ord(ch) <= TAMIL_HI for ch in (text or ""))


def tamil_char_count(text: str) -> int:
    return sum(1 for ch in (text or "") if TAMIL_LO <= ord(ch) <= TAMIL_HI)


def latin_char_count(text: str) -> int:
    return sum(1 for ch in (text or "") if ch.isalpha() and ch.isascii())


def digit_count(text: str) -> int:
    return sum(1 for ch in (text or "") if ch.isdigit())


def analyze_text(text: str) -> dict:
    tc = tamil_char_count(text)
    lc = latin_char_count(text)
    dc = digit_count(text)
    total = len(text)
    return {
        "total_chars": total,
        "tamil_chars": tc,
        "latin_chars": lc,
        "digits": dc,
        "tamil_pct": round(tc / total * 100, 2) if total else 0.0,
        "latin_pct": round(lc / total * 100, 2) if total else 0.0,
    }


# ===========================================================================
# TASK 2: Save raw OCR result as UTF-8 and analyze
# ===========================================================================
def save_raw_result():
    src = BACKEND / "_tamil_ocr_result.json"
    dst_txt = BACKEND / "_paddleocr_ta_raw.txt"
    dst_json = BACKEND / "_paddleocr_ta_raw.json"

    if not src.exists():
        print("SKIP: _tamil_ocr_result.json not found")
        return

    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    lines = []
    for i, item in enumerate(data, 1):
        text = item.get("text", "")
        score = item.get("score", 0.0)
        lines.append(f"{i:2d}. [{score:.4f}] {text}")

    with open(dst_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    all_text = " ".join(item.get("text", "") for item in data)
    stats = analyze_text(all_text)
    stats["total_lines"] = len(data)
    stats["avg_confidence"] = round(
        sum(item.get("score", 0.0) for item in data) / len(data), 4
    ) if data else 0.0

    print("TASK 2 RESULTS:")
    print(f"  Total lines: {stats['total_lines']}")
    print(f"  Tamil chars: {stats['tamil_chars']}")
    print(f"  Latin chars: {stats['latin_chars']}")
    print(f"  Digits: {stats['digits']}")
    print(f"  Avg confidence: {stats['avg_confidence']}")
    print(f"  Tamil %: {stats['tamil_pct']}%")
    print(f"  Latin %: {stats['latin_pct']}%")
    print(f"  Saved: {dst_txt}")
    print(f"  Saved: {dst_json}")
    return stats


# ===========================================================================
# TASK 5: Test on small cropped region (recognition-only if possible)
# ===========================================================================
def test_crop_recognition():
    import cv2
    import numpy as np

    print("\nTASK 5: CROP REGION RECOGNITION TEST")

    img = cv2.imread(str(IMAGE_PATH))
    if img is None:
        print("  SKIP: Cannot read image")
        return None

    h, w = img.shape[:2]
    # Crop a middle region that likely contains Tamil text
    y0, y1 = int(h * 0.35), int(h * 0.65)
    x0, x1 = int(w * 0.1), int(w * 0.9)
    crop = img[y0:y1, x0:x1]

    crop_path = BACKEND / "_tamil_crop.png"
    cv2.imwrite(str(crop_path), crop)
    print(f"  Crop saved: {crop_path} ({crop.shape[1]}x{crop.shape[0]})")

    # Try PaddleOCR TextRecognition model directly
    try:
        from paddleocr import TextRecognition

        t0 = time.time()
        rec = TextRecognition(model_name="ta_PP-OCRv5_mobile_rec")
        init_time = time.time() - t0
        print(f"  TextRecognition init: {init_time:.2f}s")

        t0 = time.time()
        result = rec.predict(str(crop_path))
        infer_time = time.time() - t0
        print(f"  Recognition inference: {infer_time:.2f}s")

        texts = []
        for res in result:
            if hasattr(res, "rec_text"):
                texts.append(res.rec_text)
            elif hasattr(res, "json"):
                try:
                    payload = json.loads(res.json) if isinstance(res.json, str) else res.json
                    texts.extend(payload.get("rec_texts", []))
                except Exception:
                    pass

        full_text = " ".join(str(t) for t in texts)
        print(f"  Output: {full_text[:200]}")
        stats = analyze_text(full_text)
        stats["init_time_s"] = round(init_time, 2)
        stats["infer_time_s"] = round(infer_time, 2)
        stats["crop_size"] = f"{crop.shape[1]}x{crop.shape[0]}"
        return stats

    except Exception as e:
        print(f"  TextRecognition API failed: {e}")
        # Fallback: try using PaddleOCR pipeline on crop
        try:
            from paddleocr import PaddleOCR

            t0 = time.time()
            ocr = PaddleOCR(
                lang="ta",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
            init_time = time.time() - t0
            print(f"  PaddleOCR init: {init_time:.2f}s")

            t0 = time.time()
            result = ocr.predict(str(crop_path))
            infer_time = time.time() - t0
            print(f"  Pipeline inference: {infer_time:.2f}s")

            texts = []
            for page in result:
                payload = page.json if hasattr(page, "json") else page
                if isinstance(payload, str):
                    payload = json.loads(payload)
                payload = payload.get("res", payload)
                texts.extend(payload.get("rec_texts", []))

            full_text = " ".join(str(t) for t in texts)
            print(f"  Output: {full_text[:200]}")
            stats = analyze_text(full_text)
            stats["init_time_s"] = round(init_time, 2)
            stats["infer_time_s"] = round(infer_time, 2)
            stats["crop_size"] = f"{crop.shape[1]}x{crop.shape[0]}"
            return stats

        except Exception as e2:
            print(f"  PaddleOCR pipeline also failed: {e2}")
            return None


# ===========================================================================
# TASK 6: Synthetic Tamil sample test
# ===========================================================================
def test_synthetic_tamil():
    print("\nTASK 6: SYNTHETIC TAMIL SAMPLE TEST")

    try:
        from PIL import Image, ImageDraw, ImageFont

        # Try to find a Tamil font
        font_paths = [
            "C:/Windows/Fonts/NotoSansTamil-Regular.ttf",
            "C:/Windows/Fonts/NirmalaUI.ttf",
            "C:/Windows/Fonts/seguiemj.ttf",
        ]
        font_path = None
        for fp in font_paths:
            if Path(fp).exists():
                font_path = fp
                break

        if font_path is None:
            print("  WARNING: No Tamil font found, using default")
            font = ImageFont.load_default()
            text = "தமிழ்"
        else:
            font = ImageFont.truetype(font_path, 48)
            text = "திருமணம்\nஅழைப்பிதழ்"

        img = Image.new("RGB", (400, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), text, fill=(0, 0, 0), font=font)

        synth_path = BACKEND / "_tamil_synthetic.png"
        img.save(str(synth_path))
        print(f"  Synthetic image saved: {synth_path}")

        # Test with TextRecognition
        from paddleocr import TextRecognition

        t0 = time.time()
        rec = TextRecognition(model_name="ta_PP-OCRv5_mobile_rec")
        init_time = time.time() - t0

        t0 = time.time()
        result = rec.predict(str(synth_path))
        infer_time = time.time() - t0

        texts = []
        for res in result:
            if hasattr(res, "rec_text"):
                texts.append(res.rec_text)
            elif hasattr(res, "json"):
                try:
                    payload = json.loads(res.json) if isinstance(res.json, str) else res.json
                    texts.extend(payload.get("rec_texts", []))
                except Exception:
                    pass

        full_text = " ".join(str(t) for t in texts)
        print(f"  Init: {init_time:.2f}s, Infer: {infer_time:.2f}s")
        print(f"  Output: {full_text}")
        stats = analyze_text(full_text)
        stats["init_time_s"] = round(init_time, 2)
        stats["infer_time_s"] = round(infer_time, 2)
        stats["synthetic"] = True
        return stats

    except Exception as e:
        print(f"  Synthetic test failed: {e}")
        return None


# ===========================================================================
# TASK 1: Inspect model dictionary (print summary)
# ===========================================================================
def inspect_model_dict():
    print("\nTASK 1: TAMIL MODEL DICTIONARY INSPECTION")

    model_dir = Path(r"D:/MINI/.ml-cache/paddle/official_models/ta_PP-OCRv5_mobile_rec")
    dict_path = model_dir / "config.json"

    if not dict_path.exists():
        print(f"  Model config not found at {dict_path}")
        return

    with open(dict_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    chars = cfg.get("PostProcess", {}).get("character_dict", [])
    tamil_chars = [c for c in chars if TAMIL_LO <= ord(c) <= TAMIL_HI]
    latin_chars = [c for c in chars if c.isalpha() and c.isascii()]

    print(f"  Model: ta_PP-OCRv5_mobile_rec")
    print(f"  Config path: {dict_path}")
    print(f"  Total chars in dict: {len(chars)}")
    print(f"  Tamil chars in dict: {len(tamil_chars)}")
    print(f"  Latin chars in dict: {len(latin_chars)}")
    print(f"  Tamil Unicode present: {len(tamil_chars) > 0}")
    print(f"  Sample Tamil: {''.join(tamil_chars[:10])}")

    # Also check v3
    v3_dir = Path(r"D:/MINI/.ml-cache/paddle/official_models/ta_PP-OCRv3_mobile_rec")
    v3_path = v3_dir / "config.json"
    if v3_path.exists():
        with open(v3_path, "r", encoding="utf-8") as f:
            v3_cfg = json.load(f)
        v3_chars = v3_cfg.get("PostProcess", {}).get("character_dict", [])
        v3_tamil = [c for c in v3_chars if TAMIL_LO <= ord(c) <= TAMIL_HI]
        print(f"\n  v3 model dict total: {len(v3_chars)}")
        print(f"  v3 model Tamil chars: {len(v3_tamil)}")


if __name__ == "__main__":
    print("=" * 70)
    print("TAMIL OCR ROOT CAUSE DIAGNOSTIC")
    print("=" * 70)

    inspect_model_dict()
    task2_stats = save_raw_result()
    task5_stats = test_crop_recognition()
    task6_stats = test_synthetic_tamil()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if task5_stats:
        print(f"Crop recognition: {task5_stats}")
    if task6_stats:
        print(f"Synthetic recognition: {task6_stats}")
