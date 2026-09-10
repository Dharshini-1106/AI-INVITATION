"""Test runner: run the OCR pipeline on all 6 invitation images and save results."""
import sys
import json
import os
import io
import logging
import time
import tempfile
import codecs

# Set stdout to UTF-8
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

os.chdir("D:/MINI/backend")
sys.path.insert(0, "app")

# Suppress noisy logs during testing
logging.basicConfig(level=logging.WARNING, format="%(message)s")

from app.core.pipeline import run_pipeline

def load_image_bytes(path):
    with open(path, "rb") as f:
        return f.read()

results = {}
for i in range(1, 7):
    img_path = f"D:/MINI/backend/test_inv{i}.png"
    if not os.path.exists(img_path):
        print(f"MISSING: {img_path}")
        continue
    
    print(f"\n{'='*60}")
    print(f"Testing Invitation {i}: {img_path}")
    print(f"{'='*60}")
    
    img_bytes = load_image_bytes(img_path)
    start = time.time()
    result = run_pipeline(img_bytes, f"test_inv{i}.png", source="gallery")
    elapsed = time.time() - start
    
    # Save raw OCR text
    raw_output = result.get("raw_text", "")
    with open(f"D:/MINI/backend/_test_inv{i}_raw.txt", "w", encoding="utf-8") as f:
        f.write(raw_output)
    
    # Save full OCR layout (per-line boxes)
    ocr_layout = result.get("ocr_layout", [])
    with open(f"D:/MINI/backend/_test_inv{i}_layout.json", "w", encoding="utf-8") as f:
        json.dump(ocr_layout, f, ensure_ascii=False, indent=2)
    
    # Save final structured output
    final_output = {
        "event_name": result.get("event_name", ""),
        "event_type": result.get("event_type", ""),
        "bride_name": result.get("bride_name", ""),
        "groom_name": result.get("groom_name", ""),
        "date": result.get("date", ""),
        "time": result.get("time", ""),
        "venue": result.get("venue", ""),
        "address": result.get("address", ""),
        "contact_number": result.get("contact_number", ""),
        "language": result.get("language", ""),
        "confidence_score": result.get("confidence_score", 0),
        "tamil_character_count": result.get("tamil_character_count", 0),
        "english_character_count": result.get("english_character_count", 0),
        "ocr_engine": result.get("ocr_engine", ""),
        "ocr_confidence": result.get("ocr_confidence", 0),
        "processing_notes": result.get("processing_notes", []),
        "events": result.get("events", []),
        "number_of_events": result.get("number_of_events", 1),
    }
    
    with open(f"D:/MINI/backend/_test_inv{i}_final.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    # Write summary to file
    with open(f"D:/MINI/backend/_test_inv{i}_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Time: {elapsed:.1f}s\n")
        f.write(f"Tamil chars: {result.get('tamil_character_count', 0)}\n")
        f.write(f"English chars: {result.get('english_character_count', 0)}\n")
        f.write(f"OCR engine: {result.get('ocr_engine', '')}\n")
        f.write(f"Event type: {result.get('event_type', '')}\n")
        f.write(f"Bride: {result.get('bride_name', '')}\n")
        f.write(f"Groom: {result.get('groom_name', '')}\n")
        f.write(f"Date: {result.get('date', '')}\n")
        f.write(f"Time: {result.get('time', '')}\n")
        f.write(f"Venue: {result.get('venue', '')}\n")
        f.write(f"Address: {result.get('address', '')}\n")
        f.write(f"Phone: {result.get('contact_number', '')}\n")
        f.write(f"Confidence: {result.get('confidence_score', 0)}\n")
        f.write(f"Notes: {result.get('processing_notes', [])}\n")
        f.write(f"\nRAW TEXT:\n{raw_output}\n")
        f.write(f"\nOCR LAYOUT:\n{json.dumps(ocr_layout, ensure_ascii=False, indent=2)}\n")
    
    print(f"Time: {elapsed:.1f}s")
    print(f"Tamil chars: {result.get('tamil_character_count', 0)}")
    print(f"English chars: {result.get('english_character_count', 0)}")
    print(f"OCR engine: {result.get('ocr_engine', '')}")
    print(f"Event type: {result.get('event_type', '')}")
    print(f"Results saved to _test_inv{i}_summary.txt")
    
    results[i] = {
        "elapsed": elapsed,
        "raw_text": raw_output,
        "final": final_output,
        "ocr_layout": ocr_layout,
    }

print("\n" + "="*60)
print("ALL TESTS COMPLETE")
print("="*60)
