"""Isolated PaddleOCR Phase 1 probe; never imported by the production API."""
import sys

from paddleocr import PaddleOCR

image_path = sys.argv[1] if len(sys.argv) > 1 else "_rapid_test.png"
language = sys.argv[2] if len(sys.argv) > 2 else "en"

print("before", flush=True)
ocr = PaddleOCR(use_angle_cls=True, lang=language, show_log=False)
print("MODEL_LOAD_OK", flush=True)
result = ocr.ocr(image_path, cls=True)
lines = [item[1][0] for page in result for item in (page or [])]
print(repr(lines), flush=True)
