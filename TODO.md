# Invitation Understanding — Backend Fix Plan

## Root Cause
- Backend Python dependencies were NOT installed (`fastapi` missing) → server wouldn't start.
- `requirements.txt` pins `torch==2.5.1` / `paddlepaddle==2.6.2` which are incompatible with Python 3.13.5.
- After fixing deps, the real OCR engine (`rapidocr-onnxruntime`) was missing → OCR fell back to rule-based
  `[text-block]` placeholders → all extracted fields were empty ("nothing is extracted").

## Steps
- [x] 1. Create Python virtual environment in `backend/`
- [x] 2. Install lightweight core stack (`requirements-core.txt`) + enable RapidOCR
- [x] 3. Make OCR honor config toggles and use RapidOCR when PaddleOCR not installed
- [x] 4. Make pipeline honor config toggles (skip heavy models when disabled)
- [x] 5. Improve ProcessingScreen error message (distinguish disconnected vs. slow)
- [x] 6. Verify backend starts and /health responds
- [x] 7. Enable & install real OCR: uncomment `rapidocr-onnxruntime`, install
      `rapidocr-onnxruntime==1.2.3` (Python 3.13-compatible), `pyclipper`, `shapely`
- [x] 8. Restart backend; startup log now shows `RapidOCR initialized (multilingual ONNX)`
- [x] 9. Verify real text extraction from `_rapid_test.png` → reads "WEDDING INVITATION / Mr Arjun & Ms Priya"
- [x] 10. Fix false PERSON entity extraction (event words like Figma, Learn, Gain, ORGANIZES tagged as names)
- [x] 11. Fix event name extraction for non-traditional event titles (workshop/training/design)
- [x] 12. Fix venue extraction for college/university/institute venues
- [x] 13. Fix day names, duration words, and city names being tagged as PERSON entities
