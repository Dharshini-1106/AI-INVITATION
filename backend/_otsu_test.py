import cv2
import numpy as np
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR
from rapidocr_onnxruntime.ch_ppocr_v3_rec.text_recognize import TextRecognizer

ROOT = Path(r"D:/MINI/backend")
tamil_model = str((ROOT / "models/tamil/model.onnx").resolve())
tamil_dict = str((ROOT / "models/tamil/dict.txt").resolve())


def repair(s):
    # Fix double-encoded utf-8 (latin1) -> tamil
    try:
        s.encode("latin-1").decode("utf-8")
        return s.encode("latin-1").decode("utf-8")
    except Exception:
        return s


def build_ocr():
    ocr = RapidOCR()
    cfg = {"model_path": tamil_model, "keys_path": tamil_dict,
           "rec_img_shape": [3, 48, 320], "rec_batch_num": 6, "use_cuda": False}
    ocr.text_recognizer = TextRecognizer(cfg)
    return ocr


img = cv2.imread(str(ROOT / "tamil_invitation.png"))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ys, xs = np.where(gray > 20)
y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
crop = img[y0:y1 + 1, x0:x1 + 1]

# otsu binary
_, otsu = cv2.threshold(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# invert so text is dark on light? PP-OCR expects dark text on light bg.
# otsu gives dark text=0 on white=255 already (text darker). keep.
bgr = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
scale = 3.0
im = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
ocr = build_ocr()
res, _ = ocr(im)
lines = []
if res:
    for box, text, score in res:
        rb = repair(text)
        lines.append("%s | raw=%r repaired=%r" % (score, text, rb))
out = "\n".join(lines)
(ROOT / "_otsu_repair.txt").write_text(out, encoding="utf-8")
print("DONE", len(lines))
