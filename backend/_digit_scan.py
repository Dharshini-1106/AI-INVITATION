import cv2
import numpy as np
import re
import pathlib
from rapidocr_onnxruntime import RapidOCR

ROOT = pathlib.Path(r"D:/MINI/backend")
img = cv2.imread(str(ROOT / "tamil_invitation.png"))
h, w = img.shape[:2]
ocr = RapidOCR()
found = []
n_strips = 40
for i in range(n_strips):
    y0 = int(h * i / n_strips)
    y1 = int(h * (i + 1) / n_strips) + 10
    band = img[y0:y1, :]
    im = cv2.resize(band, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    res, _ = ocr(im)
    if not res:
        continue
    for box, text, score in res:
        if re.search(r"\d", text):
            found.append("strip%d y~%d | %s | %s" % (i, (y0 + y1) // 2, score, text))
out = "\n".join(found) if found else "NO DIGITS FOUND"
pathlib.Path("_digit_scan.txt").write_text(out, encoding="utf-8")
print("found", len(found))
