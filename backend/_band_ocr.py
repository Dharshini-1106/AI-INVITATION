import cv2
import numpy as np
import pathlib
from rapidocr_onnxruntime import RapidOCR
from rapidocr_onnxruntime.ch_ppocr_v3_rec.text_recognize import TextRecognizer

ROOT = pathlib.Path(r"D:/MINI/backend")
tamil_model = str((ROOT / "models/tamil/model.onnx").resolve())
tamil_dict = str((ROOT / "models/tamil/dict.txt").resolve())


def build_tamil():
    ocr = RapidOCR()
    cfg = {"model_path": tamil_model, "keys_path": tamil_dict,
           "rec_img_shape": [3, 48, 320], "rec_batch_num": 6, "use_cuda": False}
    ocr.text_recognizer = TextRecognizer(cfg)
    return ocr


img = cv2.imread(str(ROOT / "tamil_invitation.png"))
h, w = img.shape[:2]
# bottom band where date/time likely are
for (y0f, y1f) in [(0.6, 1.0), (0.72, 1.0), (0.85, 1.0)]:
    y0, y1 = int(h * y0f), int(h * y1f)
    band = img[y0:y1, :]
    out = [f"=== band y[{y0}-{y1}] scale 4 ==="]
    for eng in (True, False):
        ocr = RapidOCR() if eng else build_tamil()
        im = cv2.resize(band, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        res, _ = ocr(im)
        if res:
            for box, text, score in res:
                out.append(("ENG" if eng else "TAM") + " %s | %s" % (score, text))
        else:
            out.append(("ENG" if eng else "TAM") + " NO RESULT")
    pathlib.Path("_band_ocr.txt").open("a", encoding="utf-8").write("\n".join(out) + "\n")
print("done")
