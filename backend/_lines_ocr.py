import cv2
import numpy as np
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR
from rapidocr_onnxruntime.ch_ppocr_v3_rec.text_recognize import TextRecognizer

ROOT = Path(r"D:/MINI/backend")
tamil_model = str((ROOT / "models/tamil/model.onnx").resolve())
tamil_dict = str((ROOT / "models/tamil/dict.txt").resolve())


def build_ocr():
    ocr = RapidOCR()
    cfg = {"model_path": tamil_model, "keys_path": tamil_dict,
           "rec_img_shape": [3, 48, 320], "rec_batch_num": 6, "use_cuda": False}
    ocr.text_recognizer = TextRecognizer(cfg)
    return ocr


img = cv2.imread(str(ROOT / "tamil_invitation.png"))
ocr = build_ocr()
det_boxes, _ = ocr.text_detector(img)
det_boxes = ocr.sorted_boxes(det_boxes)

crops = []
meta = []
for box in det_boxes:
    ys = box[:, 1]; xs = box[:, 0]
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    pad = 6
    y0 = max(0, ymin - pad); y1 = min(img.shape[0], ymax + pad)
    x0 = max(0, xmin - pad); x1 = min(img.shape[1], xmax + pad)
    crop = img[y0:y1, x0:x1]
    # binarize for better rec
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # contrast stretch
    g = cv2.equalizeHist(g)
    crop = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    # scale so height ~ 64
    scale = 64.0 / max(1, crop.shape[0])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    crops.append(crop)
    meta.append((ymin, ymax, xmin, xmax))

rec_res, _ = ocr.text_recognizer(crops)
out = []
for (ymin, ymax, xmin, xmax), (text, score) in zip(meta, rec_res):
    out.append("y[%d-%d] x[%d-%d] h=%d | %.3f | %s" % (
        ymin, ymax, xmin, xmax, ymax - ymin, score, text))
(ROOT / "_lines_ocr.txt").write_text("\n".join(out), encoding="utf-8")
print("DONE", len(out))
