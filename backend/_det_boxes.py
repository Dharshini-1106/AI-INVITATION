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
h, w = img.shape[:2]
# run detection only
ocr = build_ocr()
det_boxes, _ = ocr.text_detector(img)
det_boxes = ocr.sorted_boxes(det_boxes)
rows = []
for i, box in enumerate(det_boxes):
    ys = box[:, 1]
    xs = box[:, 0]
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    bh = ymax - ymin
    bw = xmax - xmin
    rows.append((ymin, ymax, xmin, xmax, bh, bw))
# sort by height desc
rows.sort(key=lambda r: -r[4])
top = rows[:25]
top.sort(key=lambda r: r[0])
out = []
for r in top:
    out.append("y[%d-%d] x[%d-%d] h=%d w=%d" % (r[0], r[1], r[2], r[3], r[4], r[5]))
(ROOT / "_det_boxes.txt").write_text("\n".join(out), encoding="utf-8")
print("total boxes", len(det_boxes), "written", len(out))
