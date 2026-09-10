import os, sys, time, traceback
sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "app"))
os.chdir(ROOT)
os.environ["OMP_NUM_THREADS"] = "4"

import cv2
import numpy as np

def load_rapid():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()

def load_tamil_rapid():
    model_path = os.path.join(ROOT, "models", "tamil", "model.onnx")
    dict_path = os.path.join(ROOT, "models", "tamil", "dict.txt")
    from rapidocr_onnxruntime import RapidOCR
    from rapidocr_onnxruntime.ch_ppocr_v3_rec.text_recognize import TextRecognizer
    ocr = RapidOCR()
    cfg = {
        "model_path": str(model_path),
        "keys_path": str(dict_path),
        "rec_img_shape": [3, 48, 320],
        "rec_batch_num": 6,
        "use_cuda": False,
    }
    ocr.text_recognizer = TextRecognizer(cfg)
    return ocr

IMG = os.path.join(ROOT, "tamil_invitation.png")
img = cv2.imread(IMG)
h, w = img.shape[:2]
print(f"Image: {w}x{h}")

en = load_rapid()
print("\n=== ENGLISH RapidOCR on full image ===")
t0 = time.time()
res_en, _ = en(IMG, use_det=True, use_rec=True)
print(f"  time={time.time()-t0:.2f}s, lines={len(res_en) if res_en else 0}")
print(f"  item0 type={type(res_en[0]) if res_en else None}, fields={len(res_en[0]) if res_en else 0}")
if res_en:
    for item in res_en[:8]:
        box, text, conf = item[0], item[1], item[2]
        print(f"  conf={float(conf):.3f} | {text!r}")

print("\n=== TAMIL RapidOCR on full image ===")
try:
    ta = load_tamil_rapid()
    t0 = time.time()
    res_ta, _ = ta(IMG, use_det=True, use_rec=True)
    print(f"  time={time.time()-t0:.2f}s, lines={len(res_ta) if res_ta else 0}")
    if res_ta:
        for item in res_ta[:8]:
            box, text, conf = item[0], item[1], item[2]
            tc = sum(1 for ch in text if 0x0B80 <= ord(ch) <= 0x0BFF)
            print(f"  tc={tc} conf={float(conf):.3f} | {text!r}")
except Exception as e:
    traceback.print_exc()
