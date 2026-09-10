import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "app"))
os.chdir(ROOT)
os.environ["OMP_NUM_THREADS"] = "4"
import cv2
import numpy as np

TAMIL_LO, TAMIL_HI = 0x0B80, 0x0BFF
def tcount(s): return sum(1 for ch in s if TAMIL_LO <= ord(ch) <= TAMIL_HI)
def latin_count(s): return sum(1 for ch in s if ch.isascii() and ch.isalpha())

from rapidocr_onnxruntime import RapidOCR
from rapidocr_onnxruntime.ch_ppocr_v3_rec.text_recognize import TextRecognizer

en_ocr = RapidOCR()
def make_tamil():
    model_path = os.path.join(ROOT, "models", "tamil", "model.onnx")
    dict_path = os.path.join(ROOT, "models", "tamil", "dict.txt")
    o = RapidOCR()
    cfg = {"model_path": model_path, "keys_path": dict_path,
           "rec_img_shape": [3, 48, 320], "rec_batch_num": 6, "use_cuda": False}
    o.text_recognizer = TextRecognizer(cfg)
    return o
ta_ocr = make_tamil()

for name in ["test_inv3.png", "tamil_invitation.png"]:
    IMG = os.path.join(ROOT, name)
    img = cv2.imread(IMG)
    h, w = img.shape[:2]
    print(f"\n##### {name} {w}x{h} #####")
    # English full pass
    t0=time.time()
    en_res, _ = en_ocr(IMG, use_det=True, use_rec=True)
    print(f"  EN det+rec: {len(en_res) if en_res else 0} lines in {time.time()-t0:.1f}s")
    if en_res:
        for item in en_res:
            box, text, conf = item[0], item[1], item[2]
            x0=min(p[0] for p in box)
            print(f"    EN x={x0:5.0f} conf={float(conf):.3f} tc={tcount(text)} lat={latin_count(text)} | {text!r}")
