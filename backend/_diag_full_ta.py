import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
os.environ["OMP_NUM_THREADS"] = "4"
import cv2
TAMIL_LO, TAMIL_HI = 0x0B80, 0x0BFF
def tcount(s): return sum(1 for ch in s if TAMIL_LO <= ord(ch) <= TAMIL_HI)
def latin_count(s): return sum(1 for ch in s if ch.isascii() and ch.isalpha())

from rapidocr_onnxruntime import RapidOCR
from rapidocr_onnxruntime.ch_ppocr_v3_rec.text_recognize import TextRecognizer

# Inspect dictionary
dict_path = os.path.join(ROOT, "models", "tamil", "dict.txt")
with open(dict_path, encoding="utf-8") as f:
    d = f.read()
print("dict chars:", len(d), repr(d[:120]))
print("has Tamil:", any(TAMIL_LO <= ord(c) <= TAMIL_HI for c in d))
print("has ascii letters:", any(65 <= ord(c) <= 122 for c in d))

def make_tamil():
    ta = RapidOCR()
    cfg={"model_path":os.path.join(ROOT,"models","tamil","model.onnx"),
         "keys_path":dict_path, "rec_img_shape":[3,48,320],"rec_batch_num":6,"use_cuda":False}
    ta.text_recognizer=TextRecognizer(cfg)
    return ta
ta = make_tamil()

IMG = os.path.join(ROOT, "test_inv6.png")
print("\n=== TAMIL model on FULL test_inv6 (both columns) ===")
t0=time.time(); r,_=ta(IMG, use_det=True, use_cls=False, use_rec=True); print(f"time={time.time()-t0:.1f}s lines={len(r) if r else 0}")
for item in (r or []):
    b,t,c=item[0],item[1],item[2]; print(f"  x={min(p[0] for p in b):6.0f} conf={float(c):.3f} tc={tcount(t)} | {t!r}")
