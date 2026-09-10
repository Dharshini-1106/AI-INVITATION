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

IMG = os.path.join(ROOT, "test_inv6.png")
img = cv2.imread(IMG); print("img", img.shape[1], img.shape[0])

# English, angle cls OFF
e1 = RapidOCR(use_angle_cls=False)
t0=time.time(); r,_=e1(IMG, use_det=True, use_cls=False, use_rec=True); dt=time.time()-t0
print(f"\n[EN use_angle_cls=False] {len(r) if r else 0} lines in {dt:.1f}s")
for item in (r or [])[:6]:
    b,t,c=item[0],item[1],item[2]; print(f"  conf={float(c):.3f} tc={tcount(t)} | {t!r}")

# Now test Tamil model on an ENGLISH crop: crop "Sri Venkateshaperumal Thunai" region
ta = RapidOCR()
cfg={"model_path":os.path.join(ROOT,"models","tamil","model.onnx"),
     "keys_path":os.path.join(ROOT,"models","tamil","dict.txt"),
     "rec_img_shape":[3,48,320],"rec_batch_num":6,"use_cuda":False}
ta.text_recognizer=TextRecognizer(cfg)
# english box from earlier: x99-450 y91-114
crop = img[91:114, 99:450]
cv2.imwrite("_eng_crop.png", crop)
t0=time.time(); r2,_=ta(_eng_crop_path if False else "_eng_crop.png", use_det=False, use_cls=False, use_rec=True); dt=time.time()-t0
print(f"\n[TAMIL model on ENGLISH crop] {len(r2) if r2 else 0} lines in {dt:.2f}s")
for item in (r2 or []):
    b,t,c=item[0],item[1],item[2]; print(f"  conf={float(c):.3f} tc={tcount(t)} lat={latin_count(t)} | {t!r}")
