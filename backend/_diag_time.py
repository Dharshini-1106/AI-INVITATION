import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np, cv2
TAMIL_LO, TAMIL_HI = 0x0B80, 0x0BFF
def tcount(s): return sum(1 for ch in s if TAMIL_LO <= ord(ch) <= TAMIL_HI)

from rapidocr_onnxruntime import RapidOCR
from rapidocr_onnxruntime.ch_ppocr_v3_rec.text_recognize import TextRecognizer

en = RapidOCR(use_angle_cls=False)
def tamil():
    ta = RapidOCR(use_angle_cls=False)
    cfg={"model_path":os.path.join(ROOT,"models","tamil","model.onnx"),
         "keys_path":os.path.join(ROOT,"models","tamil","dict.txt"),
         "rec_img_shape":[3,48,320],"rec_batch_num":6,"use_cuda":False}
    ta.text_recognizer=TextRecognizer(cfg); return ta
ta = tamil()

IMG=os.path.join(ROOT,"test_inv6_small.png") if os.path.exists(os.path.join(ROOT,"test_inv6_small.png")) else None
# make a 1200px-wide version for speed tests
big = cv2.imread(os.path.join(ROOT,"test_inv6.png"))
h,w = big.shape[:2]
small = cv2.resize(big, (1200, int(h*1200/w)), interpolation=cv2.INTER_AREA)
cv2.imwrite(os.path.join(ROOT,"_inv6_1200.png"), small)
IMG=os.path.join(ROOT,"_inv6_1200.png")

# warmup
_ = en(IMG, use_det=True, use_cls=False, use_rec=True)
print("warm")

import time
arr = cv2.imread(IMG)
t0=time.time(); r,_=en(arr, use_det=True, use_cls=False, use_rec=True); print(f"EN full call (warm) on 1200px: {time.time()-t0:.1f}s, {len(r)} lines")
t0=time.time(); r2,_=ta(IMG, use_det=True, use_cls=False, use_rec=True); print(f"TA full call (warm) on 1200px: {time.time()-t0:.1f}s, {len(r2)} lines")

# Test Tamil model on English crops (digits, date, phone) to see if digits read right
print("\n--- TA on specific English crops ---")
# "Sunday, 14th February 2016" region approx bbox from earlier: x99-422 y556-585 (but at 3000px scale)
# Use small image scale: 1200/3000=0.4
for cropname,(x,y,xs,ys) in {"date":(100,556,430,590), "phone_a":(1600,500,1980,530), "addr":(100,970,320,1000)}.items():
    xs2=int(xs*0.4); ys2=int(ys*0.4); x2=int(x*0.4); y2=int(y*0.4)
    c = small[y2:ys2, x2:xs2]
    if c.size==0: 
        print(cropname,"empty"); continue
    cv2.imwrite("_crop_"+cropname+".png", c)
    rr,_=ta("_crop_"+cropname+".png", use_det=False, use_cls=False, use_rec=True)
    txt = rr[0][1] if rr else ""
    print(f"  TA crop {cropname}: {txt!r} tc={tcount(txt)}")
