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


def preprocess_variants(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = {}
    out["gray"] = gray
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out["clahe"] = clahe.apply(gray)
    # sharpen
    k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    out["sharpen"] = cv2.filter2D(gray, -1, k)
    # binary (otsu) inverted-friendly
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out["otsu"] = b
    # adaptive
    out["adaptive"] = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 21, 10)
    # denoise + upscale
    den = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    out["denoise"] = den
    return out


def to_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def run_all():
    img = cv2.imread(str(ROOT / "tamil_invitation.png"))
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray_full > 20)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = img[y0:y1 + 1, x0:x1 + 1]
    ocr = build_ocr()
    f = (ROOT / "_tamil_ocr_variants.txt").open("w", encoding="utf-8")
    for pre in ["raw", "clahe", "sharpen", "otsu", "adaptive", "denoise"]:
        for scale in [2.0, 3.0]:
            if pre == "raw":
                base = crop
            else:
                base = to_bgr(preprocess_variants(crop)[pre])
            im = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            res, _ = ocr(im)
            f.write("==== %s scale %.1f ====\n" % (pre, scale))
            if res:
                for box, text, score in res:
                    f.write("%s | %s\n" % (score, text))
            else:
                f.write("NO RESULT\n")
            f.flush()
    f.close()
    print("DONE")


if __name__ == "__main__":
    run_all()
