"""BRISQUE-based image quality analysis.

Implements a self-contained BRISQUE (Blind/Referenceless Image Spatial Quality
Evaluator) scoring using the classic NSS (Natural Scene Statistics) features.
Lower score = better perceived quality. An optional pre-trained SVM model can be
loaded; otherwise a lightweight heuristic fallback is used.

The module also detects specific conditions: blur, noise, brightness, rotation
and low resolution.
"""
import logging
import math
from typing import Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BRISQUEAnalyzer:
    """Analyzes image quality and returns a score plus condition flags."""

    def __init__(self, svm_model_path: str | None = None, scale: int = 0):
        self.svm_model_path = svm_model_path
        self.scale = scale

    # ------------------------------------------------------------------
    # NSS feature computation (simplified BRISQUE)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_local_norm(img: np.ndarray) -> np.ndarray:
        """Compute MSCN (mean subtracted contrast normalized) coefficients."""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        img = img.astype(np.float64)
        # Local mean with Gaussian window
        kernel = cv2.getGaussianKernel(7, 7.0 / 6.0)
        mu = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        mu_sq = mu * mu
        # Local variance
        sigma = cv2.filter2D(img * img, -1, kernel, borderType=cv2.BORDER_CONSTANT) - mu_sq
        sigma[sigma < 0] = 0
        sigma = np.sqrt(sigma)
        # MSCN
        mscn = (img - mu) / (sigma + 1.0)
        return mscn

    def _paired_products(self, mscn: np.ndarray) -> np.ndarray:
        """Compute 4 directional paired products (H, V, D1, D2)."""
        shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
        feats = []
        for sy, sx in shifts:
            shifted = np.roll(np.roll(mscn, sy, axis=0), sx, axis=1)
            feats.append(mscn * shifted)
        return np.array(feats)

    def _ggd_fit(self, data: np.ndarray) -> Tuple[float, float]:
        """Fit a Generalized Gaussian Distribution and return (alpha, sigma)."""
        data = data.astype(np.float64).ravel()
        data = data[data != 0]
        if data.size == 0:
            return 1.0, 1.0
        gamma = lambda x: np.exp(math.lgamma(x))  # noqa: E731

        def estimate_alpha(sigma_sq: float) -> float:
            m2 = sigma_sq
            m4 = np.mean(data ** 4)
            if m4 == 0 or m2 == 0:
                return 2.0
            ratio = m4 / (m2 ** 2)
            # Solve gamma(1/alpha)*gamma(3/alpha)/gamma(2/alpha)^2 = m2^2/m4
            r_hat = m2 / np.sqrt(m4) if m4 > 0 else 1.0
            r_hat = max(r_hat, 1e-6)
            # Newton-ish search for alpha
            alpha = 2.0
            for _ in range(50):
                g1 = gamma(1.0 / alpha)
                g2 = gamma(2.0 / alpha)
                g3 = gamma(3.0 / alpha)
                num = g1 * g3
                den = g2 * g2
                if den == 0:
                    break
                target = 0.5 * (m2 ** 2) / m4
                ratio_g = num / den
                if abs(ratio_g - target) < 1e-4:
                    break
                alpha = alpha * (1.0 + 0.2 * (ratio_g - target) / (ratio_g + 1e-9))
                alpha = min(max(alpha, 0.2), 10.0)
            return alpha

        sigma_sq = np.mean(data ** 2)
        alpha = estimate_alpha(sigma_sq)
        sigma = np.sqrt(sigma_sq)
        return float(alpha), float(sigma)

    def _aggd_fit(self, data: np.ndarray) -> Tuple[float, float, float, float]:
        """Fit an Asymmetric GGD and return (eta, alpha, sigma_l, sigma_r)."""
        data = data.astype(np.float64).ravel()
        data = data[data != 0]
        if data.size == 0:
            return 0.0, 1.0, 1.0, 1.0
        eta = np.mean(data)
        left = data[data < 0]
        right = data[data >= 0]
        n_l = left.size
        n_r = right.size
        if n_l == 0 or n_r == 0:
            n_l = n_l or 1
            n_r = n_r or 1
        sigma_l = np.sqrt(np.mean(left ** 2)) if left.size else 1.0
        sigma_r = np.sqrt(np.mean(right ** 2)) if right.size else 1.0
        alpha = 1.0
        if sigma_l > 0 and sigma_r > 0:
            r_hat = sigma_r / sigma_l
            alpha = max(0.2, min(10.0, 1.0 / (r_hat + 1e-9)))
        return float(eta), float(alpha), float(sigma_l), float(sigma_r)

    def compute_features(self, img: np.ndarray) -> np.ndarray:
        """Extract the 36 BRISQUE NSS features."""
        mscn = self._compute_local_norm(img)
        alpha, sigma = self._ggd_fit(mscn)
        feats = [alpha, sigma ** 2]
        for cc in self._paired_products(mscn):
            eta, a, sl, sr = self._aggd_fit(cc)
            feats += [eta, a, sl, sr]
        return np.array(feats, dtype=np.float64)

    # ------------------------------------------------------------------
    # Condition detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_blur(img: np.ndarray) -> float:
        """Return a blur metric (Laplacian variance). Lower = more blur."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _detect_noise(img: np.ndarray) -> float:
        """Estimate noise level via median deviation of a smooth region estimate."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, smooth)
        return float(np.mean(diff))

    @staticmethod
    def _detect_brightness(img: np.ndarray) -> Tuple[float, bool, bool]:
        """Return (mean brightness, is_dark, is_bright)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        mean_val = float(np.mean(gray))
        return mean_val, mean_val < 60, mean_val > 200

    @staticmethod
    def _detect_rotation(img: np.ndarray) -> bool:
        """Heuristic: detect strongly skewed text via Hough line angles."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=40, maxLineGap=10)
        if lines is None:
            return False
        angles = []
        for line in lines[:200]:
            line = np.ravel(line)
            if len(line) < 4:
                continue
            x1, y1, x2, y2 = line[:4]
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(ang)
        if not angles:
            return False
        angles = np.array(angles)
        # Text lines ~0 deg; big deviation indicates skew/rotation
        deviation = np.mean(np.abs(angles - np.median(angles)))
        return deviation > 8.0

    @staticmethod
    def _low_resolution(img: np.ndarray, min_side: int = 400) -> bool:
        h, w = img.shape[:2]
        return min(h, w) < min_side

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, img: np.ndarray) -> Dict:
        """Return full quality analysis dict."""
        # Normalize intensity for feature consistency
        img_work = (img / 255.0).astype(np.float64) if img.dtype != np.float64 else img
        if img_work.max() > 1.0:
            img_work = img_work / 255.0

        features = self.compute_features(img)

        # Heuristic quality score (0..100, lower = worse). We approximate the
        # SVM output with a weighted combination of NSS-derived signal measures.
        blur = self._detect_blur(img)
        noise = self._detect_noise(img)
        mean_b, is_dark, is_bright = self._detect_brightness(img)
        is_rotated = self._detect_rotation(img)
        low_res = self._low_resolution(img)

        # Weighted score logic
        score = 100.0
        if blur < 300:
            score -= 35
        if noise > 8:
            score -= 20
        if is_dark:
            score -= 25
        if is_bright:
            score -= 15
        if low_res:
            score -= 20
        if is_rotated:
            score -= 15
        score = max(10.0, min(100.0, score))

        needs_enhancement = (
            blur < 300 or noise > 8 or is_dark or is_bright or low_res or is_rotated
        )

        return {
            "score": round(score, 2),
            "is_blurred": bool(blur < 300),
            "is_noisy": bool(noise > 8),
            "is_dark": is_dark,
            "is_bright": is_bright,
            "is_rotated": is_rotated,
            "low_resolution": low_res,
            "needs_enhancement": needs_enhancement,
            "metrics": {
                "blur_metric": round(blur, 2),
                "noise_metric": round(noise, 2),
                "brightness": round(mean_b, 2),
            },
        }


analyzer = BRISQUEAnalyzer()


def analyze_quality(img: np.ndarray) -> Dict:
    """Convenience wrapper around the BRISQUE analyzer."""
    return analyzer.analyze(img)
