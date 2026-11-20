"""Step 1a — Otsu (global) binarization, written from scratch.

Pipeline entry point: `run_otsu(img_or_path)` -> uint8 binary, ink = 255.
"""

import cv2
import numpy as np

MAX_SIDE = 1200     # longer image side after resizing (px)
BLUR_KSIZE = 5      # Gaussian pre-blur kernel (0 = off)


def load_gray(img_or_path, max_side=MAX_SIDE):
    """Path or BGR/gray array -> grayscale uint8 with max(h, w) <= max_side."""
    if isinstance(img_or_path, np.ndarray):
        gray = img_or_path
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.imread(str(img_or_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read image: {img_or_path}")
    h, w = gray.shape
    scale = max_side / max(h, w)
    if scale < 1:
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)),
                          interpolation=cv2.INTER_AREA)
    return gray


def preprocess(gray, blur_ksize=BLUR_KSIZE):
    """Light Gaussian blur (odd kernel; 0 = off) to smooth the histogram."""
    if blur_ksize and blur_ksize > 1:
        return cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return gray


def otsu_threshold(gray):
    """Otsu's threshold from the 256-bin histogram.

    t* maximises the between-class variance
        sigma_B^2(t) = (mu_T * w(t) - mu(t))^2 / (w(t) * (1 - w(t)))
    with w(t) = sum_{i<=t} p_i, mu(t) = sum_{i<=t} i p_i, mu_T = mu(255).
    Returns (t, sigma_b2 curve of length 256).
    """
    p = np.bincount(gray.ravel(), minlength=256) / gray.size
    w = np.cumsum(p)
    mu = np.cumsum(np.arange(256) * p)
    denom = w * (1.0 - w)
    sigma_b2 = np.zeros(256)
    valid = denom > 1e-12                      # both classes non-empty
    sigma_b2[valid] = (mu[-1] * w[valid] - mu[valid]) ** 2 / denom[valid]
    return int(np.argmax(sigma_b2)), sigma_b2


def otsu_binarize(gray, blur_ksize=BLUR_KSIZE):
    """Blur, then Otsu. Dark ink on light paper -> ink = 255. Returns (binary, t)."""
    smoothed = preprocess(gray, blur_ksize)
    t, _ = otsu_threshold(smoothed)
    return np.where(smoothed <= t, 255, 0).astype(np.uint8), t


def run_otsu(img_or_path):
    """Full finalized Otsu process: load/resize -> blur -> Otsu. Returns binary."""
    binary, _ = otsu_binarize(load_gray(img_or_path))
    return binary
