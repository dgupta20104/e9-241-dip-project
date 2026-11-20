"""Step 1 — Thresholding (binarization) of handwritten expression images.

All binarization functions return a uint8 image with ink = 255 and
background = 0, which is what connected-component analysis expects later.
"""

import cv2
import numpy as np


def load_gray(path, max_side=1200):
    """Read an image as grayscale uint8, shrinking it so max(h, w) <= max_side.

    Fixing the size keeps blur/block sizes meaningful across phone photos
    of different resolutions.
    """
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = gray.shape
    scale = max_side / max(h, w)
    if scale < 1:
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)),
                          interpolation=cv2.INTER_AREA)
    return gray


def preprocess(gray, blur_ksize=5):
    """Light Gaussian blur to suppress noise and smooth the histogram.

    blur_ksize = 0 disables blurring; otherwise it must be odd.
    """
    if blur_ksize and blur_ksize > 1:
        return cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return gray


def otsu_threshold(gray):
    """Otsu's threshold computed from the 256-bin histogram.

    For every candidate t, pixels <= t form class 0 (ink) and > t class 1
    (paper). Otsu picks t maximising the between-class variance
        sigma_B^2(t) = (mu_T * w(t) - mu(t))^2 / (w(t) * (1 - w(t)))
    where w(t) = sum_{i<=t} p_i, mu(t) = sum_{i<=t} i * p_i, mu_T = mu(255).

    Returns (t, sigma_b2): the threshold (int) and the full sigma_B^2 curve
    (length 256; 0 where one class is empty).
    """
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()

    levels = np.arange(256)
    w = np.cumsum(p)            # class-0 probability  w(t)
    mu = np.cumsum(levels * p)  # class-0 first moment mu(t)
    mu_t = mu[-1]               # global mean

    denom = w * (1.0 - w)
    sigma_b2 = np.zeros(256)
    valid = denom > 1e-12       # both classes non-empty
    sigma_b2[valid] = (mu_t * w[valid] - mu[valid]) ** 2 / denom[valid]

    t = int(np.argmax(sigma_b2))
    return t, sigma_b2


def otsu_binarize(gray, blur_ksize=5):
    """Blur, then binarize with Otsu. Dark strokes on light paper -> ink = 255.

    Returns (binary, t).
    """
    smoothed = preprocess(gray, blur_ksize)
    t, _ = otsu_threshold(smoothed)
    binary = np.where(smoothed <= t, 255, 0).astype(np.uint8)
    return binary, t
