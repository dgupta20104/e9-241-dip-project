"""Step 1b — Adaptive-mean thresholding, written from scratch.

A pixel is ink if  I < local_mean - C,  with the local mean taken over a
block x block window via an integral image. No morphology.
Spec: `Claude Chat Memory/Adaptive_Thresholding.md`.

Pipeline entry point: `run_adaptive(img_or_path)` -> uint8 binary, ink = 255.
"""

import math

import cv2
import numpy as np

from src.thresholding import otsu_binarize


def odd(n):
    """Round to the nearest odd integer >= 3."""
    n = int(round(n))
    return max(3, n if n % 2 else n + 1)


# ---- Frozen parameters (from the sweep in notebooks/02_adaptive_thresholding.ipynb)
HEIGHT = 1000              # image height after resizing (px)
BLUR_SIGMA = 1.0           # light Gaussian pre-blur
STROKE_W = 4               # stroke width at HEIGHT (px), eye-checked
BLOCK = odd(8 * STROKE_W)  # 33: block ~ 8 x stroke width
C = 6                      # ~2.5 x blank-paper std (2.4)


# ---- Preprocessing ---------------------------------------------------------
def to_gray(img):
    """BGR -> luminance 0.299 R + 0.587 G + 0.114 B (uint8)."""
    if img.ndim == 2:
        return img
    gray = img[..., :3].astype(np.float64) @ np.array([0.114, 0.587, 0.299])
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def gaussian_blur(gray, sigma=BLUR_SIGMA):
    """Separable Gaussian blur (size 2*ceil(3 sigma)+1, reflect borders) -> uint8."""
    r = math.ceil(3 * sigma)
    k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma ** 2))
    k /= k.sum()
    img = gray.astype(np.float64)
    for axis in (1, 0):                                   # rows, then columns
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = np.pad(img, pad, mode="reflect")
        n = img.shape[axis]
        img = sum(kv * p.take(range(i, i + n), axis=axis) for i, kv in enumerate(k))
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def prepare(img_or_path, height=HEIGHT, blur_sigma=BLUR_SIGMA):
    """Path or array -> grayscale -> resize to fixed height -> light blur (uint8)."""
    img = img_or_path
    if not isinstance(img, np.ndarray):
        img = cv2.imread(str(img_or_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_or_path}")
    gray = to_gray(img)
    h, w = gray.shape
    if h != height:
        interp = cv2.INTER_AREA if height < h else cv2.INTER_LINEAR
        gray = cv2.resize(gray, (max(1, round(w * height / h)), height), interpolation=interp)
    return gaussian_blur(gray, blur_sigma) if blur_sigma > 0 else gray


# ---- Adaptive mean ---------------------------------------------------------
def local_mean(gray, block):
    """Box mean over block x block via an int64 integral image (reflect padding)."""
    if block < 3 or block % 2 == 0:
        raise ValueError(f"block must be odd and >= 3, got {block}")
    r = block // 2
    p = np.pad(gray.astype(np.int64), r, mode="reflect")
    ii = np.zeros((p.shape[0] + 1, p.shape[1] + 1), np.int64)
    ii[1:, 1:] = p.cumsum(0).cumsum(1)
    h, w = gray.shape
    s = (ii[block:block + h, block:block + w] - ii[:h, block:block + w]
         - ii[block:block + h, :w] + ii[:h, :w])
    return s / (block * block)


def adaptive_mean(gray, block=BLOCK, C=C):
    """Ink where gray < local_mean - C. Returns (binary ink=255, threshold map T)."""
    T = local_mean(gray, block) - C
    return np.where(gray < T, 255, 0).astype(np.uint8), T


def run_adaptive(img_or_path, height=HEIGHT, estimate_params=False):
    """Full process: grayscale -> resize -> blur -> adaptive mean.

    estimate_params=False: use the frozen BLOCK / C.
    estimate_params=True:  estimate block and C for this image (`estimate_image_params`).
    Returns the binary image (ink = 255).
    """
    gray = prepare(img_or_path, height)
    if estimate_params:
        p = estimate_image_params(gray)
        binary, _ = adaptive_mean(gray, p["block"], p["C"])
    else:
        binary, _ = adaptive_mean(gray)
    return binary


# ---- Parameter estimation ---------------------------------------------------
def estimate_noise_sigma(gray, mask, win):
    """sigma_n = median local std over win x win windows that contain no ink.

    Same idea as the std of a hand-picked blank paper patch, done automatically.
    Use win = block, so sigma_n covers the same variation (noise + shading)
    that the threshold sees around the local mean. None if no ink-free window.
    """
    mean = local_mean(gray, win)
    var = local_mean(gray.astype(np.int64) ** 2, win) - mean ** 2
    blank = local_mean((mask > 0).astype(np.int64), win) == 0
    if not blank.any():
        return None
    return float(np.median(np.sqrt(np.maximum(var[blank], 0))))


def estimate_image_params(gray):
    """Per-image stroke width, noise, block and C.

    A first pass with the frozen parameters gives the ink mask; from it:
    w = median ink-run length, sigma_n = median std of ink-free windows,
    block = odd(8 w), C = 2.5 sigma_n. Falls back to frozen values if needed.
    """
    mask, _ = adaptive_mean(gray)
    try:
        w = estimate_stroke_width(gray, mask=mask)
    except ValueError:
        w = STROKE_W
    block = odd(8 * w)
    sigma_n = estimate_noise_sigma(gray, mask, block)
    c = max(1.0, round(2.5 * sigma_n, 1)) if sigma_n is not None else C
    return {"stroke_w": w, "sigma_n": sigma_n, "block": block, "C": c, "estimated": True}


def estimate_stroke_width(gray, mask=None, n_rows=20, n_cols=20, seed=0, return_runs=False):
    """w = median ink-run length in ~20 random rows/cols of an ink mask.

    mask=None uses the baseline Otsu mask (spec); `gray` is the prepared
    (already blurred) image, so Otsu runs without extra blur and `w` is
    measured at the adaptive-thresholding scale.
    """
    if mask is None:
        mask, _ = otsu_binarize(gray, blur_ksize=0)
    rng = np.random.default_rng(seed)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    rows = rng.choice(rows, min(n_rows, len(rows)), replace=False)
    cols = rng.choice(cols, min(n_cols, len(cols)), replace=False)

    def runs_of(line):
        edges = np.flatnonzero(np.diff(np.r_[0, line > 0, 0].astype(np.int8)))
        return edges[1::2] - edges[::2]

    runs = np.concatenate([runs_of(mask[r]) for r in rows] + [runs_of(mask[:, c]) for c in cols] + [[]])
    if runs.size == 0:
        raise ValueError("Otsu mask contains no ink; cannot estimate stroke width")
    w = float(np.median(runs))
    return (w, runs, mask, rows, cols) if return_runs else w
