"""Step 2 — Skew correction by projection profile.

The angle is estimated on the binary image (ink = 255) by maximising
S(theta) = sum_k p_theta(k)^2, where p_theta is the horizontal projection
profile of the ink coordinates rotated by theta. The same binary image is then
rotated (np.rot90 for the 90-degree part, nearest-neighbour inverse mapping in
NumPy for the small remainder), so it stays 0/255: no grayscale, no
interpolation, no re-binarization.
Spec: `Claude Chat Memory/Skew_Correction.md`.

Sign convention: theta > 0 means the text runs downwards to the right
(clockwise tilt on screen); `rotate_binary(binary, theta)` rotates
counter-clockwise by theta, which levels it.
"""

import numpy as np

# ---- Parameters ----------------------------------------------------------------
COARSE_STEP = 1.0    # deg, coarse search over [-90, 90)
FINE_STEP = 0.1      # deg, fine search ...
FINE_RANGE = 1.0     # ... within +-FINE_RANGE of the coarse best
R_MIN = 2       # weak-peak threshold on r = S(theta*) / median S (chosen by eye, notebook 03)


# ---- Angle estimation -------------------------------------------------------------
def ink_coords(binary):
    """(x, y) of all ink pixels, centred on the image centre (float64).

    No border / large-blob filter for now (page edges, spiral holes and the
    timestamp are kept).
    """
    ys, xs = np.nonzero(binary)
    h, w = binary.shape
    return xs - (w - 1) / 2.0, ys - (h - 1) / 2.0


def profile_score(x, y, theta_deg):
    """S(theta) = sum_k p_theta(k)^2 with y' = -x sin(theta) + y cos(theta), k = floor(y')."""
    t = np.deg2rad(theta_deg)
    k = np.floor(-x * np.sin(t) + y * np.cos(t)).astype(np.int64)
    p = np.bincount(k - k.min())
    return float(np.dot(p, p))


def wrap_angle(theta_deg):
    """Map an angle into [-90, 90) (S has period 180 deg)."""
    return (theta_deg + 90.0) % 180.0 - 90.0


def estimate_skew(binary, r_min=R_MIN):
    """Projection-profile skew estimate.

    Coarse COARSE_STEP search over [-90, 90), then FINE_STEP within +-FINE_RANGE
    of the best (wrapped mod 180, so the range has no edge).
    Peak sharpness r = S(theta*) / median of the coarse S curve; if r < r_min
    (or there is no ink) fall back to theta = 0.

    Returns dict: theta (angle to use), theta_raw (argmax), r, fallback,
    angles / scores (coarse curve), fine_angles / fine_scores.
    """
    x, y = ink_coords(binary)
    angles = np.arange(-90.0, 90.0, COARSE_STEP)
    if x.size == 0:
        return {"theta": 0.0, "theta_raw": 0.0, "r": 0.0, "fallback": True,
                "angles": angles, "scores": np.zeros_like(angles),
                "fine_angles": np.array([]), "fine_scores": np.array([])}

    scores = np.array([profile_score(x, y, a) for a in angles])
    best = angles[np.argmax(scores)]
    n_fine = int(round(FINE_RANGE / FINE_STEP))
    fine_angles = best + FINE_STEP * np.arange(-n_fine, n_fine + 1)
    fine_scores = np.array([profile_score(x, y, a) for a in fine_angles])
    i = int(np.argmax(fine_scores))
    theta_raw = float(round(wrap_angle(fine_angles[i]), 6))

    r = float(fine_scores[i] / np.median(scores))
    fallback = r < r_min
    return {"theta": 0.0 if fallback else theta_raw, "theta_raw": theta_raw, "r": r,
            "fallback": fallback, "angles": angles, "scores": scores,
            "fine_angles": fine_angles, "fine_scores": fine_scores}


# ---- Rotation ------------------------------------------------------------------
def split_angle(theta_deg):
    """theta = 90 k + rem with |rem| <= 45."""
    k = int(round(theta_deg / 90.0))
    return k, theta_deg - 90.0 * k


def rotate_binary(binary, theta_deg):
    """Rotate a 0/255 image counter-clockwise by theta, keeping it 0/255.

    90 k part: np.rot90 (lossless). Remainder: inverse mapping with nearest
    neighbour on an enlarged canvas W' = W|cos| + H|sin|, H' = W|sin| + H|cos|:
    each output pixel (x', y') takes the value of the input pixel nearest to
    R(-rem) (x', y') (about the centres). Pixels mapping outside the input are
    paper (0), so no false ink at the corners. Equivalent to cv2.warpAffine
    with INTER_NEAREST.
    """
    k, rem = split_angle(theta_deg)
    out = np.ascontiguousarray(np.rot90(binary, k)) if k % 4 else binary.copy()
    if abs(rem) < 1e-9:
        return out
    h, w = out.shape
    t = np.deg2rad(rem)
    c, s = np.cos(t), np.sin(t)
    W, H = int(np.ceil(w * abs(c) + h * abs(s))), int(np.ceil(w * abs(s) + h * abs(c)))
    yo, xo = np.mgrid[0:H, 0:W].astype(np.float64)
    xo -= (W - 1) / 2.0                               # output coords about the canvas centre
    yo -= (H - 1) / 2.0
    # forward CCW on screen (y down): x' = x c + y s, y' = -x s + y c  ->  inverse:
    xs = np.rint(xo * c - yo * s + (w - 1) / 2.0).astype(np.int64)
    ys = np.rint(xo * s + yo * c + (h - 1) / 2.0).astype(np.int64)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    rot = np.zeros((H, W), binary.dtype)
    rot[inside] = out[ys[inside], xs[inside]]
    return rot


def deskew(binary, r_min=R_MIN):
    """Estimate the skew on `binary` and rotate that same binary.

    Returns (rotated binary, info). Does not re-threshold (no adaptive_otsu call).
    """
    info = estimate_skew(binary, r_min)
    return rotate_binary(binary, info["theta"]), info
