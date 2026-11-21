"""End-to-end pipeline: image path -> output of every step implemented so far.

Each step's image result is saved directly in outputs/ as
`Output_<image_name>_stepN_<name>.png`, one file per stage.

Steps implemented:
    1. Thresholding  (adaptive mean; optional per-image parameter estimation) -> step1_binary
    2. Skew correction (projection profile on step1_binary, rotate that binary,
       nearest neighbour; no re-binarization)                              -> step2_binary

Add each new step as a `step_*` function below and call it in `run_pipeline`.

Usage:
    from src.pipeline import run_pipeline
    results = run_pipeline("data/raw/photos/xyz.jpg")

    # or from the project root:
    python -m src.pipeline data/raw/photos/xyz.jpg [--estimate-params]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # allow `python src/pipeline.py` / IDE Run button
    sys.path.insert(0, str(ROOT))

from src.adaptive_thresholding import run_adaptive  # noqa: E402
from src.skew_correction import deskew  # noqa: E402

OUTPUTS = ROOT / "outputs"


# ---- Step 1: thresholding ----------------------------------------------------
def step_threshold(image_path, estimate_params=False):
    """Image path -> binary image (ink = 255) via adaptive mean thresholding.

    estimate_params=False: frozen BLOCK / C.
    estimate_params=True:  block / C estimated for this image.
    """
    return run_adaptive(image_path, estimate_params=estimate_params)


# ---- Step 2: skew correction ---------------------------------------------------
def step_deskew(binary):
    """Binary in -> (rotated binary, info): angle and rotation both on the binary."""
    return deskew(binary)


# ---- Full pipeline -----------------------------------------------------------
def output_path(image_path, step_name):
    """outputs/Output_<image_name>_<step_name>.png"""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    return OUTPUTS / f"Output_{Path(image_path).stem}_{step_name}.png"


def run_pipeline(image_path, estimate_params=False, save=True):
    """Run all implemented steps on one image.

    estimate_params: estimate adaptive-threshold block / C per image.
    Returns a dict {step name: output}. With save=True each image output is
    also written to outputs/Output_<image_name>_<step name>.png
    (non-image entries such as `step2_info` are not saved).
    """
    image_path = Path("/Users/dgupta/IISc/SEM 1/DIP/Project/" + image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = {}
    results["step1_binary"] = step_threshold(image_path, estimate_params)
    results["step2_binary"], results["step2_info"] = step_deskew(results["step1_binary"])

    if save:
        for name, img in results.items():
            if isinstance(img, np.ndarray):
                cv2.imwrite(str(output_path(image_path, name)), img)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the DIP pipeline on one image.")
    parser.add_argument("image", help="path to the input image")
    parser.add_argument("--estimate-params", action="store_true",
                        help="estimate adaptive-threshold block / C for this image")
    args = parser.parse_args()

    res = run_pipeline(args.image, estimate_params=args.estimate_params)
    info = res["step2_info"]
    print(f"Skew: theta* = {info['theta_raw']:.1f} deg, r = {info['r']:.3f}, "
          f"fallback = {info['fallback']} -> rotated by {info['theta']:.1f} deg")
    for name, img in res.items():
        if isinstance(img, np.ndarray):
            print("Saved", output_path(args.image, name))
