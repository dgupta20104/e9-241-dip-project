"""End-to-end pipeline: image path -> output of every step implemented so far.

Each step's image result is saved directly in outputs/ as
`Output_<image_name>_stepN_<name>.png`, one file per stage.

Steps implemented:
    1. Thresholding  (adaptive mean, block / C always estimated per image)  -> step1_binary,
       step1_params; stroke width also stored as `stroke_width` for later steps
    2. Skew correction (projection profile on step1_binary, rotate that binary,
       nearest neighbour; no re-binarization)                              -> step2_binary

Add each new step as a `step_*` function below and call it in `run_pipeline`.

Usage:
    from src.pipeline import run_pipeline
    results = run_pipeline("data/raw/photos/xyz.jpg")

    # or from the project root:
    python -m src.pipeline data/raw/photos/xyz.jpg
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
def step_threshold(image_path):
    """Image path -> (binary ink = 255, params) via adaptive mean thresholding.

    Block / C are always estimated for this image. params holds stroke_w,
    sigma_n, block, C; stroke_w is in px at the binary's scale.
    """
    return run_adaptive(image_path, return_params=True)


# ---- Step 2: skew correction ---------------------------------------------------
def step_deskew(binary):
    """Binary in -> (rotated binary, info): angle and rotation both on the binary."""
    return deskew(binary)


# ---- Full pipeline -----------------------------------------------------------
def output_path(image_path, step_name):
    """outputs/Output_<image_name>_<step_name>.png"""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    return OUTPUTS / f"Output_{Path(image_path).stem}_{step_name}.png"


def run_pipeline(image_path, save=True):
    """Run all implemented steps on one image.

    Returns a dict {step name: output}. With save=True each image output is
    also written to outputs/Output_<image_name>_<step name>.png
    (non-image entries such as `step1_params`, `stroke_width`, `step2_info`
    are not saved). `stroke_width` (px) is measured in step 1 and is valid
    for the step-1 and step-2 binaries (rotation keeps the scale).
    """
    image_path = Path("/Users/dgupta/IISc/SEM 1/DIP/Project/" + image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = {}
    results["step1_binary"], results["step1_params"] = step_threshold(image_path)
    results["stroke_width"] = results["step1_params"]["stroke_w"]
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
    args = parser.parse_args()

    res = run_pipeline(args.image)
    p = res["step1_params"]
    print(f"Threshold: stroke width = {res['stroke_width']:g} px, block = {p['block']}, "
          f"C = {p['C']}, sigma_n = {p['sigma_n']}")
    info = res["step2_info"]
    print(f"Skew: theta* = {info['theta_raw']:.1f} deg, r = {info['r']:.3f}, "
          f"fallback = {info['fallback']} -> rotated by {info['theta']:.1f} deg")
    for name, img in res.items():
        if isinstance(img, np.ndarray):
            print("Saved", output_path(args.image, name))
