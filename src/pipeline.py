"""End-to-end pipeline: image path -> output of every step implemented so far.

Each step's result is saved in  outputs/Output_<image_name>/  as
`stepN_<name>.png`, so the folder shows the image after every stage.

Steps implemented:
    1. Thresholding  (adaptive mean by default, Otsu optional) -> binary, ink = 255

Add each new step as a `step_*` function below and call it in `run_pipeline`.

Usage:
    from src.pipeline import run_pipeline
    results = run_pipeline("data/raw/photos/xyz.jpg")

    # or from the project root:
    python -m src.pipeline data/raw/photos/xyz.jpg [--method otsu|adaptive|adaptive_est]
"""

from pathlib import Path

import cv2

from src.adaptive_thresholding import run_adaptive
from src.thresholding import run_otsu

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


# ---- Step 1: thresholding ----------------------------------------------------
def step_threshold(image_path, method="adaptive"):
    """Image path -> binary image (ink = 255).

    method: "adaptive"     adaptive mean with frozen BLOCK / C (default)
            "adaptive_est" adaptive mean with per-image estimated block / C
            "otsu"         global Otsu
    """
    if method == "adaptive":
        return run_adaptive(image_path)
    if method == "adaptive_est":
        return run_adaptive(image_path, estimate_params=True)
    if method == "otsu":
        return run_otsu(image_path)
    raise ValueError(f"Unknown thresholding method: {method!r}")


# ---- Full pipeline -----------------------------------------------------------
def output_dir(image_path):
    """outputs/Output_<image_name>/ (created if missing)."""
    d = OUTPUTS / f"Output_{Path(image_path).stem}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_pipeline(image_path, method="adaptive", save=True):
    """Run all implemented steps on one image.

    Returns a dict {step name: output}. With save=True each output is also
    written to outputs/Output_<image_name>/.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = {}
    results["step1_binary"] = step_threshold(image_path, method)

    if save:
        out = output_dir(image_path)
        for name, img in results.items():
            cv2.imwrite(str(out / f"{name}.png"), img)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the DIP pipeline on one image.")
    parser.add_argument("image", help="path to the input image")
    parser.add_argument("--method", default="adaptive",
                        choices=["adaptive", "adaptive_est", "otsu"],
                        help="thresholding method (default: adaptive)")
    args = parser.parse_args()

    res = run_pipeline(args.image, method=args.method)
    print(f"Saved {list(res)} to {output_dir(args.image)}")
