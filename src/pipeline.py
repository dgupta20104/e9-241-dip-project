"""End-to-end pipeline: image path -> output of every step implemented so far.

Only the output of the latest step is saved, in outputs/pipeline/ as
`Output_<image_name>_<name>.png` (currently `..._step3_line_boxes.png`:
the deskewed binary with the step-3 line boxes drawn). Intermediate
results are returned in the dict but not written to disk.

Steps implemented:
    1. Thresholding  (adaptive Otsu, block / C always estimated per image)  -> step1_binary,
       step1_params; stroke width also stored as `stroke_width` for later steps
    2. Skew correction (projection profile on step1_binary, rotate that binary,
       nearest neighbour; no re-binarization)                              -> step2_binary
    3. Line segmentation (row projection profile + gap tolerance on step2_binary,
       thresholds = multiples of stroke_width)                             -> step3_lines,
       step3_info; then a per-line loop -> step3_line_crops (list of {index, box, crop});
       step3_line_boxes = boxes drawn on step2_binary (the saved image)

Add each new step as a `step_*` function below and call it in `run_pipeline`.

Usage:
    from src.pipeline import run_pipeline
    results = run_pipeline("data/photos/xyz.jpg")

    # or from the project root:
    python -m src.pipeline data/photos/xyz.jpg
"""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # allow `python src/pipeline.py` / IDE Run button
    sys.path.insert(0, str(ROOT))

from src.adaptive_thresholding import run_adaptive  # noqa: E402
from src.skew_correction import deskew  # noqa: E402
from src.line_segmentation import crop_line, segment_lines  # noqa: E402

OUTPUTS = ROOT / "outputs" / "pipeline"


# ---- Step 1: thresholding ----------------------------------------------------
def step_threshold(image_path):
    """Image path -> (binary ink = 255, params) via adaptive (local) Otsu thresholding.

    Block / C are always estimated for this image. params holds stroke_w,
    sigma_n, block, C; stroke_w is in px at the binary's scale.
    """
    return run_adaptive(image_path, return_params=True)


# ---- Step 2: skew correction ---------------------------------------------------
def step_deskew(binary):
    """Binary in -> (rotated binary, info): angle and rotation both on the binary."""
    return deskew(binary)


# ---- Step 3: line segmentation --------------------------------------------------
def step_lines(binary, stroke_width):
    """Deskewed binary -> (lines, info); lines = [(start, end), ...] half-open rows.

    Uses the step-1 stroke width (not re-estimated).
    """
    return segment_lines(binary, stroke_width)


# ---- Visualisation of the final output -----------------------------------------
def draw_line_boxes(binary, line_crops):
    """Binary (ink = 255) -> BGR image with each line's box and index drawn.

    Boxes are the half-open (y0, y1, x0, x1) from `step3_line_crops`.
    """
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for item in line_crops:
        y0, y1, x0, x1 = item["box"]
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)
        cv2.putText(vis, str(item["index"]), (x0 + 4, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return vis


# ---- Full pipeline -----------------------------------------------------------
SAVED_OUTPUT = "step3_line_boxes"      # the only image written to outputs/pipeline/


def output_path(image_path, step_name):
    """outputs/pipeline/Output_<image_name>_<step_name>.png"""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    return OUTPUTS / f"Output_{Path(image_path).stem}_{step_name}.png"


def run_pipeline(image_path, save=True):
    """Run all implemented steps on one image.

    Returns a dict {step name: output}. With save=True only
    results[SAVED_OUTPUT] (the latest step's output) is written to
    outputs/pipeline/Output_<image_name>_<SAVED_OUTPUT>.png; intermediate results
    stay in the dict only.
    `stroke_width` (px) is measured in step 1 and is valid for the step-1 and
    step-2 binaries (rotation keeps the scale); pass it to every later step.
    `step3_line_crops[k]` = {"index": k, "box": (y0, y1, x0, x1) half-open in
    step2_binary coordinates, "crop": line image trimmed to its ink columns}.
    """
    image_path = Path("/Users/dgupta/IISc/SEM 1/DIP/Project/" + image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = {}
    results["step1_binary"], results["step1_params"] = step_threshold(image_path)
    results["stroke_width"] = results["step1_params"]["stroke_w"]
    results["step2_binary"], results["step2_info"] = step_deskew(results["step1_binary"])
    results["step3_lines"], results["step3_info"] = step_lines(results["step2_binary"],
                                                               results["stroke_width"])

    # Per-line loop: Step 4 onwards runs here on each line crop
    # (pass results["stroke_width"] to every step).
    results["step3_line_crops"] = []
    for k, (start, end) in enumerate(results["step3_lines"]):
        crop, box = crop_line(results["step2_binary"], start, end)
        results["step3_line_crops"].append({"index": k, "box": box, "crop": crop})

    results["step3_line_boxes"] = draw_line_boxes(results["step2_binary"],
                                                  results["step3_line_crops"])

    if save:
        cv2.imwrite(str(output_path(image_path, SAVED_OUTPUT)), results[SAVED_OUTPUT])
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
    li = res["step3_info"]
    print(f"Lines: G_max = {li['gap_max']} px, H_min = {li['min_h']} px -> "
          f"{len(res['step3_lines'])} lines {res['step3_lines']}")
    print("Saved", output_path(args.image, SAVED_OUTPUT))
