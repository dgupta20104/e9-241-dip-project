r"""End-to-end pipeline: image path -> LaTeX for the whole image.

Two files are saved in outputs/pipeline/ per image:
    Output_<image_name>.tex                    the LaTeX of all lines (final output)
    Output_<image_name>_step3_line_boxes.png   deskewed binary with the line boxes,
                                               numbered like the lines in the .tex
Other intermediate results are returned in the dict but not written to disk.

Steps implemented:
    1. Thresholding  (adaptive Otsu, block / C always estimated per image)  -> step1_binary,
       step1_params; stroke width also stored as `stroke_width` for later steps
    2. Skew correction (projection profile on step1_binary, rotate that binary,
       nearest neighbour; no re-binarization)                              -> step2_binary
    3. Line segmentation (row projection profile + gap tolerance on step2_binary,
       thresholds = multiples of stroke_width)                             -> step3_lines,
       step3_info; then a per-line loop -> step3_line_crops (list of {index, box, crop});
       step3_line_boxes = boxes drawn on step2_binary
    4. Prediction (trained model, outputs/model/best.pt) on every line crop, inside the
       per-line loop -> step3_line_crops[k]["latex"], step4_line_latex (list, top->bottom);
       lines joined with the LaTeX new-line `\\` inside \begin{gathered} -> step4_latex

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
from src.model.predict import DEFAULT_WEIGHTS, load_model, predict_crops  # noqa: E402

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


# ---- Step 4: prediction with the trained model ---------------------------------
_MODEL_CACHE = {}      # weights path -> (model, vocab), so the model is loaded only once


def get_model(weights=DEFAULT_WEIGHTS):
    """Load the trained model and its vocabulary the first time, then reuse them."""
    key = str(weights)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(weights)
    return _MODEL_CACHE[key]


def step_predict(crop, model, vocab):
    """One line crop (binary, white ink 255 on black) -> LaTeX string for that line."""
    # predict_crops works on a list of images; we give it a list with just this crop.
    # ink_white=True because our crops have white ink on a black background.
    return predict_crops(model, vocab, [crop], ink_white=True)[0]


def combine_lines(line_latex):
    r"""List of per-line LaTeX strings (top -> bottom) -> one LaTeX block.

    Lines are separated by the LaTeX new-line token `\\`. They are wrapped in
    \begin{gathered} ... \end{gathered}, because `\\` alone does not break a
    line in math mode; inside `gathered` each line shows up as its own centred
    line (paste the block between $$ ... $$, needs the amsmath package).
    No lines -> empty string.
    """
    if not line_latex:
        return ""
    body = " \\\\\n".join(line_latex)          # " \\" + newline between lines, none after the last
    return "\\begin{gathered}\n" + body + "\n\\end{gathered}\n"


# ---- Visualisation of the line boxes -------------------------------------------
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
SAVED_IMAGE = "step3_line_boxes"       # image written next to the .tex file


def output_path(image_path, step_name):
    """outputs/pipeline/Output_<image_name>_<step_name>.png"""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    return OUTPUTS / f"Output_{Path(image_path).stem}_{step_name}.png"


def latex_path(image_path):
    """outputs/pipeline/Output_<image_name>.tex"""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    return OUTPUTS / f"Output_{Path(image_path).stem}.tex"


def run_pipeline(image_path, save=True, weights=DEFAULT_WEIGHTS):
    """Run all implemented steps on one image.

    Returns a dict {step name: output}. With save=True the final LaTeX
    (results["step4_latex"]) is written to outputs/pipeline/Output_<image_name>.tex
    and the line-box image to outputs/pipeline/Output_<image_name>_step3_line_boxes.png;
    other intermediate results stay in the dict only.
    `stroke_width` (px) is measured in step 1 and is valid for the step-1 and
    step-2 binaries (rotation keeps the scale); pass it to every later step.
    `step3_line_crops[k]` = {"index": k, "box": (y0, y1, x0, x1) half-open in
    step2_binary coordinates, "crop": line image trimmed to its ink columns,
    "latex": the model's LaTeX for that line}.
    `weights`: model checkpoint used in step 4 (loaded once, then reused).
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

    # Step 4 needs the trained model; load it once (reused on later calls).
    model, vocab = get_model(weights)

    # Per-line loop: Step 4 onwards runs here on each line crop
    # (pass results["stroke_width"] to every step that needs it).
    results["step3_line_crops"] = []
    results["step4_line_latex"] = []
    for k, (start, end) in enumerate(results["step3_lines"]):
        crop, box = crop_line(results["step2_binary"], start, end)
        latex = step_predict(crop, model, vocab)            # Step 4: this line -> LaTeX
        results["step3_line_crops"].append({"index": k, "box": box, "crop": crop,
                                            "latex": latex})
        results["step4_line_latex"].append(latex)

    # All lines, top to bottom, joined with the LaTeX new-line `\\` into one block.
    results["step4_latex"] = combine_lines(results["step4_line_latex"])

    results["step3_line_boxes"] = draw_line_boxes(results["step2_binary"],
                                                  results["step3_line_crops"])

    if save:
        latex_path(image_path).write_text(results["step4_latex"], encoding="utf-8")
        cv2.imwrite(str(output_path(image_path, SAVED_IMAGE)), results[SAVED_IMAGE])
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
    for k, latex in enumerate(res["step4_line_latex"]):
        print(f"  line {k}: {latex}")
    print("LaTeX:\n" + res["step4_latex"])
    print("Saved", latex_path(args.image))
    print("Saved", output_path(args.image, SAVED_IMAGE))
