"""Use the trained model: line crops / images -> LaTeX strings.

Used by src/pipeline.py (Step 4: one call per line crop), or directly:

    from src.model.predict import load_model, predict_crops
    model, vocab = load_model()                       # once, loads outputs/model/best.pt
    crops = [item["crop"] for item in results["step3_line_crops"]]
    latex_lines = predict_crops(model, vocab, crops)  # one LaTeX string per line crop

Pipeline crops are binary with WHITE ink (255) on black - that is the default
here (ink_white=True). For other images (e.g. CROHME renders: dark ink on white)
pass ink_white=False, or None to let it guess. Photos must go through Steps 1-3
first; this only binarizes with a fixed threshold of 128.

Command line (from the project root):
    python -m src.model.predict data/processed/crohme/test/images/<id>.png
    python -m src.model.predict a.png b.png --weights outputs/model/best.pt --ink black
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:          # allow `python src/model/predict.py` / IDE Run button
    sys.path.insert(0, str(ROOT))

from src.model.model import Im2LatexModel, get_device, pad_images, prepare_image  # noqa: E402
from src.model.vocab import Vocab, to_latex  # noqa: E402

DEFAULT_WEIGHTS = ROOT / "outputs" / "model" / "best.pt"
MAX_LEN = 150          # stop writing after this many tokens


def load_model(weights=DEFAULT_WEIGHTS, device=None):
    """Checkpoint file -> (model ready to predict, vocab).

    The vocabulary is read from the checkpoint itself, so the token indices
    always match the weights.
    """
    device = get_device(device)
    ckpt = torch.load(weights, map_location=device, weights_only=True)
    vocab = Vocab(ckpt["vocab"])
    model = Im2LatexModel(**ckpt["config"])          # same architecture as in training
    model.load_state_dict(ckpt["model"])             # trained weights
    model.to(device).eval()                          # eval: BatchNorm uses its stored statistics
    return model, vocab


@torch.no_grad()
def predict_crops(model, vocab, crops, ink_white=True, batch_size=16, max_len=MAX_LEN):
    """List of images (numpy) -> list of LaTeX strings, same order.

    Images are prepared exactly like in training (crop to ink, height 64)
    and read in batches of `batch_size`.
    """
    device = next(model.parameters()).device
    inputs = [prepare_image(c, ink_white) for c in crops]
    latex = []
    for b in range(0, len(inputs), batch_size):
        x, widths = pad_images(inputs[b:b + batch_size])
        out = model.greedy_decode(x.to(device), widths.to(device), max_len)
        latex += [to_latex(vocab.decode(ids)) for ids in out.cpu().tolist()]
    return latex


def predict_image(model, vocab, path, ink_white=None):
    """One image file -> LaTeX string."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return predict_crops(model, vocab, [img], ink_white)[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Image(s) -> LaTeX with the trained model.")
    ap.add_argument("images", nargs="+", help="image file(s), one expression line each")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="checkpoint (.pt)")
    ap.add_argument("--ink", choices=["auto", "white", "black"], default="auto",
                    help="colour of the ink in the images (auto = guess)")
    ap.add_argument("--device", default=None, help="force cuda / mps / cpu")
    args = ap.parse_args(argv)

    ink_white = {"auto": None, "white": True, "black": False}[args.ink]
    model, vocab = load_model(args.weights, args.device)
    for path in args.images:
        print(f"{path}\t{predict_image(model, vocab, path, ink_white)}")


if __name__ == "__main__":
    main()
