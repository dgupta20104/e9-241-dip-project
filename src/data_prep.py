"""Step 5 (data preparation): CROHME 2019 InkML -> rendered images + ground truth.

CROHME is online ink (pen strokes), so every expression is rendered to a
raster image here. Nothing in Steps 1-4 is changed; this module only reads
`data/raw/Crohome Dataset/` and writes `data/processed/crohme/`.

Output (per split in train / valid / test, iterated from the split lists):
    data/processed/crohme/<split>/images/<id>.png   uint8 grayscale, dark ink (0)
                                                    on white paper (255), anti-aliased
    data/processed/crohme/<split>.jsonl             one JSON record per expression
    data/processed/crohme/<split>_truth.tsv         quick view: image \t latex
                                                    (whitespace collapsed to one space)

Record fields (`<split>.jsonl`):
    id, split, inkml (path rel. to project root), image (rel. to data/processed/crohme),
    height, width, latex (truth without the $...$), truth_raw, category (or null),
    layout (symbol layout tree from the split list), thickness (px), scale (px per ink
    unit), ink_min ([xmin, ymin] in ink units), offset ([ox, oy]):
    pixel = (ink - ink_min) * scale + offset,
    symbols: [{label, trace_ids, box}] in InkML traceGroup order (NOT left to right;
    sort by box x0 if needed), orphan_trace_ids (ink in no symbol group, still
    drawn), missing_refs (traceDataRef ids with no trace, skipped).

Boxes are half-open `[y0, y1, x0, x1]` in image pixels (same convention as
`crop_line` in src/line_segmentation.py) and include the stroke radius, so thin
symbols like `-` get a real height. Note: `run_adaptive` resizes to height
1000, so scale boxes by 1000 / height before comparing with pipeline output.

Rendering:
    * One uniform scale per expression (no aspect change), chosen so that the
      median symbol size max(w, h) = SYMBOL_PX. Raw coordinates differ by ~5
      orders of magnitude between sources (formulaire ~1 unit, KME ~10^4).
    * Stroke thickness 2-4 px, fixed per file from a hash of its id (reproducible).
    * Traces shorter than 1 px after scaling (dots of i, j, !, ., \\ldots) are
      drawn as filled discs, since a zero-length polyline may draw nothing.
    * No blur / noise: the image is clean, so `image < 128` doubles as the
      pixel-level ground-truth binary for Step 1.

Usage (from the project root):
    python -m src.data_prep                       # all splits
    python -m src.data_prep --splits valid --limit 20

    from src.data_prep import load_split, parse_inkml, render_expression
    records = load_split("valid")
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "Crohome Dataset"   # split lists live here
OUT = ROOT / "data" / "processed" / "crohme"
SPLITS = ("train", "valid", "test")

SYMBOL_PX = 48          # median symbol size max(w, h) after scaling, px
PAD = 24                # white border around the ink, px
THICKNESSES = (2, 3, 4)  # stroke width, picked per file
MAX_SIDE = 6000         # safety cap on the rendered image side, px
# Inherently tiny / flat symbols, ignored when measuring the scale (e.g. `\cdots + 4`
# = three dots + two symbols would otherwise get the dot size as its median).
SMALL_SYMBOLS = {".", ",", "-", "\\prime", "\\ldots", "\\cdots", "\\cdot"}
SHIFT = 4               # cv2 fixed-point bits (sub-pixel drawing)


# ---- InkML parsing ------------------------------------------------------------
def _local(tag):
    """'{http://www.w3.org/2003/InkML}trace' -> 'trace' (files differ in namespaces)."""
    return tag.rsplit("}", 1)[-1]


def _children(el, name):
    return [c for c in el if _local(c.tag) == name]


def _annotation(el, kind):
    """Text of the direct-child <annotation type=kind> of el, or None."""
    for a in _children(el, "annotation"):
        if a.get("type") == kind:
            return (a.text or "").strip() or None
    return None


def _parse_trace(text):
    """'x y [t], x y [t], ...' -> (n, 2) float array (extra channels dropped)."""
    pts = []
    for p in text.strip().split(","):
        v = p.split()
        if len(v) >= 2:
            pts.append((float(v[0]), float(v[1])))
    return np.array(pts, dtype=np.float64).reshape(-1, 2)


def parse_inkml(path):
    """InkML file -> dict(truth_raw, latex, category, traces, symbols, orphan_trace_ids,
    missing_refs).

    traces: {trace id: (n, 2) array of x, y}, in file order.
    symbols: [{label, trace_ids}] from every traceGroup that has traceViews (the
    outer "From ITF"/"Segmentation" group has none and is skipped).
    The expression truth is read only from the root's direct-child annotation
    (nested traceGroups also carry type="truth").
    """
    root = ET.parse(path).getroot()

    traces = {}
    for t in root.iter():
        if _local(t.tag) == "trace" and t.text and t.text.strip():
            pts = _parse_trace(t.text)
            if len(pts):
                traces[t.get("id")] = pts

    symbols, used, missing = [], set(), []
    for g in root.iter():
        if _local(g.tag) != "traceGroup":
            continue
        refs = [v.get("traceDataRef") for v in _children(g, "traceView")]
        if not refs:
            continue
        ok = [r for r in refs if r in traces]
        missing += [r for r in refs if r not in traces]
        if ok:
            symbols.append({"label": _annotation(g, "truth"), "trace_ids": ok})
            used.update(ok)

    truth_raw = _annotation(root, "truth") or ""
    return {
        "truth_raw": truth_raw,
        "latex": truth_raw.strip().strip("$").strip(),
        "category": _annotation(root, "category"),
        "traces": traces,
        "symbols": symbols,
        "orphan_trace_ids": [k for k in traces if k not in used],
        "missing_refs": missing,
    }


# ---- Rendering ------------------------------------------------------------------
def thickness_for(expr_id):
    """Deterministic stroke width in THICKNESSES from the expression id."""
    return THICKNESSES[zlib.crc32(expr_id.encode()) % len(THICKNESSES)]


def ink_scale(parsed, symbol_px=SYMBOL_PX):
    """px per ink unit so that the median symbol size max(w, h) = symbol_px.

    Symbols in SMALL_SYMBOLS are left out of the median unless nothing else is
    left; falls back to the whole expression's size when no symbol has extent
    (e.g. a lone dot).
    """
    traces = parsed["traces"]
    sizes, small = [], []
    for s in parsed["symbols"]:
        pts = np.concatenate([traces[i] for i in s["trace_ids"]])
        size = np.ptp(pts, axis=0).max()
        if size > 0:
            (small if s["label"] in SMALL_SYMBOLS else sizes).append(size)
    sizes = sizes or small
    if sizes:
        unit = float(np.median(sizes))
    else:
        unit = float(np.ptp(np.concatenate(list(traces.values())), axis=0).max())
    return symbol_px / unit if unit > 0 else 1.0


def render_expression(parsed, thickness=3, symbol_px=SYMBOL_PX, pad=PAD):
    """Parsed InkML -> (image, symbols with boxes, transform).

    image: uint8, ink 0 on paper 255. Symbol boxes are half-open [y0, y1, x0, x1]
    including the stroke radius. transform = {scale, offset, xmin, ymin}:
    pixel = (ink - [xmin, ymin]) * scale + offset.
    """
    traces = parsed["traces"]
    allpts = np.concatenate(list(traces.values()))
    lo = allpts.min(axis=0)

    scale = ink_scale(parsed, symbol_px)
    extent = np.ptp(allpts, axis=0) * scale
    if extent.max() + 2 * pad > MAX_SIDE:            # pathological sizes only
        scale *= (MAX_SIDE - 2 * pad) / extent.max()
        extent = np.ptp(allpts, axis=0) * scale
    offset = pad + thickness / 2.0

    width = int(np.ceil(extent[0] + 2 * offset)) + 1
    height = int(np.ceil(extent[1] + 2 * offset)) + 1
    img = np.full((height, width), 255, np.uint8)

    px = {k: (v - lo) * scale + offset for k, v in traces.items()}
    f = 1 << SHIFT
    radius = thickness / 2.0
    for p in px.values():
        q = np.round(p * f).astype(np.int32)
        if np.ptp(p, axis=0).max() < 1.0:              # dot: zero-length stroke
            c = tuple(int(v) for v in np.round(p.mean(axis=0) * f))
            cv2.circle(img, c, max(1, int(round(radius * f))), 0, -1, cv2.LINE_AA, SHIFT)
        else:
            cv2.polylines(img, [q.reshape(-1, 1, 2)], False, 0, thickness, cv2.LINE_AA, SHIFT)

    symbols = []
    for s in parsed["symbols"]:
        p = np.concatenate([px[i] for i in s["trace_ids"]])
        x0, y0 = np.floor(p.min(axis=0) - radius).astype(int)
        x1, y1 = np.ceil(p.max(axis=0) + radius).astype(int) + 1
        box = [max(0, y0), min(height, y1), max(0, x0), min(width, x1)]
        symbols.append({"label": s["label"], "trace_ids": s["trace_ids"],
                        "box": [int(v) for v in box]})

    transform = {"scale": float(scale), "offset": [offset, offset],
                 "xmin": float(lo[0]), "ymin": float(lo[1])}
    return img, symbols, transform


def draw_symbol_boxes(img, symbols, color=(0, 0, 255)):
    """Grayscale render + symbol boxes -> BGR visualisation (for visual checks)."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for s in symbols:
        y0, y1, x0, x1 = s["box"]
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), color, 1)
    return vis


# ---- Dataset processing -----------------------------------------------------------
def read_split_list(split):
    """Split list -> [(inkml path, layout string)] (paths rel. to DATASET)."""
    rows = []
    with open(DATASET / f"crohme2019_{split}.txt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.strip():
                rel, _, layout = line.partition("\t")
                rows.append((DATASET / rel, layout.strip()))
    return rows


def process_split(split, limit=None, out=OUT, verbose=True):
    """Render every expression of a split and write images, <split>.jsonl and
    <split>_truth.tsv. Returns the list of records."""
    rows = read_split_list(split)[:limit]
    ids = [p.stem for p, _ in rows]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    if dupes:
        raise ValueError(f"{split}: duplicate ids {sorted(dupes)[:5]}")

    img_dir = out / split / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    records, failed = [], []
    for n, (path, layout) in enumerate(rows, 1):
        expr_id = path.stem
        try:
            parsed = parse_inkml(path)
            thickness = thickness_for(expr_id)
            img, symbols, tf = render_expression(parsed, thickness)
        except Exception as e:                       # keep going, report at the end
            failed.append((expr_id, repr(e)))
            continue
        rel_img = f"{split}/images/{expr_id}.png"
        cv2.imwrite(str(out / rel_img), img)
        records.append({
            "id": expr_id,
            "split": split,
            "inkml": str(path.relative_to(ROOT)),
            "image": rel_img,
            "height": img.shape[0],
            "width": img.shape[1],
            "latex": parsed["latex"],
            "truth_raw": parsed["truth_raw"],
            "category": parsed["category"],
            "layout": layout,
            "thickness": thickness,
            "scale": tf["scale"],
            "offset": tf["offset"],
            "ink_min": [tf["xmin"], tf["ymin"]],
            "symbols": symbols,
            "orphan_trace_ids": parsed["orphan_trace_ids"],
            "missing_refs": parsed["missing_refs"],
        })
        if verbose and n % 1000 == 0:
            print(f"  {split}: {n}/{len(rows)}", flush=True)

    with open(out / f"{split}.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / f"{split}_truth.tsv", "w", encoding="utf-8") as fh:
        fh.write("image\tlatex\n")
        for r in records:
            fh.write(f"{r['image']}\t{' '.join(r['latex'].split())}\n")  # no tabs/newlines

    if verbose:
        n_sym = sum(len(r["symbols"]) for r in records)
        print(f"{split}: {len(records)}/{len(rows)} rendered, {n_sym} symbols, "
              f"{len(failed)} failed -> {out / split}")
        for expr_id, err in failed:
            print(f"  FAILED {expr_id}: {err}")
    return records


def load_split(split, out=OUT):
    """Read <split>.jsonl back -> list of records (image paths rel. to `out`)."""
    with open(out / f"{split}.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render CROHME 2019 InkML to images + GT.")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--limit", type=int, default=None, help="first N files per split")
    args = ap.parse_args(argv)
    for split in args.splits:
        process_split(split, args.limit)


if __name__ == "__main__":
    sys.exit(main())
