"""Step 3 — Line segmentation by horizontal projection profile.

Input: the deskewed binary from Step 2 (ink = 255, paper = 0) and the stroke
width w measured once in Step 1 (`results["stroke_width"]`, px; never
re-estimated here). Row profile p(y) = number of ink pixels in row y; a row is
an ink row if p(y) >= INK_MIN. A top-to-bottom scan with two states (IDLE /
IN-LINE) bridges empty runs of up to G_max = 3w rows (fraction bar <->
numerator / denominator) and ends a line once more than G_max empty rows
follow its last ink row. Lines shorter than H_min = 3w rows are dropped.
Spec: `Claude Chat Memory/Line_Segmentation.md`.

All intervals are half-open: a line (start, end) is the rows B[start:end];
a box (y0, y1, x0, x1) is B[y0:y1, x0:x1] in page coordinates.
"""

import numpy as np

# ---- Parameters (multiples of the stroke width w, except INK_MIN) ---------------
INK_MIN = 2      # px: a row needs >= INK_MIN ink pixels to count as an ink row
GAP_K = 3        # G_max = round(GAP_K * w): longest empty run still inside a line
MIN_H_K = 3      # H_min = round(MIN_H_K * w): shortest line kept


# ---- Profile and scan ------------------------------------------------------------
def row_profile(binary):
    """p(y) = number of ink pixels (> 0) in row y, y = 0..H-1 (int64)."""
    return np.count_nonzero(binary, axis=1).astype(np.int64)


def find_lines(profile, ink_min, gap_max, min_h):
    """Two-state row scan over the profile -> list of half-open (start, end).

    IDLE: the first ink row (p >= ink_min) starts a line (start = y).
    IN-LINE: an ink row updates last = y; at an empty row the gap is y - last
    (number of empty rows so far); once gap > gap_max the line ends at
    end = last + 1 (the last ink row, not the end of the gap). If the image
    ends while IN-LINE the line is closed the same way. Lines with
    end - start < min_h are dropped.
    """
    lines = []
    in_line = False
    start = last = 0
    for y, p in enumerate(profile):
        if p >= ink_min:
            if not in_line:
                in_line, start = True, y
            last = y
        elif in_line and y - last > gap_max:
            lines.append((start, last + 1))
            in_line = False
    if in_line:
        lines.append((start, last + 1))
    return [(s, e) for s, e in lines if e - s >= min_h]


# ---- Cropping --------------------------------------------------------------------
def ink_columns(band):
    """Half-open column range (x0, x1) of the ink in `band`; (0, W) if it has no ink."""
    cols = np.flatnonzero(np.count_nonzero(band, axis=0))
    if cols.size == 0:
        return 0, band.shape[1]
    return int(cols[0]), int(cols[-1]) + 1


def crop_line(binary, start, end, trim=True):
    """Rows B[start:end], trimmed to their ink columns if trim.

    Returns (crop, box), box = (y0, y1, x0, x1) half-open in page coordinates.
    """
    band = binary[start:end]
    x0, x1 = ink_columns(band) if trim else (0, binary.shape[1])
    return band[:, x0:x1], (int(start), int(end), x0, x1)


# ---- Entry ----------------------------------------------------------------------
def segment_lines(binary, stroke_w, ink_min=INK_MIN, gap_k=GAP_K, min_h_k=MIN_H_K):
    """Split a (deskewed) binary page into text lines.

    stroke_w is required: the Step-1 stroke width (px at this binary's scale).
    Returns (lines, info): lines = [(start, end), ...] top to bottom (half-open
    rows); info = stroke_w, gap_max, min_h, ink_min, profile.
    """
    gap_max = int(round(gap_k * stroke_w))
    min_h = int(round(min_h_k * stroke_w))
    profile = row_profile(binary)
    lines = find_lines(profile, ink_min, gap_max, min_h)
    info = {"stroke_w": stroke_w, "gap_max": gap_max, "min_h": min_h,
            "ink_min": ink_min, "profile": profile}
    return lines, info
