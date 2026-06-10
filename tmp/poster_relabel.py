#!/usr/bin/env python
"""One-off: enlarge the burned-in cell labels of the poster grid PNGs.

For each 2x2 grid (ingredients / psf / dirty): find the small white label in
the bottom-center window of every cell, erase it (patch-copy from the clean
region above for smooth backgrounds, masked median-inpaint for the noisy
dirty field), then redraw the label bigger in the bottom-left corner.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, median_filter

FONT = ImageFont.truetype(
    "/Users/alarion239/miniforge3/envs/EuclidPolishEnv/lib/python3.12/"
    "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf", 44)


def _text_mask(win: np.ndarray, thresh: float) -> np.ndarray:
    gray = win.mean(axis=2) if win.ndim == 3 else win
    mask = gray > thresh
    # Exclude interiors of big bright blobs (real objects): text strokes are
    # thin, so a pixel whose 7x7 neighborhood is all-bright is not text.
    interior = median_filter((gray > thresh).astype(np.uint8), size=7) == 1
    return binary_dilation(mask & ~interior, iterations=4)


def erase_label(arr, x0, x1, y0, y1, *, mode, thresh=90):
    win = arr[y0:y1, x0:x1]
    mask = _text_mask(win, thresh)
    if not mask.any():
        return
    if mode == "patch":
        # Replace the text bounding box with the region directly above it.
        ys, xs = np.where(mask)
        by0, by1 = ys.min() - 4, ys.max() + 5
        bx0, bx1 = xs.min() - 4, xs.max() + 5
        h = by1 - by0
        sy1 = y0 + by0            # absolute top of the bbox
        arr[sy1:sy1 + h, x0 + bx0:x0 + bx1] = \
            arr[sy1 - h - 6:sy1 - 6, x0 + bx0:x0 + bx1]
    else:                          # masked median fill (noisy background)
        med = median_filter(win, size=(11, 11, 1) if win.ndim == 3 else (11, 11))
        win[mask] = med[mask]


def relabel(path, labels, *, mode, margin=14):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    H, W = arr.shape[:2]
    c = H // 2
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]           # (row, col)
    for (r, k), lab in zip(cells, labels):
        ox, oy = k * (W // 2), r * (H // 2)
        # old label window: bottom 14% strip, central 70% of the cell
        erase_label(arr, ox + int(0.15 * c), ox + int(0.85 * c),
                    oy + int(0.84 * c), oy + c, mode=mode)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    for (r, k), lab in zip(cells, labels):
        ox, oy = k * (W // 2), r * (H // 2)
        d.text((ox + margin, oy + c - margin), lab, font=FONT,
               fill=(255, 255, 255), anchor="lb",
               stroke_width=2, stroke_fill=(0, 0, 0))
    img.save(path)
    print("wrote", path)


relabel("fig/poster/ingredients_grid.png", ["Lens", "Sérsic", "TNG", "Star"],
        mode="patch")
relabel("fig/poster/psf_grid.png", ["VIS", "Y", "J", "H"], mode="patch")
relabel("fig/poster/dirty_grid.png", ["VIS", "Y", "J", "H"], mode="median")
