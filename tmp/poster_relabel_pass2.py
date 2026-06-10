#!/usr/bin/env python
"""Pass 2: erase the faint ghost remnants of the OLD small labels.

The new big labels live at the bottom-LEFT (x < 0.27 cell); the ghosts sit in
the bottom-center strip. Erase only x in [0.30, 0.85] so the new labels are
untouched. Low threshold catches the anti-aliased gray leftovers.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, median_filter


def cleanup(path, *, mode, thresh):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    H, W = arr.shape[:2]
    c = H // 2
    for r in (0, 1):
        for k in (0, 1):
            ox, oy = k * (W // 2), r * (H // 2)
            x0, x1 = ox + int(0.30 * c), ox + int(0.85 * c)
            y0, y1 = oy + int(0.84 * c), oy + c
            win = arr[y0:y1, x0:x1]
            gray = win.mean(axis=2)
            mask = gray > thresh
            interior = median_filter((gray > thresh).astype(np.uint8),
                                     size=7) == 1
            mask = binary_dilation(mask & ~interior, iterations=4)
            if not mask.any():
                continue
            if mode == "patch":
                ys, xs = np.where(mask)
                by0, by1 = max(0, ys.min() - 4), ys.max() + 5
                bx0, bx1 = max(0, xs.min() - 4), xs.max() + 5
                h = by1 - by0
                sy1 = y0 + by0
                arr[sy1:sy1 + h, x0 + bx0:x0 + bx1] = \
                    arr[sy1 - h - 6:sy1 - 6, x0 + bx0:x0 + bx1]
            else:
                med = median_filter(win, size=(11, 11, 1))
                win[mask] = med[mask]
    Image.fromarray(arr).save(path)
    print("cleaned", path)


cleanup("fig/poster/ingredients_grid.png", mode="patch", thresh=40)
cleanup("fig/poster/psf_grid.png", mode="patch", thresh=60)
cleanup("fig/poster/dirty_grid.png", mode="median", thresh=200)
