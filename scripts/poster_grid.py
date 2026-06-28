#!/usr/bin/env python3
"""Composite several images into ONE tiled grid PNG for the poster.

Instead of stacking panels by hand in LaTeX, render the 2x2 (or RxC) grid once,
here, so the poster just ``\\includegraphics`` a single file. Cells may be FITS
(rendered with the house asinh-grayscale stretch from ``poster_render``) or
already-rendered images (PNG/JPG, placed as-is). Optional per-cell labels are
burned into the image.

Two input modes
---------------
1. Separate files (ingredients, per-band PSFs, per-band dirty-from-files):

   python3 scripts/poster_grid.py lens.fits sersic.fits tng.fits star.fits \\
           -o fig/poster/ingredients_grid.png --labels "lens,Sersic,TNG,star"

   python3 scripts/poster_grid.py psf_VIS.fits psf_Y.fits psf_J.fits psf_H.fits \\
           -o fig/poster/psf_grid.png --labels VIS,Y,J,H --crop 110 --scale-pct 70

2. One stacked multi-band FITS split into cells (the 4-band dirty cube — e.g.
   an inference ``original_stack.fits`` of shape (4,H,W)):

   python3 scripts/poster_grid.py --stack original_stack.fits \\
           -o fig/poster/dirty_grid.png        # bands + labels read from header

Cells fill the grid row-major: index 0 = top-left, then left->right, top->bottom.
For a 2x2 band grid that is VIS (TL), Y (TR), J (BL), H (BR).
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from astropy.io import fits

# Reuse the house stretch / FITS helpers (poster_render.py sits next to this).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poster_render import block_mean, center_crop, grayscale_norm  # noqa: E402


# ---------------------------------------------------------------------------
def _square(arr: np.ndarray) -> np.ndarray:
    """Centre-crop a (H,W) or (H,W,C) array to a square."""
    h, w = arr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return arr[y0:y0 + s, x0:x0 + s]


def _load_fits_2d(path: str, hdu: int | None) -> np.ndarray:
    with fits.open(path) as hl:
        if hdu is not None:
            data = hl[hdu].data
        else:
            data = next((h.data for h in hl if getattr(h, "data", None) is not None
                         and np.ndim(h.data) >= 2), None)
    if data is None:
        raise ValueError(f"no 2-D image data in {path}")
    data = np.asarray(data, dtype=np.float32)
    while data.ndim > 2:
        data = data[0]
    return data


def _split_stack(path: str, hdu: int | None) -> tuple[list[np.ndarray], list[str]]:
    """Split a multi-band FITS into a list of 2-D band arrays + band names.

    Handles a 3-D cube ((N,H,W) or (H,W,N)) or several 2-D image HDUs."""
    with fits.open(path) as hl:
        names: list[str] = []
        # 3-D cube in one HDU
        idx = hdu if hdu is not None else 0
        data = hl[idx].data
        hdr = hl[idx].header
        if data is not None and np.ndim(data) == 3:
            cube = np.asarray(data, dtype=np.float32)
            # band axis = the short one (e.g. 4)
            band_axis = int(np.argmin(cube.shape))
            cube = np.moveaxis(cube, band_axis, 0)         # (N,H,W)
            bands = [cube[i] for i in range(cube.shape[0])]
            raw = hdr.get("BANDS") or hdr.get("BAND") or ""
            names = [t.strip().replace("_E", "") for t in str(raw).split(",") if t.strip()]
            return bands, names
        # else: collect 2-D image HDUs
        bands = [np.asarray(h.data, dtype=np.float32)
                 for h in hl if getattr(h, "data", None) is not None
                 and np.ndim(h.data) == 2]
    return bands, names


def _prep_fits_cell(arr: np.ndarray, args, scale, lohi) -> np.ndarray:
    if args.crop:
        arr = center_crop(arr, args.crop)
    arr = block_mean(arr, args.downsample)
    normed, _, _ = grayscale_norm(
        arr, scale_pct=args.scale_pct, clip=(args.clip[0], args.clip[1]),
        scale=scale, lohi=lohi)
    return _square(normed)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="*", help="cell image files (FITS or PNG/JPG)")
    p.add_argument("--stack", default=None,
                   help="one multi-band FITS to split into cells")
    p.add_argument("-o", "--out", required=True, help="output PNG")
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--cols", type=int, default=2)
    p.add_argument("--labels", default="", help="comma-separated per-cell labels")
    p.add_argument("--cell", type=int, default=600, help="per-cell side (px)")
    p.add_argument("--gutter", type=float, default=0.03,
                   help="gap between cells, as a fraction of a cell")
    p.add_argument("--bg", default="black", help="background / gutter color")
    p.add_argument("--hdu", type=int, default=None)
    p.add_argument("--crop", type=int, default=0, help="center-crop FITS NxN first")
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--scale-pct", type=float, default=90.0)
    p.add_argument("--clip", type=float, nargs=2, default=(0.5, 99.5),
                   metavar=("LO", "HI"))
    p.add_argument("--shared-stretch", action="store_true",
                   help="one common asinh stretch across all FITS cells")
    p.add_argument("--cmap", default="gray")
    p.add_argument("--invert", action="store_true")
    p.add_argument("--label-size", type=float, default=15.0)
    p.add_argument("--label-color", default="white")
    args = p.parse_args()

    # ---- assemble the list of cells: (array, origin, is_rgb) + labels --------
    cells: list[tuple[np.ndarray, str, bool]] = []
    labels = [s.strip() for s in args.labels.split(",")] if args.labels else []

    if args.stack:
        bands, names = _split_stack(args.stack, args.hdu)
        if not labels:
            labels = names
        # shared stretch across bands? compute from concatenation of cropped arrays
        scale = lohi = None
        prepped = [center_crop(b, args.crop) if args.crop else b for b in bands]
        prepped = [block_mean(b, args.downsample) for b in prepped]
        if args.shared_stretch:
            allpix = np.concatenate([b.ravel() for b in prepped])
            _, scale, lohi = grayscale_norm(
                allpix, scale_pct=args.scale_pct, clip=(args.clip[0], args.clip[1]))
        for b in prepped:
            normed, _, _ = grayscale_norm(
                b, scale_pct=args.scale_pct, clip=(args.clip[0], args.clip[1]),
                scale=scale, lohi=lohi)
            cells.append((_square(normed), "lower", False))
    else:
        if not args.inputs:
            p.error("provide cell files, or use --stack")
        # optional shared stretch across FITS inputs
        scale = lohi = None
        if args.shared_stretch:
            farrs = []
            for f in args.inputs:
                if f.lower().endswith((".fits", ".fit")):
                    a = _load_fits_2d(f, args.hdu)
                    if args.crop:
                        a = center_crop(a, args.crop)
                    farrs.append(block_mean(a, args.downsample).ravel())
            if farrs:
                _, scale, lohi = grayscale_norm(
                    np.concatenate(farrs), scale_pct=args.scale_pct,
                    clip=(args.clip[0], args.clip[1]))
        for f in args.inputs:
            if f.lower() in ("blank", "-", "none"):  # empty placeholder cell
                cells.append((np.zeros((8, 8), dtype=np.float32), "lower", False))
            elif f.lower().endswith((".fits", ".fit")):
                arr = _prep_fits_cell(_load_fits_2d(f, args.hdu), args, scale, lohi)
                cells.append((arr, "lower", False))
            else:                                   # already-rendered image
                img = mpimg.imread(f)
                if img.dtype != np.uint8 and img.max() > 1.0:
                    img = img / 255.0
                cells.append((_square(img), "upper", img.ndim == 3))

    n = args.rows * args.cols
    if len(cells) < n:
        cells += [(np.zeros((4, 4)), "lower", False)] * (n - len(cells))
    cells = cells[:n]
    labels = (labels + [""] * n)[:n]

    # ---- draw -----------------------------------------------------------------
    dpi = 100
    cell_in = args.cell / dpi
    fig, axes = plt.subplots(
        args.rows, args.cols, dpi=dpi,
        figsize=(args.cols * cell_in, args.rows * cell_in),
        squeeze=False)
    fig.patch.set_facecolor(args.bg)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=args.gutter, hspace=args.gutter)
    for k, ax in enumerate(axes.ravel()):
        arr, origin, is_rgb = cells[k]
        if is_rgb:
            ax.imshow(arr, origin=origin, interpolation="nearest")
        else:
            a = 1.0 - arr if args.invert else arr
            ax.imshow(a, origin=origin, cmap=args.cmap, vmin=0.0, vmax=1.0,
                      interpolation="nearest")
        ax.set_axis_off()
        if labels[k]:
            ax.text(0.5, 0.025, labels[k], transform=ax.transAxes,
                    ha="center", va="bottom", color=args.label_color,
                    fontsize=args.label_size, fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.55, pad=2.5,
                              edgecolor="none"))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=dpi, facecolor=args.bg)
    plt.close(fig)
    print(f"wrote {args.out}  ({args.rows}x{args.cols}, {n} cells)")


if __name__ == "__main__":
    main()
