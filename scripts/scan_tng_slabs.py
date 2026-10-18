#!/usr/bin/env python
"""Scan the TNG50 SKIRT atlas for "slab" renders.

Some SKIRT frames carry flat, straight-edged sheets of faint light around
the galaxy (e.g. subhalo 504560): rendering structure, not galaxy light.
Particle halos are dotty (neighbouring pixels disagree); slabs are flat.
For every donor orientation (native VIS frame), inside the circle the
renderer keeps (radius holding 99.9 % of the VIS light) and among its faint
pixels (3-px smoothed brightness below 1e-7 of the frame's flux), report
the share that is flat: non-zero and within 10 % of the 9x9 median.

Writes a CSV (one row per orientation, highest ``flat_area`` first) and a
contact sheet of the top frames for a by-eye review.

Usage: scan_tng_slabs.py --out-dir DIR [--data-dir DIR] [--workers N] [--sheet N]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, median_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from euclid_polish.config import Config  # noqa: E402

KEEP_FRACTION = 0.999          # the renderer's crop: _centered_rotation_crop_slices(enclosed_fraction=0.999)
FAINT_FRACTION = 1e-7          # of the frame's flux per pixel (3-px smoothing)
FLAT_TOLERANCE = 0.10          # within 10 % of the 9x9 median
MEDIAN_SIZE = 9


def keep_geometry(image: np.ndarray) -> tuple[int, int, float, np.ndarray]:
    """Centre (brightest 5-px smoothed pixel), keep radius and radius map."""
    smooth = gaussian_filter(image, 5.0)
    cy, cx = np.unravel_index(int(np.argmax(smooth)), image.shape)
    yy, xx = np.indices(image.shape)
    radius = np.hypot(yy - cy, xx - cx)
    order = np.argsort(radius.ravel())
    cumulative = np.cumsum(image.ravel()[order]) / image.sum()
    index = min(int(np.searchsorted(cumulative, KEEP_FRACTION)), cumulative.size - 1)
    return int(cy), int(cx), float(radius.ravel()[order][index]), radius


def flat_masks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(faint pixels inside the keep circle, the flat ones among them, keep radius)."""
    image = np.nan_to_num(np.asarray(image, dtype=np.float64))
    flux = image.sum()
    if not flux > 0:
        empty = np.zeros(image.shape, bool)
        return empty, empty, 0.0
    _, _, keep_radius, radius = keep_geometry(image)
    faint = (radius < keep_radius) & (gaussian_filter(image, 3.0) < FAINT_FRACTION * flux)
    median = median_filter(image, size=MEDIAN_SIZE)
    flat = faint & (image > 0) & (np.abs(image - median) < FLAT_TOLERANCE * np.maximum(median, 1e-300))
    return faint, flat, keep_radius


def flat_area(image: np.ndarray) -> dict[str, float]:
    """Share of the kept faint pixels (``flat_area``) and of their light
    (``flat_light``) that is flat, plus the keep radius in native pixels."""
    faint, flat, keep_radius = flat_masks(image)
    image = np.nan_to_num(np.asarray(image, dtype=np.float64))
    return {"flat_area": float(flat.sum() / max(int(faint.sum()), 1)),
            "flat_light": float(image[flat].sum() / max(float(image[faint].sum()), 1e-300)),
            "keep_radius_px": keep_radius}


def _scan_one(path: str) -> dict[str, object]:
    name = Path(path).name                                   # TNG<id>_O<k>_Euclid_VIS.fits
    subhalo, orientation = name.split("_")[0][3:], int(name.split("_")[1][1:])
    try:
        row = flat_area(fits.getdata(path))
    except Exception as exc:                                  # noqa: BLE001 — one broken frame must not stop the scan
        return {"subhalo_id": subhalo, "orientation": orientation, "path": path, "error": repr(exc)}
    return {"subhalo_id": subhalo, "orientation": orientation, "path": path, **row, "error": ""}


def contact_sheet(rows: list[dict], out_path: str, n: int) -> None:
    rows = [r for r in rows if not r["error"]][:n]
    if not rows:
        return
    cols = 6
    nrows = (len(rows) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(3.0 * cols, 3.1 * nrows), dpi=90, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, rows, strict=False):
        image = np.nan_to_num(fits.getdata(row["path"]).astype(np.float64))
        cy, cx, keep_radius, _ = keep_geometry(image)
        half = int(keep_radius * 1.1)
        y0, x0 = max(cy - half, 0), max(cx - half, 0)
        crop = image[y0:cy + half, x0:cx + half]
        stretched = np.arcsinh(crop / (1e-9 * image.sum()))
        ax.imshow(stretched, cmap="gray", origin="lower", interpolation="nearest")
        ax.set_title(f"{row['subhalo_id']} O{row['orientation']} flat {row['flat_area']:.3f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=Config.TNG_SKIRT_DIR)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--sheet", type=int, default=60,
                        help="frames on the contact sheet (highest flat_area)")
    args = parser.parse_args(argv)
    paths = sorted(glob.glob(os.path.join(args.data_dir, "*", "TNG*_O*_Euclid_VIS.fits")))
    if not paths:
        print(f"no VIS frames under {args.data_dir}", file=sys.stderr)
        return 1
    os.makedirs(args.out_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(_scan_one, paths, chunksize=8))
    rows.sort(key=lambda r: -float(r.get("flat_area", -1.0) or -1.0))
    fields = ["subhalo_id", "orientation", "flat_area", "flat_light", "keep_radius_px", "path", "error"]
    with open(os.path.join(args.out_dir, "tng_slab_scan.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    contact_sheet(rows, os.path.join(args.out_dir, "tng_slab_scan_top.png"), args.sheet)
    errors = sum(1 for r in rows if r["error"])
    print(f"scanned {len(rows)} frames ({errors} unreadable) -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
