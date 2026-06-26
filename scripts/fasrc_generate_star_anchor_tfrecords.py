#!/usr/bin/env python
"""Build star-anchor TFRecords from the centered per-star cutouts.

Reuses the cutouts we already download for PSF extraction
(``data/euclid_stars/cutouts/<band>/star_NNNN_<size>.fits``) plus the catalog
photometry in ``stars.csv`` — preferring the raw PSF flux ``flux_psf_uJy``
(TPHOT, total point-source flux) and falling back to ``magnitude``. Each
cutout is fetched at the star's catalog RA/Dec, so the star sits at the
central pixel; all four bands are co-registered on the common 0.10″/pix grid.

For each usable star we write a ``(dirty_anchor, hr_anchor)`` pair —
structurally identical to the synthetic ``(dirty, hr)`` pair so the
training dataset reuses the supervised lane machinery:

  * ``dirty_anchor`` — the 4-band LR cube (electrons over the stack, via
    each band's MAGZERO), a ``stamp``-sized crop centered on the star.
  * ``hr_anchor``    — a 1-channel HR (0.05″/pix, 2× the LR) target that is
    zero everywhere except ONE pixel: the star's WCS-exact nearest HR pixel,
    set to the catalog VIS flux in electrons. The training loss masks to
    ``target > 0``, so only that pixel is supervised (operator-free — no PSF).

Output: ``$DATA_DIR/images/records_v2_star_anchor/{dirty,hr}_anchor_{train,validate}.tfrecord``.

Run where the cutouts live (FASRC):
    python scripts/fasrc_generate_star_anchor_tfrecords.py --size 512 --stamp 128
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.euclid.photometry import (
    ab_mag_to_electrons, adu_per_s_to_electrons, uJy_to_electrons,
)
from euclid_polish.image.tfio import open_multiband_writer
from euclid_polish.image import Image

SCALE = Config.DEFAULT_REBIN_FACTOR          # LR→HR factor (2)
STAR_ANCHOR_RECORDS_DIR = os.path.join(Config.DATA_DIR, "images/records_v2_star_anchor")


def star_cutout_path(star_id: int, band_name: str, size: int) -> str:
    return os.path.join(
        Config.cutout_dir_for_band(band_name), f"star_{star_id:04d}_{size}.fits")


def load_star_cube(star_id: int, size: int) -> Optional[Tuple[np.ndarray, fits.Header]]:
    """Load the 4 co-registered band cutouts, convert each ADU/s→e⁻, and
    stack to ``(H, W, 4)`` in canonical band order. Returns ``(cube, vis_header)``
    or ``None`` if any band is missing, unreadable/corrupt, or the shapes
    disagree (so one bad cutout skips the star, never crashing the job)."""
    planes = []
    vis_header = None
    for band_name in Config.LR_INPUT_BAND_NAMES:
        path = star_cutout_path(star_id, band_name, size)
        if not os.path.isfile(path):
            return None
        # A truncated / empty / otherwise corrupt cutout must skip the star,
        # not crash the whole job (same as every other FITS-reading pipeline).
        # ``np.asarray(..., dtype)`` copies while the file is still open so the
        # array stays valid after the (possibly memmapped) HDUList closes.
        try:
            with fits.open(path) as hdul:
                if hdul[0].data is None:
                    raise OSError("no image data in primary HDU")
                arr = np.asarray(hdul[0].data, dtype=np.float32)
                header = hdul[0].header.copy()
                magzero = float(header.get(
                    "MAGZERO", Config.get_band(band_name).sim_zeropoint_e))
        except Exception as e:
            print(f"  ⚠️  star {star_id:04d} {band_name}: unreadable cutout "
                  f"({type(e).__name__}: {e}) — skipping star")
            return None
        if arr.ndim != 2:
            return None
        planes.append(adu_per_s_to_electrons(arr, magzero, Config.get_band(band_name)))
        if band_name == Config.BAND_VIS.name:
            vis_header = header.copy()
    if len({p.shape for p in planes}) != 1:        # bands disagree on size
        return None
    return np.stack(planes, axis=-1).astype(np.float32), vis_header


def star_pixel(vis_header: fits.Header, ra: float, dec: float) -> Tuple[float, float]:
    """Star's (col_x, row_y) float pixel in the VIS cutout, from its WCS."""
    wcs = WCS(vis_header)
    x, y = wcs.all_world2pix(ra, dec, 0)         # 0-based, (x=col, y=row)
    return float(x), float(y)


def make_anchor_example(
    cube_e: np.ndarray, star_xy: Tuple[float, float], flux_e: float, stamp: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Crop a ``stamp``-sized LR cube centered on the star and render the HR
    delta target. Returns ``(lr_cube (stamp,stamp,4), hr_target (2*stamp,2*stamp,1))``
    or ``None`` if the star is too close to the cutout edge for a full stamp."""
    h, w, _ = cube_e.shape
    sx, sy = star_xy
    half = stamp // 2
    # Top-left of the stamp, centered on the star, clamped into the cutout.
    x0 = int(round(sx)) - half
    y0 = int(round(sy)) - half
    if x0 < 0 or y0 < 0 or x0 + stamp > w or y0 + stamp > h:
        x0 = min(max(0, x0), w - stamp)
        y0 = min(max(0, y0), h - stamp)
        if x0 < 0 or y0 < 0:                      # cutout smaller than the stamp
            return None
    lr_cube = np.ascontiguousarray(cube_e[y0:y0 + stamp, x0:x0 + stamp, :])

    # Star position inside the stamp, then on the 2× HR grid → nearest HR pixel.
    sx_s, sy_s = sx - x0, sy - y0
    hr_col = int(round(SCALE * sx_s))
    hr_row = int(round(SCALE * sy_s))
    hr_col = min(max(0, hr_col), SCALE * stamp - 1)
    hr_row = min(max(0, hr_row), SCALE * stamp - 1)
    hr_target = np.zeros((SCALE * stamp, SCALE * stamp, 1), dtype=np.float32)
    hr_target[hr_row, hr_col, 0] = np.float32(flux_e)
    return lr_cube, hr_target


def _row_float(row: dict, key: str) -> Optional[float]:
    v = row.get(key)
    if v is None or str(v).strip() == "":
        return None
    try:
        f = float(v)
        return f if f == f else None        # drop NaN
    except (TypeError, ValueError):
        return None


def anchor_flux_e(row: dict) -> Optional[float]:
    """Star-anchor delta value in electrons. Prefer the raw PSF flux
    (``flux_psf_uJy`` → electrons, physical scale); fall back to the catalog
    magnitude for older catalogs that predate the flux columns."""
    flux_uJy = _row_float(row, "flux_psf_uJy")
    if flux_uJy is not None and flux_uJy > 0:
        return uJy_to_electrons(flux_uJy, Config.BAND_VIS)
    mag = _row_float(row, "magnitude")
    if mag is not None:
        return ab_mag_to_electrons(mag, Config.BAND_VIS)
    return None


def _read_stars(stars_csv: str, limit: Optional[int]):
    with open(stars_csv) as fh:
        rows = list(csv.DictReader(fh))
    return rows[:limit] if limit else rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=Config.DEFAULT_CUTOUT_SIZE,
                   help="Cutout side (px) used at download time.")
    p.add_argument("--stamp", type=int, default=128,
                   help="LR stamp side (px); must be ≥ the training LR crop "
                        "(HR_CROP/2) so the random crop always keeps the star.")
    p.add_argument("--stars-csv", default=os.path.join(
        Config.DATA_DIR, "euclid_stars", "stars.csv"))
    p.add_argument("--output-dir", default=STAR_ANCHOR_RECORDS_DIR)
    p.add_argument("--valid-every", type=int, default=10,
                   help="Every Nth usable star goes to the validate split.")
    p.add_argument("--snr-min", type=float, default=None,
                   help="Skip stars whose PSF flux_psf_uJy / fluxerr_psf_uJy "
                        "is below this (well-measured flux only). Off by default.")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    min_stamp = Config.DEFAULT_HR_CROP_SIZE // SCALE
    if args.stamp < min_stamp:
        raise SystemExit(f"--stamp ({args.stamp}) must be ≥ HR_CROP/2 = {min_stamp}")

    rows = _read_stars(args.stars_csv, args.limit)
    band_names = Config.LR_INPUT_BAND_NAMES
    counts = {"train": 0, "validate": 0}
    skipped = 0
    with contextlib.ExitStack() as stack:
        writers = {
            (subset, kind): stack.enter_context(
                open_multiband_writer(f"{kind}_anchor_{subset}", args.output_dir))
            for subset in ("train", "validate") for kind in ("dirty", "hr")
        }
        usable = 0
        for row in rows:
            sid = int(row["id"])
            # Optional SNR cut on the raw PSF photometry.
            if args.snr_min is not None and args.snr_min > 0:
                flux = _row_float(row, "flux_psf_uJy")
                ferr = _row_float(row, "fluxerr_psf_uJy")
                if not (flux and ferr and ferr > 0 and flux / ferr >= args.snr_min):
                    skipped += 1
                    continue
            flux_e = anchor_flux_e(row)
            if flux_e is None or flux_e <= 0:
                skipped += 1
                continue
            loaded = load_star_cube(sid, args.size)
            if loaded is None:
                skipped += 1
                continue
            cube_e, vis_header = loaded
            try:
                sx, sy = star_pixel(vis_header, float(row["ra"]), float(row["dec"]))
            except Exception:                      # missing/invalid WCS
                skipped += 1
                continue
            ex = make_anchor_example(cube_e, (sx, sy), flux_e, args.stamp)
            if ex is None:
                skipped += 1
                continue
            lr_cube, hr_target = ex
            subset = "validate" if (usable % args.valid_every == 0) else "train"
            idx = counts[subset]
            writers[(subset, "dirty")].write(Image(
                data=lr_cube, pixel_scale_arcsec=Config.BAND_VIS.pixel_scale_lr_arcsec,
                band_names=band_names, is_clean=False, index=idx, subset=subset), index=idx)
            writers[(subset, "hr")].write(Image(
                data=hr_target, pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                band_names=(Config.HR_TARGET_BAND_NAME,), is_clean=True,
                index=idx, subset=subset), index=idx)
            counts[subset] += 1
            usable += 1

    print(f"star-anchor records → {args.output_dir}")
    print(f"  train={counts['train']}  validate={counts['validate']}  "
          f"skipped={skipped}  (stamp {args.stamp} LR / {SCALE*args.stamp} HR)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
