#!/usr/bin/env python
"""Step 0 of the star-anchor work: verify the catalog-magnitude → electrons
scale against real star cutouts.

For each star in ``data/euclid_stars/stars.csv`` we load its VIS cutout,
convert it from archive ADU/s to electrons over the stack (same MAGZERO
conversion the model input uses, ``euclid.photometry.adu_per_s_to_electrons``),
measure its flux in circular apertures, and compare to the electrons implied
by the catalog magnitude (``ab_mag_to_electrons``). A ratio ≈ 1 in the large
aperture means catalog-mag electrons ≈ total measured electrons — i.e. the
anchor delta-target (built from catalog mag) is on the same scale the model
sees.

The anchor flux is now the catalog ``flux_vis_psf`` (TPHOT PSF-fitting flux,
µJy) — already a *total* point-source flux — converted to electrons via the
physical µJy→AB→e⁻ path (``uJy_to_electrons``). So the large-aperture ratio
should land near 1 with no aperture correction; a gross constant offset at
all radii instead points to a zeropoint/units mistake. This script only
reports; it changes nothing.

Run where the cutouts live (FASRC):
    python scripts/verify_star_photometry.py --n 40 --size 512
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from astropy.io import fits

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.catalog.photometry import (
    ab_mag_to_electrons, adu_per_s_to_electrons, uJy_to_electrons,
)


def _catalog_electrons(row: dict, band) -> float:
    """Catalog-implied electrons: prefer the raw PSF flux (flux_psf_uJy,
    physical µJy→e⁻), fall back to the magnitude for older catalogs."""
    flux = row.get("flux_psf_uJy")
    try:
        f = float(flux)
        if f > 0:
            return uJy_to_electrons(f, band)
    except (TypeError, ValueError):
        pass
    return ab_mag_to_electrons(float(row["magnitude"]), band)


def _aperture_sum(img_e: np.ndarray, r: float) -> float:
    """Background-subtracted circular-aperture sum about the image centre.

    Background = median in the annulus ``[1.5r, 2.5r]``; the star is assumed
    at the geometric centre (the cutouts are centred on the catalog RA/Dec)."""
    ny, nx = img_e.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(yy - cy, xx - cx)
    ann = (rr >= 1.5 * r) & (rr <= 2.5 * r)
    bkg = float(np.nanmedian(img_e[ann])) if np.any(ann) else 0.0
    ap = rr <= r
    return float(np.nansum(img_e[ap] - bkg))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=40, help="Number of stars to check.")
    p.add_argument("--size", type=int, default=Config.DEFAULT_CUTOUT_SIZE)
    p.add_argument("--stars-csv", default=os.path.join(
        Config.DATA_DIR, "euclid_stars", "stars.csv"))
    p.add_argument("--radii", type=float, nargs="+", default=[3.0, 6.0, 12.0])
    args = p.parse_args()

    band = Config.BAND_VIS
    vis_dir = Config.cutout_dir_for_band("VIS")
    with open(args.stars_csv) as fh:
        rows = list(csv.DictReader(fh))[: args.n]

    print(f"VIS sim_zeropoint_e = {band.sim_zeropoint_e:.3f}  "
          f"(electrons-over-stack scale)\n"
          f"checking {len(rows)} stars from {args.stars_csv}\n")
    header = "  id     mag       cat_e   " + "  ".join(
        f"r={r:g}px" for r in args.radii)
    print(header)

    ratios = {r: [] for r in args.radii}
    for row in rows:
        sid = int(row["id"])
        mag = float(row["magnitude"])
        path = os.path.join(vis_dir, f"star_{sid:04d}_{args.size}.fits")
        if not os.path.isfile(path):
            continue
        with fits.open(path) as hdul:
            arr = np.asarray(hdul[0].data, dtype=np.float32)
            magzero = float(hdul[0].header.get("MAGZERO", band.sim_zeropoint_e))
        img_e = adu_per_s_to_electrons(arr, magzero, band)
        cat_e = _catalog_electrons(row, band)
        sums = [_aperture_sum(img_e, r) for r in args.radii]
        for r, s in zip(args.radii, sums):
            if cat_e > 0:
                ratios[r].append(s / cat_e)
        cells = "  ".join(f"{s / cat_e:7.3f}" for s in sums)
        print(f"  {sid:<5d} {mag:6.3f} {cat_e:10.3g}   {cells}")

    print("\nmedian measured/catalog flux ratio per aperture radius:")
    for r in args.radii:
        vals = ratios[r]
        if vals:
            print(f"  r={r:g}px : median={np.median(vals):.3f}  "
                  f"(n={len(vals)}, IQR "
                  f"{np.percentile(vals, 25):.3f}-{np.percentile(vals, 75):.3f})")
    print("\n→ ratio (measured / catalog) per radius:\n"
          "   • a gross constant offset at ALL radii ⇒ a zeropoint/units bug\n"
          "     (the catalog flux is µJy, AB ZP {:.2f}; verify it round-trips).\n"
          "   • with PSF flux, the moderate-aperture ratio ≈ 1 ⇒ no correction.\n"
          "     If you instead anchor on an aperture flux and the ratio rises\n"
          "     with radius, the correction factor is that (>1) ratio itself —\n"
          "     anchor_flux = catalog_flux × ratio — NOT 1/ratio."
          .format(Config.AB_ZP_UJY))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
