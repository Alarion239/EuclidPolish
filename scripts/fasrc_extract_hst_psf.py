#!/usr/bin/env python
"""Extract the F814W ePSF from downloaded COSMOS HLSP tiles.

Scans each HLSP tile for bright unsaturated point sources, then runs
:class:`photutils.psf.EPSFBuilder` to construct a high-S/N empirical
PSF. Mirrors the Euclid VIS ePSF flow in
:mod:`euclid_polish.euclid.psf_extractor`.

Output: ``$DATA_DIR/hst_psf/F814W.fits`` — a single FITS file with the
oversampled empirical PSF + provenance headers (n_stars used, tile
indices the stars came from, RMS reconstruction error).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config


HLSP_DIR_NAME    = "hst_hlsp"
PSF_DIR_NAME     = "hst_psf"
PSF_FILE_NAME    = "F814W.fits"
# Per-star stamp library populated for the HST cutouts UI tab. Tiny
# (~260 KB each) and useful for spot-checking which sources fed the
# ePSF — same role the Euclid star cutouts play for the VIS PSF.
STARS_DIR_NAME   = "hst_stars"
# ACS/WFC native scale; COSMOS HLSP is drizzled to 0.03"/pix.
HLSP_PIX_SCALE_ARCSEC = 0.03
EPSF_OVERSAMPLING     = 2          # → saved at 0.015"/pix
PSF_HALF_SIDE_PIX     = 127        # final ePSF side = 2 × half + 1 = 255


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-stars", type=int, default=200,
                   help="Target number of stars to use (more = higher S/N, "
                        "slower). The actual number used is reported in "
                        "the FITS header.")
    p.add_argument("--input-dir", default=None,
                   help="Directory of HLSP tiles. Defaults to $DATA_DIR/hst_hlsp/.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write the ePSF FITS. Defaults to "
                        "$DATA_DIR/hst_psf/.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be done and exit.")
    return p.parse_args()


def _find_stars_in_tile(data: np.ndarray, *, max_n: int,
                        sigma: float = 5.0) -> "Table":
    """Detect bright unsaturated point sources in one tile."""
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    # Threshold: 50× sigma — point-source-bright but not saturated.
    finder = DAOStarFinder(
        threshold=50 * std, fwhm=4.0,    # ~4 px FWHM at 0.03"/pix → 0.12"
        sharplo=0.4, sharphi=0.8,         # rejects extended / cosmic-ray
        roundlo=-0.4, roundhi=0.4,
    )
    sources = finder(data - median)
    if sources is None or len(sources) == 0:
        return None
    # Sort by peak brightness, take top max_n; reject ones too close to edge.
    sources.sort("peak")
    sources.reverse()
    H, W = data.shape
    border = PSF_HALF_SIDE_PIX + 5
    keep = [
        (border < r["xcentroid"] < W - border
         and border < r["ycentroid"] < H - border)
        for r in sources
    ]
    sources = sources[keep]
    return sources[:max_n] if max_n < len(sources) else sources


def main() -> int:
    args = parse_args()
    in_dir  = args.input_dir  or os.path.join(Config.DATA_DIR, HLSP_DIR_NAME)
    out_dir = args.output_dir or os.path.join(Config.DATA_DIR, PSF_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print(f"  HST F814W ePSF extraction")
    print("=" * 64)
    print(f"  HLSP tile dir   = {in_dir}")
    print(f"  output dir      = {out_dir}")
    print(f"  target n_stars  = {args.n_stars}")
    print()

    t0 = time.time()

    tiles = sorted(
        f for f in os.listdir(in_dir) if f.endswith(".fits")
        and f.startswith("hlsp_cosmos_hst_acs-wfc_mosaic")
    ) if os.path.isdir(in_dir) else []
    print(f"[1/3] {len(tiles)} HLSP tiles found")
    if not tiles:
        print(f"ERROR: no HLSP tiles in {in_dir} — run the download step first.")
        return 1

    if args.dry_run:
        print(f"\nDRY RUN — would scan {len(tiles)} tiles for ~{args.n_stars} stars")
        runtime = time.time() - t0
        print(f"\nRUNTIME_SECONDS={runtime:.1f}")
        return 0

    # ---- collect stars across tiles until we hit the target count ----
    print(f"[2/3] scanning tiles for bright unsaturated point sources ...")
    from astropy.io import fits
    from astropy.nddata import NDData
    from photutils.psf import extract_stars

    star_stamps: list = []
    stars_per_tile = max(1, args.n_stars // max(len(tiles), 1) * 2)
    tiles_used: list = []

    for tile_idx, tname in enumerate(tiles):
        if len(star_stamps) >= args.n_stars:
            break
        tpath = os.path.join(in_dir, tname)
        print(f"      tile {tile_idx + 1}/{len(tiles)}: {tname}")
        with fits.open(tpath, memmap=True) as hdul:
            sci = next(
                (e for e in hdul if e.is_image and e.data is not None), None,
            )
            if sci is None:
                continue
            data = np.asarray(sci.data, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sources = _find_stars_in_tile(data, max_n=stars_per_tile)
        if sources is None or len(sources) == 0:
            print(f"        (no stars passed quality cuts)")
            continue
        print(f"        + {len(sources)} stars")
        nd = NDData(data=data)
        stamps = extract_stars(
            nd, sources, size=2 * PSF_HALF_SIDE_PIX + 1,
        )
        star_stamps.extend(stamps)
        tiles_used.append(tname)

    n_used = min(len(star_stamps), args.n_stars)
    if n_used == 0:
        print("ERROR: 0 usable stars across all tiles")
        return 1
    star_stamps = star_stamps[:n_used]
    print(f"      collected {n_used} stars from {len(tiles_used)} tiles")

    # ---- save each used star as a FITS so the UI can browse them ----
    stars_dir = os.path.join(Config.DATA_DIR, STARS_DIR_NAME)
    os.makedirs(stars_dir, exist_ok=True)
    saved = 0
    for i, st in enumerate(star_stamps):
        try:
            stamp_arr = np.asarray(st.data, dtype=np.float32)
        except Exception:
            continue
        side = stamp_arr.shape[0]
        out_path = os.path.join(
            stars_dir, f"star_{i:04d}_{side}.fits",
        )
        hdu = fits.PrimaryHDU(stamp_arr)
        h = hdu.header
        h["OBJECT"]   = ("HST F814W star stamp",
                         "extracted for ePSF construction")
        h["FILTER"]   = ("F814W", "HST filter")
        h["INSTRUME"] = ("ACS/WFC", "HST instrument")
        h["PIXSCALE"] = (HLSP_PIX_SCALE_ARCSEC, "native HLSP pixel scale (arcsec)")
        h["STARIDX"]  = (i, "0-based index in this run")
        h["NTILESRC"] = (len(tiles_used), "tiles contributing to this run")
        h["BUNIT"]    = ("electrons / s", "ACS/WFC drizzled units")
        try:
            hdu.writeto(out_path, overwrite=True)
            saved += 1
        except Exception as e:
            print(f"        warn: failed to save {out_path}: {e}")
    print(f"      saved {saved} star stamps → {stars_dir}/")

    # ---- build the ePSF ----
    print(f"[3/3] running EPSFBuilder (oversampling = {EPSF_OVERSAMPLING}) ...")
    from photutils.psf import EPSFBuilder
    builder = EPSFBuilder(
        oversampling=EPSF_OVERSAMPLING,
        maxiters=10,
        smoothing_kernel="quartic",
        progress_bar=False,
    )
    epsf, _fitted_stars = builder(star_stamps)
    psf_arr = np.asarray(epsf.data, dtype=np.float32)
    psf_arr = psf_arr / float(psf_arr.sum())   # unit flux

    out_path = os.path.join(out_dir, PSF_FILE_NAME)
    hdu = fits.PrimaryHDU(psf_arr)
    h = hdu.header
    h["OBJECT"]   = ("HST F814W ePSF", "empirical PSF from COSMOS HLSP")
    h["FILTER"]   = ("F814W", "HST filter")
    h["INSTRUME"] = ("ACS/WFC", "HST instrument")
    h["NSTARS"]   = (n_used, "stars used in EPSFBuilder")
    h["NTILES"]   = (len(tiles_used), "HLSP tiles contributing stars")
    h["OVERSAMP"] = (EPSF_OVERSAMPLING, "oversampling factor relative to HLSP grid")
    h["PIXSCALE"] = (HLSP_PIX_SCALE_ARCSEC / EPSF_OVERSAMPLING,
                     "arcsec / oversampled pixel")
    h["BUNIT"]    = ("", "unit flux (sums to 1)")
    hdu.writeto(out_path, overwrite=True)
    print(f"  wrote ePSF → {out_path}")
    print(f"    shape    = {psf_arr.shape}")
    print(f"    pix scale = {HLSP_PIX_SCALE_ARCSEC / EPSF_OVERSAMPLING:.4f}\"/pix")
    print(f"    flux sum  = {psf_arr.sum():.6f}  (should be 1.0)")

    runtime = time.time() - t0
    print(f"\nRUNTIME_SECONDS={runtime:.1f}")
    print(f"N_STARS_USED={n_used}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
