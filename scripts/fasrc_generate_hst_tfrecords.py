#!/usr/bin/env python
"""Generate clean (HST) + dirty (Euclid-forward) TFRecord pairs.

For each selected COSMOS2025 galaxy:

  1. Cut an HR-grid-sized HST cutout from the local HLSP tile that
     contains it (``Cutout2D`` + WCS).
  2. Background-subtract, broadcast to 4 bands using the COSMOS catalog
     per-band fluxes (HR_target has HST F814W morphology in every
     channel, but each channel gets its own electron normalisation).
  3. Apply the pre-computed differential kernel A to get the Euclid-PSF
     view, sum-rebin to LR scale (0.10″/pix), add per-band Euclid noise.
  4. Write the (HR, LR) pair into the standard ``MultiBandSkyImage``
     TFRecord layout — same schema the synthetic generator uses.

Output records land under ``$DATA_DIR/images/records_v2_hst/`` so the
trainer can mix them with the existing synthetic records via the
``hst_fraction`` knob.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config


HLSP_DIR     = os.path.join(Config.DATA_DIR, "hst_hlsp")
KERNEL_PATH  = os.path.join(Config.DATA_DIR, "hst_psf", "diff_kernel_VIS.fits")
OUT_DIR      = os.path.join(Config.DATA_DIR, "images", "records_v2_hst")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=6400,
                   help="HST-derived training pairs to write.")
    p.add_argument("--n-valid", type=int, default=200,
                   help="HST-derived validation pairs to write.")
    p.add_argument("--image-size", type=int, default=510,
                   help="HR cutout side in HR pixels (0.05\"/pix).")
    p.add_argument("--hlsp-dir",   default=HLSP_DIR)
    p.add_argument("--kernel",     default=KERNEL_PATH)
    p.add_argument("--output-dir", default=OUT_DIR)
    p.add_argument("--dry-run", action="store_true",
                   help="Plan only — no FITS reads, no TFRecord writes.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Tile manifest
# ---------------------------------------------------------------------------

class HLSPTileIndex:
    """Lightweight RA/Dec footprint cache so we can map galaxy → tile.

    Avoids opening every tile FITS just to figure out which one contains
    a given (ra, dec). Reads only each tile's primary header once.
    """

    def __init__(self, hlsp_dir: str):
        from astropy.io import fits
        from astropy.wcs import WCS
        self.entries: List[Dict] = []
        for fname in sorted(os.listdir(hlsp_dir)):
            if not (fname.endswith(".fits")
                    and fname.startswith("hlsp_cosmos_hst_acs-wfc_mosaic")):
                continue
            path = os.path.join(hlsp_dir, fname)
            with fits.open(path, memmap=True) as hdul:
                sci = next(
                    (e for e in hdul
                     if e.is_image and e.data is not None), None,
                )
                if sci is None:
                    continue
                wcs = WCS(sci.header)
                H, W = sci.data.shape
                # Footprint corners: convert pixel corners → RA/Dec.
                corners_pix = np.array([[0, 0], [W, 0], [0, H], [W, H]])
                radec = wcs.all_pix2world(corners_pix, 0)
                ra_min  = float(radec[:, 0].min())
                ra_max  = float(radec[:, 0].max())
                dec_min = float(radec[:, 1].min())
                dec_max = float(radec[:, 1].max())
                self.entries.append({
                    "path":    path,
                    "ra_min":  ra_min, "ra_max": ra_max,
                    "dec_min": dec_min, "dec_max": dec_max,
                    "shape":   (H, W),
                })
        if not self.entries:
            raise FileNotFoundError(f"no HLSP tiles in {hlsp_dir}")

    def find_tile(self, ra: float, dec: float) -> Optional[str]:
        """Return the FITS path of the tile containing (ra, dec), or None."""
        for e in self.entries:
            if (e["ra_min"] <= ra <= e["ra_max"]
                and e["dec_min"] <= dec <= e["dec_max"]):
                return e["path"]
        return None


# ---------------------------------------------------------------------------
# Pair synthesis
# ---------------------------------------------------------------------------

def _hr_pixel_side_arcsec() -> float:
    return Config.DEFAULT_PIXEL_SCALE


def _resample_hlsp_to_hr(hlsp_cutout: np.ndarray,
                         hlsp_scale: float = 0.03,
                         hr_scale: float = None) -> np.ndarray:
    """Resample 0.03″ HLSP cutout onto the 0.05″ HR grid."""
    from scipy.ndimage import zoom
    hr_scale = hr_scale or _hr_pixel_side_arcsec()
    factor = hlsp_scale / hr_scale   # 0.6 → output is smaller
    out = zoom(hlsp_cutout.astype(np.float32),
               zoom=factor, order=3, mode="constant")
    return out


def _make_pair(
    hr_cutout_4ch: np.ndarray,            # (H, W, 4) HR clean
    *,
    diff_kernel: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Forward-model the HR cube to a 4-band LR cube at 0.10″/pix.

    For each band: A ⊛ HR → sum-rebin ×2 → Poisson + read + sky + dark.
    """
    from scipy import signal as scipy_signal
    from euclid_polish.sky.multiband_forward import _apply_band_noise

    H, W, C = hr_cutout_4ch.shape
    assert H % 2 == 0 and W % 2 == 0
    lr_cube = np.zeros((H // 2, W // 2, C), dtype=np.float32)
    for k, band_name in enumerate(Config.LR_INPUT_BAND_NAMES):
        band = Config.get_band(band_name)
        # Convolve with the differential kernel (A) instead of the
        # Euclid PSF, so the chain is photometrically Euclid-equivalent
        # while accounting for the HST-baked-in PSF.
        convolved = scipy_signal.fftconvolve(
            hr_cutout_4ch[..., k], diff_kernel, mode="same",
        )
        # Sum-rebin ×2 (photometric: conserves total electrons per LR pixel).
        rebinned = convolved.reshape(H // 2, 2, W // 2, 2).sum(axis=(1, 3))
        # Apply per-band Euclid noise (Poisson + read, optional artifacts).
        lr_cube[..., k] = _apply_band_noise(rebinned, band, rng)
    return lr_cube


def _broadcast_hst_to_4bands(
    hst_2d: np.ndarray,
    flux_per_band_e: Tuple[float, float, float, float],
) -> np.ndarray:
    """Scale a unit-flux 2D template into a (H, W, 4) per-band cube.

    Same photometric contract as the existing
    :func:`add_sersic_to_bands` path: morphology is band-independent,
    each band gets its own total electron count from the catalog.
    """
    s = float(hst_2d.sum())
    if not np.isfinite(s) or s <= 0:
        return None
    unit = (hst_2d / s).astype(np.float32)
    flux = np.asarray(flux_per_band_e, dtype=np.float32)
    return unit[:, :, None] * flux[None, None, :]


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 64)
    print(f"  HST → Euclid TFRecord pair generation")
    print("=" * 64)
    print(f"  HLSP dir     = {args.hlsp_dir}")
    print(f"  kernel       = {args.kernel}")
    print(f"  output       = {args.output_dir}")
    print(f"  n_train      = {args.n_train}")
    print(f"  n_valid      = {args.n_valid}")
    print(f"  image_size   = {args.image_size} (HR pixels)")
    print(f"  dry run      = {args.dry_run}")
    print()

    t0 = time.time()
    H_hr = int(args.image_size)
    if H_hr % 2 != 0:
        print(f"ERROR: image_size must be even (got {H_hr})")
        return 1
    # HR side in arcsec → HLSP-pixel side needed before resample to HR.
    hr_side_arcsec   = H_hr * Config.DEFAULT_PIXEL_SCALE      # e.g. 510 × 0.05 = 25.5"
    hlsp_side_pix    = int(np.ceil(hr_side_arcsec / 0.03))     # cutout size in HLSP pixels

    print(f"[1/4] indexing HLSP tiles ...")
    if not os.path.isdir(args.hlsp_dir):
        print(f"ERROR: HLSP dir not found: {args.hlsp_dir}")
        print("       Run scripts/fasrc_download_hst_hlsp.py first.")
        return 1
    tiles = HLSPTileIndex(args.hlsp_dir)
    print(f"      {len(tiles.entries)} tiles indexed")

    print(f"[2/4] loading differential kernel + COSMOS catalog ...")
    if not os.path.isfile(args.kernel):
        print(f"ERROR: differential kernel not found at {args.kernel}")
        print("       Run scripts/fasrc_compute_differential_kernel.py first.")
        return 1
    from euclid_polish.sky.cosmos2025 import open_cosmos2025
    from euclid_polish.sky.differential_kernel import DifferentialKernel
    dk = DifferentialKernel.from_fits(args.kernel)
    catalog = open_cosmos2025()
    print(f"      kernel shape = {dk.data.shape} DC gain = {dk.dc_gain:.4f}")
    print(f"      {len(catalog):,} catalog galaxies after quality cuts")

    if args.dry_run:
        print("\nDRY RUN — would synthesise "
              f"{args.n_train + args.n_valid} pairs at {H_hr}² HR.")
        runtime = time.time() - t0
        print(f"\nRUNTIME_SECONDS={runtime:.1f}")
        return 0

    print(f"[3/4] selecting galaxies that fall on the HLSP coverage ...")
    # Walk catalog row-by-row, keep galaxies whose RA/Dec lands on a tile.
    rng = np.random.default_rng()
    catalog_indices = np.arange(len(catalog))
    rng.shuffle(catalog_indices)
    target_total = args.n_train + args.n_valid

    from astropy.io import fits
    from astropy.nddata import Cutout2D
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.wcs import WCS
    from euclid_polish.sky.tfrecord import open_multiband_writer
    from euclid_polish.sky.types import MultiBandSkyImage

    print(f"[4/4] streaming pairs to {args.output_dir} ...")
    pairs_written = 0
    pairs_skipped = 0
    pairs_per_subset = {"train": args.n_train, "validate": args.n_valid}

    # Open both clean + dirty writers per subset.
    summary = {"subsets": {}}
    for subset, target_n in pairs_per_subset.items():
        sub_done = 0
        with open_multiband_writer(
            f"clean_{subset}", records_dir=args.output_dir,
        ) as cw, open_multiband_writer(
            f"dirty_{subset}", records_dir=args.output_dir,
        ) as dw, open_multiband_writer(
            f"hr_{subset}", records_dir=args.output_dir,
        ) as hw:
            for i in catalog_indices:
                if sub_done >= target_n:
                    break
                ra  = float(catalog.ra_deg[i])
                dec = float(catalog.dec_deg[i])
                tile_path = tiles.find_tile(ra, dec)
                if tile_path is None:
                    pairs_skipped += 1
                    continue
                try:
                    with fits.open(tile_path, memmap=True) as hdul:
                        sci = next(
                            (e for e in hdul
                             if e.is_image and e.data is not None),
                            None,
                        )
                        if sci is None:
                            pairs_skipped += 1
                            continue
                        wcs = WCS(sci.header)
                        cutout = Cutout2D(
                            sci.data, SkyCoord(ra * u.deg, dec * u.deg),
                            size=(hlsp_side_pix, hlsp_side_pix),
                            wcs=wcs, mode="strict",
                        )
                except Exception as e:
                    print(f"      cutout failed @ ({ra:.4f}, {dec:.4f}): "
                          f"{type(e).__name__}: {e}")
                    pairs_skipped += 1
                    continue

                # Resample HLSP 0.03″ → HR 0.05″, then centre-crop to target.
                hr_resampled = _resample_hlsp_to_hr(
                    np.asarray(cutout.data, dtype=np.float32),
                )
                Hh, Wh = hr_resampled.shape
                if Hh < H_hr or Wh < H_hr:
                    pairs_skipped += 1
                    continue
                i0 = (Hh - H_hr) // 2
                j0 = (Wh - H_hr) // 2
                hr_clean = hr_resampled[i0:i0 + H_hr, j0:j0 + H_hr]
                # Background subtract: median outer-annulus.
                annulus = np.concatenate([
                    hr_clean[:8, :].ravel(),  hr_clean[-8:, :].ravel(),
                    hr_clean[8:-8, :8].ravel(), hr_clean[8:-8, -8:].ravel(),
                ])
                bg = float(np.median(annulus[np.isfinite(annulus)]))
                hr_clean = np.maximum(hr_clean - bg, 0.0)
                # Broadcast to 4 bands with per-galaxy electron flux.
                fluxes = tuple(
                    float(catalog.bulge_flux_e[i, k]
                          + catalog.disk_flux_e[i, k])
                    for k in range(Config.NUM_LR_CHANNELS)
                )
                hr_cube = _broadcast_hst_to_4bands(hr_clean, fluxes)
                if hr_cube is None:
                    pairs_skipped += 1
                    continue

                # Forward-model to LR.
                lr_cube = _make_pair(hr_cube, diff_kernel=dk.data, rng=rng)

                # Wrap as MultiBandSkyImage and write.
                clean_img = MultiBandSkyImage(
                    data=hr_cube, pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                    band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True,
                    metadata={"source": "hst_hlsp", "cosmos_id": int(catalog.catalog_id[i])},
                )
                dirty_img = MultiBandSkyImage(
                    data=lr_cube,
                    pixel_scale_arcsec=Config.VIS_PIXEL_SCALE_ARCSEC,
                    band_names=Config.LR_INPUT_BAND_NAMES, is_clean=False,
                    metadata={"source": "hst_hlsp", "cosmos_id": int(catalog.catalog_id[i])},
                )
                hr_vis_only = MultiBandSkyImage(
                    data=hr_cube[..., :1].copy(),
                    pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
                    band_names=("VIS",), is_clean=True,
                    metadata={"source": "hst_hlsp", "cosmos_id": int(catalog.catalog_id[i])},
                )
                cw.write(clean_img, index=sub_done)
                dw.write(dirty_img, index=sub_done)
                hw.write(hr_vis_only, index=sub_done)
                sub_done += 1
                pairs_written += 1
                if sub_done % 50 == 0:
                    print(f"      {subset}: {sub_done}/{target_n} written")

        summary["subsets"][subset] = {
            "written": sub_done,
            "target":  target_n,
        }
        print(f"      ✓ {subset}: {sub_done} pairs written")

    runtime = time.time() - t0
    summary["pairs_written"] = pairs_written
    summary["pairs_skipped"] = pairs_skipped
    summary["elapsed_s"]     = round(runtime, 1)
    with open(os.path.join(args.output_dir, "generation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"  wrote {pairs_written} pairs total ({pairs_skipped} skipped — "
          f"position fell outside HLSP coverage or cutout failed)")
    print(f"\nRUNTIME_SECONDS={runtime:.1f}")
    print(f"PAIRS_WRITTEN={pairs_written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
