#!/usr/bin/env python
"""Build A satisfying ``A ⊛ H = E`` and save as a FITS sidecar.

Loads:
  * the HST F814W ePSF written by ``scripts/fasrc_extract_hst_psf.py``
    (``$DATA_DIR/hst_psf/F814W.fits``)
  * the Euclid VIS empirical PSF (``$DATA_DIR/euclid_psf/euclid_psf_VIS.fits``)

Resamples both onto the project HR grid (``Config.DEFAULT_PIXEL_SCALE`` =
0.05″/pix) and solves the differential-kernel equation via
:func:`euclid_polish.sky.differential_kernel.compute_differential_kernel`.

The output kernel is what we apply to HST cutouts at training time to
generate "Euclid-equivalent" LR — see
``scripts/fasrc_generate_hst_tfrecords.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config


HST_PSF_PATH = os.path.join(Config.DATA_DIR, "hst_psf", "F814W.fits")
DIFF_KERNEL_PATH = os.path.join(
    Config.DATA_DIR, "hst_psf", "diff_kernel_VIS.fits",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regularisation", type=float, default=1e-3,
                   help="Wiener regulariser as a fraction of max|H_hat|. "
                        "Larger → smoother kernel, more low-pass.")
    p.add_argument("--hst-psf", default=HST_PSF_PATH,
                   help="Path to HST F814W ePSF FITS.")
    p.add_argument("--output", default=DIFF_KERNEL_PATH,
                   help="Where to write the differential kernel FITS.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done and exit.")
    return p.parse_args()


def _resample_to_hr_grid(psf_data: np.ndarray, src_scale: float) -> np.ndarray:
    """Resample a PSF from ``src_scale`` arcsec/pix to the HR grid."""
    from scipy.ndimage import zoom
    target_scale = Config.DEFAULT_PIXEL_SCALE   # 0.05"
    if abs(src_scale - target_scale) < 1e-6:
        return np.asarray(psf_data, dtype=np.float64)
    factor = src_scale / target_scale            # < 1 if src is finer
    out = zoom(psf_data.astype(np.float64), zoom=factor,
               order=3, mode="constant", grid_mode=False)
    # Re-normalise to unit flux (zoom doesn't preserve sum exactly).
    s = out.sum()
    if s > 0:
        out = out / s
    return out


def _centre_crop_to(a: np.ndarray, side: int) -> np.ndarray:
    """Crop ``a`` to the centred square of ``side`` pixels (odd)."""
    H, W = a.shape
    if side > H or side > W:
        # pad if a is smaller than side
        pad_h = max(0, (side - H + 1) // 2)
        pad_w = max(0, (side - W + 1) // 2)
        a = np.pad(a, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
        H, W = a.shape
    i0 = (H - side) // 2
    j0 = (W - side) // 2
    return a[i0:i0 + side, j0:j0 + side]


def main() -> int:
    args = parse_args()
    print("=" * 64)
    print(f"  Differential kernel A = E / H")
    print("=" * 64)
    print(f"  HST PSF       = {args.hst_psf}")
    print(f"  Euclid PSF    = {Config.BAND_VIS.psf_fits_filename} "
          f"(in {Config.EUCLID_PSF_DIR})")
    print(f"  regularisation = {args.regularisation}")
    print(f"  output         = {args.output}")
    print()

    t0 = time.time()

    print(f"[1/3] loading HST and Euclid PSFs ...")
    from astropy.io import fits
    from euclid_polish.euclid.psf_library import psf_path_for_band
    from euclid_polish.euclid.types import PSF
    from euclid_polish.sky.differential_kernel import (
        DifferentialKernel, compute_differential_kernel,
    )

    if not os.path.isfile(args.hst_psf):
        print(f"ERROR: HST PSF not found at {args.hst_psf}")
        print("       Run scripts/fasrc_extract_hst_psf.py first.")
        return 1

    euclid_psf_path = psf_path_for_band("VIS")
    if not euclid_psf_path or not os.path.isfile(euclid_psf_path):
        print(f"ERROR: Euclid VIS PSF not found at {euclid_psf_path}")
        return 1

    # HST
    with fits.open(args.hst_psf, memmap=False) as hdul:
        hst_data  = np.asarray(hdul[0].data, dtype=np.float64)
        hst_scale = float(hdul[0].header.get("PIXSCALE", 0.015))
        hst_filter = str(hdul[0].header.get("FILTER", "F814W"))
    print(f"      HST PSF : shape={hst_data.shape} scale={hst_scale:.4f}\"/pix"
          f"  flux={hst_data.sum():.4f}")

    # Euclid
    euclid_psf = PSF.from_fits(euclid_psf_path)
    euclid_data  = np.asarray(euclid_psf.data, dtype=np.float64)
    euclid_scale = euclid_psf.pixel_scale
    print(f"      Euclid PSF : shape={euclid_data.shape} "
          f"scale={euclid_scale:.4f}\"/pix flux={euclid_data.sum():.4f}")

    if args.dry_run:
        print(f"\nDRY RUN — would resample both to {Config.DEFAULT_PIXEL_SCALE}\"/pix"
              " and solve the differential kernel")
        runtime = time.time() - t0
        print(f"\nRUNTIME_SECONDS={runtime:.1f}")
        return 0

    print(f"[2/3] resampling both PSFs onto the HR grid "
          f"({Config.DEFAULT_PIXEL_SCALE:.3f}\"/pix) ...")
    e_hr = _resample_to_hr_grid(euclid_data, euclid_scale)
    h_hr = _resample_to_hr_grid(hst_data,    hst_scale)
    side = max(e_hr.shape[0], h_hr.shape[0]) | 1     # force odd
    e_hr = _centre_crop_to(e_hr, side)
    h_hr = _centre_crop_to(h_hr, side)
    # Re-normalise after cropping (small flux losses at the edges).
    e_hr /= e_hr.sum(); h_hr /= h_hr.sum()
    print(f"      common grid : {e_hr.shape}, both unit-flux normalised")

    print(f"[3/3] solving A_hat = E_hat · conj(H_hat) / (|H_hat|² + reg²) ...")
    a = compute_differential_kernel(
        e_hr, h_hr, regularisation=args.regularisation,
    )
    print(f"      kernel shape = {a.shape}")
    print(f"      DC gain      = {a.sum():.4f}  (should be ~1)")

    dk = DifferentialKernel(
        data=a,
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        euclid_band="VIS",
        hst_filter=hst_filter,
        regularisation=args.regularisation,
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    dk.save(args.output)
    print(f"  wrote kernel → {args.output}")

    runtime = time.time() - t0
    print(f"\nRUNTIME_SECONDS={runtime:.1f}")
    print(f"KERNEL_DC_GAIN={a.sum():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
