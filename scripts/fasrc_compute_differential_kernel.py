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

from scipy.ndimage import zoom
from astropy.io import fits

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.euclid.psf_library import psf_path_for_band
from euclid_polish.euclid.types import PSF
from euclid_polish.observability.reporter import Reporter
from euclid_polish.sky.differential_kernel import (
    DifferentialKernel, compute_differential_kernel,
)


HST_PSF_PATH = os.path.join(Config.DATA_DIR, "hst_psf", "F814W.fits")
DIFF_KERNEL_PATH = os.path.join(
    Config.DATA_DIR, "hst_psf", "diff_kernel_VIS.fits",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regularisation", type=float, default=1e-2,
                   help="Wiener regulariser as a fraction of max|H_hat|. "
                        "Larger → smoother kernel, more low-pass. The "
                        "default 1e-2 is calibrated for empirical PSFs "
                        "(measured ePSFs have noise that 1e-3 would "
                        "amplify ~1000× through 1/Ĥ at high frequencies).")
    p.add_argument("--common-side", type=int, default=511,
                   help="Force both PSFs to N×N (odd) after resampling and "
                        "before the FFT. The previous behaviour (pick "
                        "max of the two native sizes) zero-pads the "
                        "smaller PSF, creating a sharp content→zero "
                        "discontinuity whose FFT is a sinc-like ringing "
                        "that leaks back into Â. 511 (≈25.6″ at 0.05″/px) "
                        "covers essentially all of a 513² HST F814W ePSF "
                        "and the full Euclid VIS 6-spike pattern (spike "
                        "tips reach ~17″ from centre at most), without "
                        "the wasted zero canvas of a 1023² grid. The "
                        "script prints the % of flux retained after the "
                        "crop so you can spot the case where it's eating "
                        "real signal (<99% is a red flag — widen "
                        "--common-side).")
    p.add_argument("--hst-psf", default=HST_PSF_PATH,
                   help="Path to HST F814W ePSF FITS.")
    p.add_argument("--output", default=DIFF_KERNEL_PATH,
                   help="Where to write the differential kernel FITS.")
    p.add_argument("--bg-subtract", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Median-subtract + positivity floor on both PSFs "
                        "right before the FFT. The median of a star-stamp "
                        "ePSF where the PSF support is ~1/9 of the stamp "
                        "area is the background level — subtracting it "
                        "removes the additive bg offset and clipping "
                        "negatives kills the negative half of bg noise, "
                        "which would otherwise leak into Â through 1/Ĥ at "
                        "high frequencies. Re-normalises to sum=1 so the "
                        "DC gain of the kernel stays at 1. Use "
                        "--no-bg-subtract to disable. Only affects this "
                        "script — the on-disk PSF FITS files are never "
                        "modified.")
    p.add_argument("--border-pixels", type=int, default=10,
                   help="Zero out the first/last N rows AND columns of "
                        "each PSF after bg-subtract and before the FFT, "
                        "AND of the output kernel A after the FFT inverse. "
                        "On the inputs this cleans resampling/edge "
                        "artefacts that would otherwise inject high-freq "
                        "content into Â. On the output it suppresses the "
                        "1-2 px bright border that the Wiener inverse "
                        "leaves behind when reg is tiny (the noise floor "
                        "of Ĥ near Nyquist collects at the array edges "
                        "after `ifftshift`). 0 disables. Both inputs and "
                        "output are renormalised to sum=1 after masking "
                        "so the kernel's DC gain stays at 1.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done and exit.")
    return p.parse_args()


def _bg_subtract_and_clip(psf: np.ndarray) -> np.ndarray:
    """Median-subtract + positivity floor; renormalise to unit flux.

    Two things this is *not*:

    * Not a hard threshold-and-keep. We subtract the median first so PSF
      pixels lose the additive bg offset (otherwise the core ends up
      slightly over-weighted relative to the wings after renormalising).
    * Not a "smooth out the wings" operation. Only the *negative* tail
      of background noise is touched — positive wing pixels above the
      median pass through unchanged, so genuine low-amplitude PSF
      structure isn't clipped.

    Assumes the PSF's bright support occupies a small fraction of the
    stamp area (so the median tracks the per-pixel background). That's
    true for empirical star stamps after extraction; verify visually
    before turning this on for an unusual stamp size.
    """
    bg = float(np.median(psf))
    cleaned = np.maximum(psf - bg, 0.0).astype(np.float64)
    s = cleaned.sum()
    if s > 0:
        cleaned = cleaned / s
    return cleaned


def _zero_borders(psf: np.ndarray, *, border_pixels: int) -> np.ndarray:
    """Smoothly taper the outer ``border_pixels`` rows and columns of
    ``psf`` to zero with a cosine (Hann) ramp.

    A *hard* zero step at row ``border_pixels`` — the original v1 of
    this helper — is itself a sharp discontinuity. Its 2-D Fourier
    transform spreads as a sinc along the perpendicular axis, which
    after the Wiener inverse manifests as bright horizontal +
    vertical rays through the centre of ``A`` (and through ``A⊛H``).
    A cosine ramp from 0 at the very edge to 1 at the inner edge of
    the border region kills the discontinuity (and therefore the
    rays) while still suppressing whatever ePSF noise lives near the
    boundary, which was the original motivation.

    Geometry: pixel ``i`` in ``[0, border)`` is multiplied by
    ``0.5·(1 − cos(π·i/border))``. So pixel 0 is 0 (full attenuation)
    and pixel ``border`` is 1 (untouched). Same on the trailing edge.
    Interior pixels (between the two border strips) pass through.

    ``border_pixels=0`` is a no-op. Returns a unit-flux-renormalised
    array; raises if the taper would consume the whole stamp.
    """
    if border_pixels <= 0:
        return psf
    H, W = psf.shape
    b = int(border_pixels)
    if b * 2 >= min(H, W):
        raise ValueError(
            f"border_pixels={b} too large for PSF shape "
            f"{psf.shape} — taper would consume the whole stamp"
        )
    # Per-axis 1-D weight: cosine ramp on each side, 1 in the middle.
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(b) / b))
    wy = np.ones(H, dtype=np.float64); wy[:b] = ramp; wy[-b:] = ramp[::-1]
    wx = np.ones(W, dtype=np.float64); wx[:b] = ramp; wx[-b:] = ramp[::-1]
    out = psf.astype(np.float64, copy=False) * (wy[:, None] * wx[None, :])
    s = out.sum()
    if s > 0:
        out = out / s
    return out


def _resample_to_hr_grid(psf_data: np.ndarray, src_scale: float) -> np.ndarray:
    """Resample a PSF from ``src_scale`` arcsec/pix to the HR grid."""
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
    reporter = Reporter.from_env()
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

    reporter.set_stage("loading HST and Euclid PSFs")
    print(f"[1/3] loading HST and Euclid PSFs ...")

    if not os.path.isfile(args.hst_psf):
        reporter.error(f"HST PSF not found at {args.hst_psf}")
        print(f"ERROR: HST PSF not found at {args.hst_psf}")
        print("       Run scripts/fasrc_extract_hst_psf.py first.")
        return 1

    # ``psf_path_for_band`` keys off a ``BandConfig`` (it reads
    # ``.psf_fits_filename``), not a string — passing "VIS" gave the
    # cryptic ``AttributeError: 'str' object has no attribute
    # 'psf_fits_filename'`` because the wrong type was indexed into.
    euclid_psf_path = psf_path_for_band(Config.BAND_VIS)
    if not euclid_psf_path or not os.path.isfile(euclid_psf_path):
        reporter.error(f"Euclid VIS PSF not found at {euclid_psf_path}")
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

    reporter.set_stage("resampling both PSFs onto the HR grid")
    print(f"[2/3] resampling both PSFs onto the HR grid "
          f"({Config.DEFAULT_PIXEL_SCALE:.3f}\"/pix) ...")
    e_hr = _resample_to_hr_grid(euclid_data, euclid_scale)
    h_hr = _resample_to_hr_grid(hst_data,    hst_scale)
    print(f"      resampled : E {e_hr.shape}  H {h_hr.shape}")

    # Force a common, sensible grid. Picking max() of the two native
    # sizes (the previous behaviour) zero-pads the smaller PSF, which
    # creates a sharp content→zero boundary whose FFT is sinc-like
    # ringing — that ringing then dominates Â at high frequencies.
    # --common-side keeps both PSFs on a grid that fits their actual
    # support without dumping huge amounts of synthetic "zero data"
    # into the FFT.
    side = args.common_side | 1                       # force odd
    e_flux_before = float(e_hr.sum())
    h_flux_before = float(h_hr.sum())
    e_hr = _centre_crop_to(e_hr, side)
    h_hr = _centre_crop_to(h_hr, side)
    e_retained = float(e_hr.sum()) / e_flux_before if e_flux_before > 0 else 0.0
    h_retained = float(h_hr.sum()) / h_flux_before if h_flux_before > 0 else 0.0
    # Re-normalise after cropping (small flux losses at the edges).
    e_hr /= e_hr.sum(); h_hr /= h_hr.sum()
    print(f"      common grid : {e_hr.shape}, both unit-flux normalised")
    print(f"      flux retained after crop : "
          f"E {e_retained*100:.3f}%  H {h_retained*100:.3f}%")
    if e_retained < 0.99 or h_retained < 0.99:
        reporter.warn(
            f">1% of PSF flux outside --common-side={args.common_side} crop"
        )
        print(f"      WARNING: >1% of PSF flux was outside the "
              f"--common-side={args.common_side} crop. The spikes "
              f"probably extend further than the crop — bump "
              f"--common-side to 511 (or higher) and re-run.")

    if args.bg_subtract:
        e_bg = float(np.median(e_hr))
        h_bg = float(np.median(h_hr))
        e_hr = _bg_subtract_and_clip(e_hr)
        h_hr = _bg_subtract_and_clip(h_hr)
        print(f"      bg-subtract : E median={e_bg:+.3e}  "
              f"H median={h_bg:+.3e} → subtracted, negatives clipped, "
              f"renormalised")

    if args.border_pixels > 0:
        # Flux still inside the border *after* bg-subtract: anything
        # non-trivial here means the PSF support extends into the mask
        # region and we're clipping real wing pixels. <0.1 % is safe.
        b = args.border_pixels
        e_border_flux = float(e_hr[:b, :].sum() + e_hr[-b:, :].sum()
                              + e_hr[:, :b].sum() + e_hr[:, -b:].sum()
                              # corners counted twice above — subtract them.
                              - e_hr[:b, :b].sum() - e_hr[:b, -b:].sum()
                              - e_hr[-b:, :b].sum() - e_hr[-b:, -b:].sum())
        h_border_flux = float(h_hr[:b, :].sum() + h_hr[-b:, :].sum()
                              + h_hr[:, :b].sum() + h_hr[:, -b:].sum()
                              - h_hr[:b, :b].sum() - h_hr[:b, -b:].sum()
                              - h_hr[-b:, :b].sum() - h_hr[-b:, -b:].sum())
        e_hr = _zero_borders(e_hr, border_pixels=b)
        h_hr = _zero_borders(h_hr, border_pixels=b)
        print(f"      border-zero : {b} px each side → "
              f"E flux discarded={e_border_flux*100:.3f}%, "
              f"H flux discarded={h_border_flux*100:.3f}%, "
              f"renormalised")

    reporter.set_stage("solving A_hat via Wiener inverse")
    print(f"[3/3] solving A_hat = E_hat · conj(H_hat) / (|H_hat|² + reg²) ...")
    a = compute_differential_kernel(
        e_hr, h_hr, regularisation=args.regularisation,
    )
    print(f"      kernel shape = {a.shape}")
    print(f"      DC gain      = {a.sum():.4f}  (should be ~1)")

    if args.border_pixels > 0:
        # Mirror the input-side border-zero on the output kernel itself.
        # The Wiener inverse amplifies Ĥ's noise floor where |Ĥ| is small;
        # at low (or zero) regularisation that noise is concentrated at
        # high spatial frequencies, which `ifftshift` + central crop park
        # at the array borders as a bright 1-2 px rim. A's real support
        # is centred — same assumption as the input PSFs — so zeroing the
        # outer ``--border-pixels`` rows/cols and renormalising to sum=1
        # removes the artefact without touching the in-band response.
        b = args.border_pixels
        a64 = a.astype(np.float64, copy=False)
        a_border_flux = float(a64[:b, :].sum() + a64[-b:, :].sum()
                              + a64[:, :b].sum() + a64[:, -b:].sum()
                              - a64[:b, :b].sum() - a64[:b, -b:].sum()
                              - a64[-b:, :b].sum() - a64[-b:, -b:].sum())
        a = _zero_borders(a64, border_pixels=b).astype(np.float32)
        print(f"      kernel border-zero : {b} px each side → "
              f"flux discarded={a_border_flux*100:.4f}% (renormalised)")
        print(f"      DC gain (post-mask) = {a.sum():.4f}")

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
