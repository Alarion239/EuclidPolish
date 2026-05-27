#!/usr/bin/env python
"""Generate clean (HST) + dirty (Euclid-forward) TFRecord pairs.

For each selected COSMOS2025 sky position (the catalog only picks the
RA/Dec; per-source catalog fluxes are not used here):

  1. Cut an HR-grid-sized HST F814W cutout from the local HLSP tile that
     contains it (``Cutout2D`` + WCS). HLSP delivers sky-subtracted
     mosaics in e⁻/s, so we use those pixel values as-is — no extra
     bg-subtract, no clip-to-zero. A 25″ chunk of sky contains many
     visible sources and HST already gives each of them its real
     F814W photometry; we want to preserve all of them.
  2. Resample 0.03″ HLSP → 0.05″ HR with an area correction that
     preserves total flux through the zoom (scipy's cubic-spline zoom
     interpolates surface brightness; we multiply by the per-pixel-area
     ratio so the integrated flux is also conserved).
  3. Convert HST F814W rate → Euclid VIS electrons via the AB-zeropoint
     ratio and the VIS stack integration time:
        VIS_HR_e = rate × 10^(-0.4·(ZP_VIS - ZP_HST)) × t_total_VIS
     NISP HR channels are painted as ``VIS_HR_e × typical_ratio[band]``
     using a catalog-derived median ``e_band / e_VIS`` — a per-pixel
     global colour. Per-source colours are lost but the absolute scale
     is correct for noise statistics.
  4. Reject the stamp if its brightest pixel would make the analytic
     forward operator ``A`` produce noise-amplified ringing larger than
     ``max_relative_noise × σ_LR`` — see ``_is_stamp_too_bright``.
  5. Apply the pre-computed differential kernel A to get the Euclid-PSF
     view, sum-rebin to LR scale (0.10″/pix), add per-band Euclid noise.
  6. Write the (HR, LR) pair into the standard ``MultiBandSkyImage``
     TFRecord layout — same schema the synthetic generator uses.

Output records land under ``$DATA_DIR/images/records_v2_hst/`` so the
trainer can mix them with the existing synthetic records via the
``hst_fraction`` knob.
"""

from __future__ import annotations

import os

from concurrent.futures import (
FIRST_COMPLETED, ProcessPoolExecutor, wait,
)

# Force-hide GPUs BEFORE TF is imported, even if SLURM allocated one.
# Heavy work here is ``scipy.signal.fftconvolve`` (FFT, CPU only) over
# the 511² baseline kernel × 4 bands per cutout — no GPU path exists.
# ``setdefault`` would be a no-op when SLURM has already set
# ``CUDA_VISIBLE_DEVICES`` (which it does on every ``--gres=gpu:N``
# job), so we use a hard assignment to override SLURM's value.
_prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if _prev_cuda:
    print(f"[fasrc_generate_hst_tfrecords] SLURM set "
          f"CUDA_VISIBLE_DEVICES={_prev_cuda!r}; overriding to '' because "
          f"this pipeline is CPU-only by design (FFT + process pool). "
          f"Submit without --gres=gpu next time to free the device "
          f"for someone else.", flush=True)

# Cap per-process thread counts BEFORE numpy/scipy are imported. With a
# 16-worker pool on 16 SLURM CPUs, the default lets each worker spawn
# 16 intra-op threads → 256 OS threads contending for 16 cores → all
# per-cutout work crawls. Pinning to 1 thread per process restores 1:1
# worker-to-core mapping. setdefault honours an explicit override from
# the sbatch script if the caller really wants multi-threaded BLAS.
os.environ.setdefault("OMP_NUM_THREADS",         "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",    "1")
os.environ.setdefault("MKL_NUM_THREADS",         "1")

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from scipy import signal as scipy_signal
from scipy.ndimage import zoom

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.observability.reporter import Reporter
from euclid_polish.sky.cosmos2025 import open_cosmos2025
from euclid_polish.sky.differential_kernel import DifferentialKernel
from euclid_polish.sky.noise import apply_band_noise
from euclid_polish.sky.tfrecord import open_multiband_writer
from euclid_polish.sky.types import MultiBandSkyImage


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
    p.add_argument("--kernel",     default=KERNEL_PATH,
                   help="Analytic Wiener differential kernel FITS, the "
                        "forward operator A applied to every HR stamp.")
    p.add_argument("--output-dir", default=OUT_DIR)
    p.add_argument("--n-workers", type=int, default=_default_n_workers(),
                   help="Process pool size for parallel cutout + forward "
                        "modelling. 0 disables the pool (single-process "
                        "fallback, useful for debugging). Default reads "
                        "SLURM_CPUS_PER_TASK if set, else os.cpu_count().")
    p.add_argument("--max-mag", type=float, default=24.0,
                   help="Brightest-only catalog cut on the COSMOS2025 "
                        "total VIS magnitude (bulge+disk). Default 24.0 "
                        "sits inside Euclid's single-stack 5σ "
                        "point-source depth (~24.5 in VIS) so generated "
                        "pairs are visually obvious in the dirty image "
                        "— faint sources below the noise floor make "
                        "training data the model can't actually learn "
                        "from. Bump to 25 if you specifically want to "
                        "include detection-threshold galaxies; lower "
                        "(22, 21) to focus on bright, well-resolved "
                        "morphology only.")
    p.add_argument("--max-relative-noise", type=float, default=5.0,
                   help="Reject any HR stamp whose brightest pixel "
                        "would make A(ε) (HST-side shot noise propagated "
                        "through the analytic kernel) exceed this many "
                        "times the per-pixel Euclid LR noise budget σ_LR. "
                        "Default 5 means A(ε) up to 5× σ_LR is fine; "
                        "ringing larger than that around bright/saturated "
                        "stars dominates the training signal and is what "
                        "we're throwing away.")
    p.add_argument("--oversample", type=float, default=3.0,
                   help="Catalog candidate pool size as a multiple of "
                        "(n_train + n_valid). Bright-stamp rejection "
                        "throws candidates out, so we need spares.")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan only — no FITS reads, no TFRecord writes.")
    return p.parse_args()


def _default_n_workers() -> int:
    """How many worker processes to spawn by default.

    Honour SLURM's allocation when running as a job; fall back to the
    machine's cpu_count() for local runs. Cap at 1 if neither is set.
    """
    env = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, os.cpu_count() or 1)


def _euclid_vis_sigma_lr_per_pixel() -> float:
    """Per-pixel Euclid VIS LR noise σ in electrons (no signal term).

    The noise budget for a single LR pixel of the VIS stack is the
    sky + dark Poisson floor combined with the read-noise quadrature
    sum over ``n_exposures``:

        σ_LR = sqrt(sky_e + dark_e + n_exp · read_noise_e²)

    where ``sky_e`` and ``dark_e`` are integrated over the LR pixel
    area and total stack time. This matches the ``sigma_floor_e``
    expression used by :func:`apply_band_noise`.
    """
    band = Config.BAND_VIS
    t_total    = band.t_total_s
    pixel_area = band.pixel_scale_lr_arcsec ** 2
    sky_e   = band.sky_e_per_s_per_arcsec2 * pixel_area * t_total
    dark_e  = band.dark_e_per_s_per_pix * t_total
    read_var = band.n_exposures * (band.read_noise_e ** 2)
    return float(np.sqrt(sky_e + dark_e + read_var))


# ---------------------------------------------------------------------------
# Tile manifest
# ---------------------------------------------------------------------------

class HLSPTileIndex:
    """Lightweight RA/Dec footprint cache so we can map galaxy → tile.

    Avoids opening every tile FITS just to figure out which one contains
    a given (ra, dec). Reads only each tile's primary header once.
    """

    def __init__(self, hlsp_dir: str):
        self.entries: List[Dict] = []
        skipped: List[Tuple[str, str]] = []   # (fname, reason)
        for fname in sorted(os.listdir(hlsp_dir)):
            if not (fname.endswith(".fits")
                    and fname.startswith("hlsp_cosmos_hst_acs-wfc_mosaic")):
                continue
            path = os.path.join(hlsp_dir, fname)
            # Wrap the per-tile read so one truncated / corrupt FITS
            # (e.g. left over from a killed download) doesn't abort the
            # whole indexing pass. astropy can raise here, in WCS, or
            # mid-iteration in next(); a bare ``except Exception`` is
            # the right blast radius — we record the failure and keep
            # building the index from the surviving tiles.
            try:
                with fits.open(path, memmap=True) as hdul:
                    sci = next(
                        (e for e in hdul
                         if e.is_image and e.data is not None), None,
                    )
                    if sci is None:
                        skipped.append((fname, "no image HDU"))
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
            except Exception as e:
                skipped.append((fname, f"{type(e).__name__}: {e}"))
                continue
        if skipped:
            print(f"      WARN: skipped {len(skipped)} bad tile(s):")
            for fname, reason in skipped:
                print(f"        - {fname}  ({reason})")
            print("      Re-run the tiles download to repair "
                  "(it'll detect truncated files via size mismatch).")
        if not self.entries:
            raise FileNotFoundError(
                f"no usable HLSP tiles in {hlsp_dir} "
                f"({len(skipped)} were corrupt/skipped)"
            )

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
                         hlsp_scale: float = None,
                         hr_scale: float = None) -> np.ndarray:
    """Resample 0.03″ HLSP cutout onto the 0.05″ HR grid (flux-conserving).

    ``scipy.ndimage.zoom`` interpolates per-pixel values: a zoomed-down
    output pixel ≈ the underlying input value at that location, NOT
    the integral over the larger HR pixel area. So zoom alone preserves
    surface brightness but loses ~36 % of total flux on a 0.03″ → 0.05″
    downsample. We multiply by ``(HR_pixel_area / HLSP_pixel_area)``
    afterwards so each output value is the rate integrated over the HR
    pixel — matching the units the downstream forward model expects.
    """
    hlsp_scale = (hlsp_scale if hlsp_scale is not None
                  else Config.HST_HLSP_PIXEL_SCALE_ARCSEC)
    hr_scale = hr_scale or _hr_pixel_side_arcsec()
    factor = hlsp_scale / hr_scale   # 0.6 → output is smaller
    out = zoom(hlsp_cutout.astype(np.float32),
               zoom=factor, order=3, mode="constant")
    area_correction = (hr_scale / hlsp_scale) ** 2    # ≈ 2.78
    return (out * np.float32(area_correction)).astype(np.float32)


def _kernel_forward(hr_band: np.ndarray, diff_kernel: np.ndarray) -> np.ndarray:
    """Apply the analytic differential kernel A: HST-PSF → Euclid-PSF.

    Pure FFT convolution on a single band image — no learned residual,
    no denoiser. The kernel is the one solved for by
    ``scripts/fasrc_compute_differential_kernel.py`` (saved as
    ``$DATA_DIR/hst_psf/diff_kernel_VIS.fits``).
    """
    return scipy_signal.fftconvolve(hr_band, diff_kernel, mode="same")


def _is_stamp_too_bright(
    hr_stamp: np.ndarray,
    *,
    diff_kernel: np.ndarray,
    sigma_lr_band: float,
    max_relative_noise: float,
) -> Tuple[bool, Dict[str, float]]:
    """Decide whether the analytic-A forward pass will ring around a bright pixel.

    The forward operator A is linear, so after applying it to a noisy
    HST observation::

        A(HST_obs) = A(H·I + ε) = E·I + A(ε)

    The artefact term ``A(ε)`` has its biggest contribution at the
    brightest pixel — Poisson shot noise variance scales with signal,
    so the standard deviation of ``A(ε)`` at the peak pixel is

        σ(A·ε)(x_max) ≈ |A|_peak · σ(ε)(x_max) ≈ |A|_peak · √S_max

    where ``S_max`` is the brightest HR pixel value in Euclid-scaled
    electrons (VIS channel — the band the kernel was solved for) and
    ``|A|_peak = max(abs(diff_kernel))``. We reject the stamp when
    this contribution would exceed ``max_relative_noise`` times the
    per-pixel Euclid LR noise budget σ_LR (sky + dark + read² · n_exp):

        √S_max · |A|_peak  >  k · σ_LR

    i.e. ``S_max > (k · σ_LR / |A|_peak)²``. Non-positive S_max is
    handled by the ``max(S_max, 0)`` clamp inside the sqrt — a stamp
    where the brightest pixel is ≤ 0 cannot have shot noise.
    """
    if max_relative_noise <= 0:
        # Caller asked to disable the filter — let everything through.
        return False, {"S_max": float(np.nan), "threshold_S_max": float(np.inf)}
    a_peak = float(np.max(np.abs(diff_kernel)))
    if a_peak <= 0 or sigma_lr_band <= 0:
        # Defensive: a zero kernel or zero noise budget makes the
        # threshold either +inf or 0; either way the math degenerates.
        return False, {
            "S_max": float(np.max(hr_stamp[..., 0])),
            "threshold_S_max": float(np.inf),
        }
    # VIS is channel 0 by the cube layout in _hst_to_euclid_hr_cube.
    s_max = float(np.max(hr_stamp[..., 0]))
    s_clamped = max(s_max, 0.0)
    artifact_sigma = a_peak * float(np.sqrt(s_clamped))
    threshold = float(max_relative_noise) * float(sigma_lr_band)
    reject = artifact_sigma > threshold
    threshold_s_max = (threshold / a_peak) ** 2 if a_peak > 0 else float("inf")
    diagnostics = {
        "S_max":           s_max,
        "a_peak":          a_peak,
        "sigma_lr":        float(sigma_lr_band),
        "artifact_sigma":  artifact_sigma,
        "threshold":       threshold,
        "threshold_S_max": threshold_s_max,
    }
    return reject, diagnostics


def _make_pair(
    hr_cutout_4ch: np.ndarray,            # (H, W, 4) HR clean
    *,
    diff_kernel: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Forward-model the HR cube to a 4-band LR cube at 0.10″/pix.

    For each band: ``A ⊛ ·`` (analytic differential kernel) →
    sum-rebin ×2 → Poisson + read + sky + dark.
    """

    H, W, C = hr_cutout_4ch.shape
    assert H % 2 == 0 and W % 2 == 0
    lr_cube = np.zeros((H // 2, W // 2, C), dtype=np.float32)
    for k, band_name in enumerate(Config.LR_INPUT_BAND_NAMES):
        band = Config.get_band(band_name)
        convolved = _kernel_forward(hr_cutout_4ch[..., k], diff_kernel)
        # Sum-rebin ×2 (photometric: conserves total electrons per LR pixel).
        rebinned = MultiBandSkyImage.rebin_array(convolved, 2)
        # Apply per-band Euclid noise (Poisson + read; artifacts off for
        # synthetic HST templates — adding them is a knob we can flip
        # later if the trained model overfits to the smoother dirty path).
        lr_cube[..., k] = apply_band_noise(rebinned, band, rng,
                                           add_artifacts=False)
    return lr_cube


# ---------------------------------------------------------------------------
# Per-galaxy worker — top-level so ProcessPoolExecutor can pickle it
# ---------------------------------------------------------------------------

# Module globals populated by ``_init_worker`` once per process. None
# before init; checking them in ``_process_one_galaxy`` is what catches
# a missing ``initializer=`` in the pool setup.
_WORKER_KERNEL:             Optional[np.ndarray] = None
_WORKER_IMAGE_SIZE:         int                  = 0
_WORKER_TYPICAL_RATIOS:     Optional[np.ndarray] = None
_WORKER_SIGMA_LR:           float                = 0.0
_WORKER_MAX_RELATIVE_NOISE: float                = 0.0


def _init_worker(
    kernel_path: str,
    image_size: int,
    typical_band_ratios: np.ndarray,
    sigma_lr_band: float,
    max_relative_noise: float,
) -> None:
    """Called once per worker process at pool startup.

    Loads the analytic differential kernel ``A`` and pins the per-band
    electron ratios + noise threshold into module-level globals so each
    worker pays the I/O + parse cost exactly once — not per-galaxy.
    Pickling the kernel through every task argument would be ~1 MB per
    submission × 6400 galaxies = 6 GB of pickle churn for nothing.
    """
    global _WORKER_KERNEL, _WORKER_IMAGE_SIZE, _WORKER_TYPICAL_RATIOS
    global _WORKER_SIGMA_LR, _WORKER_MAX_RELATIVE_NOISE
    _WORKER_KERNEL             = DifferentialKernel.from_fits(kernel_path).data
    _WORKER_IMAGE_SIZE         = int(image_size)
    _WORKER_TYPICAL_RATIOS     = np.asarray(typical_band_ratios, dtype=np.float32)
    _WORKER_SIGMA_LR           = float(sigma_lr_band)
    _WORKER_MAX_RELATIVE_NOISE = float(max_relative_noise)

    # Heartbeat so a future stall in worker init is visible from the
    # SLURM log instead of looking identical to a healthy "just hasn't
    # finished cutout #1 yet" state.
    print(f"      [worker {os.getpid()}] init complete "
          f"(kernel={_WORKER_KERNEL.shape}, σ_LR={_WORKER_SIGMA_LR:.2f}, "
          f"max_rel_noise={_WORKER_MAX_RELATIVE_NOISE:.2f})", flush=True)


def _process_one_galaxy(
    task: Tuple[int, float, float, str, int, int],
) -> Tuple[Optional[Tuple[int, np.ndarray, np.ndarray]], Optional[str]]:
    """Cut + resample + forward-model one HST chunk of sky.

    Pure function — no shared state beyond the module globals set by
    ``_init_worker``. The kernel is the only non-trivial state and is
    immutable across calls. The RNG is seeded per task so two workers
    processing the same catalog_idx would produce identical output.

    Note: ``catalog_idx`` only selects *which RA/Dec to centre on* —
    its per-band catalog fluxes are no longer used. The HST cutout's
    own photometry sets the brightness of every source in the field.

    Returns ``((catalog_idx, hr_cube, lr_cube), None)`` on success.
    Returns ``(None, reason)`` when the chunk can't be cut (off-tile,
    malformed FITS, cutout smaller than the target HR size, HST data
    all-NaN, or stamp rejected by the bright-pixel filter).
    """
    catalog_idx, ra, dec, tile_path, hlsp_side_pix, seed = task
    if _WORKER_TYPICAL_RATIOS is None or _WORKER_KERNEL is None:
        # If we ever forget the initializer, fail loudly rather than
        # silently using None and crashing deep in the FFT.
        raise RuntimeError(
            "_process_one_galaxy called without _init_worker — "
            "pass initializer=_init_worker to the pool."
        )

    try:
        with fits.open(tile_path, memmap=True) as hdul:
            sci = next(
                (e for e in hdul if e.is_image and e.data is not None),
                None,
            )
            if sci is None:
                return None, "no-image-hdu"
            wcs = WCS(sci.header)
            cutout = Cutout2D(
                sci.data,
                SkyCoord(ra * u.deg, dec * u.deg),
                size=(hlsp_side_pix, hlsp_side_pix),
                wcs=wcs, mode="strict",
            )
    except Exception:
        return None, "cutout-failed"

    hr_resampled = _resample_hlsp_to_hr(
        np.asarray(cutout.data, dtype=np.float32),
    )
    Hh, Wh = hr_resampled.shape
    H_hr = _WORKER_IMAGE_SIZE
    if Hh < H_hr or Wh < H_hr:
        # Cutout smaller than the HR target — drop. We don't want
        # ``crop_array``'s zero-padding behaviour here because a partial
        # cutout would leak zero-valued sky into the training pair.
        return None, "cutout-too-small"
    hr_clean_rate = MultiBandSkyImage.crop_array(hr_resampled, H_hr)

    # No sky-offset subtraction here. HLSP COSMOS F814W mosaics are
    # already sky-subtracted by HAP, so the cutout median is ≈ 0 by
    # design. The Euclid forward model also does its own symmetric sky
    # add/subtract in ``apply_band_noise`` — pre-subtracting on the HST
    # side is redundant.

    hr_cube = _hst_to_euclid_hr_cube(hr_clean_rate, _WORKER_TYPICAL_RATIOS)
    if hr_cube is None:
        return None, "hst-cube-allnan"

    # Bright-stamp rejection: measure S_max from the fully-built HR
    # cube but BEFORE the expensive forward modelling. A rejected
    # stamp would only produce ringing-dominated dirty pixels.
    reject, _diag = _is_stamp_too_bright(
        hr_cube,
        diff_kernel=_WORKER_KERNEL,
        sigma_lr_band=_WORKER_SIGMA_LR,
        max_relative_noise=_WORKER_MAX_RELATIVE_NOISE,
    )
    if reject:
        return None, "rejected-bright"

    # Per-task RNG so noise is reproducible across runs at the same seed
    # but independent across galaxies in a single run.
    rng = np.random.default_rng(int(seed))
    lr_cube = _make_pair(
        hr_cube,
        diff_kernel=_WORKER_KERNEL,
        rng=rng,
    )
    return (catalog_idx, hr_cube, lr_cube), None


def _hst_to_euclid_hr_cube(
    hst_rate_per_hr_pix: np.ndarray,
    typical_band_ratios: np.ndarray,
) -> Optional[np.ndarray]:
    """HST F814W rate → 4-band Euclid HR cube (total electrons per HR pixel).

    Photometric chain — every source in the cutout keeps its real
    Euclid-magnitude brightness because we never normalise the cube:

    * **VIS**: ``hst_rate × rate_ratio_VIS/HST × t_total_VIS``.
      The rate ratio compensates for the slight sensitivity difference
      between HST F814W (ZP ≈ 25.94 e⁻/s) and Euclid VIS (ZP = 25.50);
      at the same AB magnitude Euclid VIS gives ~0.67× the HST rate.
      Multiplying by VIS stack time (2260 s) converts rate → total
      electrons over the Euclid stack.
    * **NISP[k]**: ``VIS_HR × typical_band_ratios[k]``. The catalog-
      derived median ratio of ``e_band / e_VIS`` per source already
      bundles colour, ZP, and integration-time differences into one
      number per band — so each NISP channel is the VIS HR scaled by
      a global "typical galaxy" colour. Per-source colours are lost
      (every star and galaxy in the cube share the same colour) but
      the absolute scale is correct, which is what matters for noise
      statistics during training.
    """
    if not np.isfinite(hst_rate_per_hr_pix).any():
        return None

    vis = Config.BAND_VIS
    rate_ratio_vis_hst = 10.0 ** (
        -0.4 * (Config.HST_ACS_F814W_AB_ZP_E_PER_S - vis.zeropoint_ab_e_per_s)
    )
    vis_hr_e_total = (
        hst_rate_per_hr_pix.astype(np.float32)
        * np.float32(rate_ratio_vis_hst * vis.t_total_s)
    )

    ratios = np.asarray(typical_band_ratios, dtype=np.float32)
    if ratios.shape != (Config.NUM_LR_CHANNELS,):
        raise ValueError(
            f"typical_band_ratios shape {ratios.shape} != "
            f"({Config.NUM_LR_CHANNELS},)"
        )
    # Broadcast: (H, W) × (4,) → (H, W, 4) without an explicit loop.
    return (vis_hr_e_total[:, :, None] * ratios[None, None, :]).astype(np.float32)


def main() -> int:
    args = parse_args()
    reporter = Reporter.from_env()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 64)
    print(f"  HST → Euclid TFRecord pair generation (analytic-A only)")
    print("=" * 64)
    print(f"  HLSP dir         = {args.hlsp_dir}")
    print(f"  kernel           = {args.kernel}")
    print(f"  output           = {args.output_dir}")
    print(f"  n_train          = {args.n_train}")
    print(f"  n_valid          = {args.n_valid}")
    print(f"  image_size       = {args.image_size} (HR pixels)")
    print(f"  max_rel_noise    = {args.max_relative_noise}")
    print(f"  oversample       = {args.oversample}")
    print(f"  dry run          = {args.dry_run}")
    print()

    t0 = time.time()
    H_hr = int(args.image_size)
    if H_hr % 2 != 0:
        reporter.error(f"image_size must be even (got {H_hr})")
        print(f"ERROR: image_size must be even (got {H_hr})")
        return 1
    # HR side in arcsec → HLSP-pixel side needed before resample to HR.
    hr_side_arcsec   = H_hr * Config.DEFAULT_PIXEL_SCALE      # e.g. 510 × 0.05 = 25.5"
    hlsp_side_pix    = int(np.ceil(hr_side_arcsec / 0.03))     # cutout size in HLSP pixels

    reporter.set_stage("indexing HLSP tiles")
    print(f"[1/4] indexing HLSP tiles ...")
    if not os.path.isdir(args.hlsp_dir):
        reporter.error(f"HLSP dir not found: {args.hlsp_dir}")
        print(f"ERROR: HLSP dir not found: {args.hlsp_dir}")
        print("       Run scripts/fasrc_download_hst_hlsp.py first.")
        return 1
    tiles = HLSPTileIndex(args.hlsp_dir)
    print(f"      {len(tiles.entries)} tiles indexed")

    reporter.set_stage("loading kernel + COSMOS catalog")
    print(f"[2/4] loading analytic kernel + COSMOS catalog ...")
    if not os.path.isfile(args.kernel):
        reporter.error(f"analytic kernel ({args.kernel}) does not exist.")
        print(f"ERROR: analytic kernel ({args.kernel}) does not exist. "
              "Run scripts/fasrc_compute_differential_kernel.py first.")
        return 1
    dk = DifferentialKernel.from_fits(args.kernel)
    print(f"      analytic kernel shape = {dk.data.shape} "
          f"DC gain = {dk.dc_gain:.4f}")
    sigma_lr_band = _euclid_vis_sigma_lr_per_pixel()
    print(f"      Euclid VIS σ_LR per pixel = {sigma_lr_band:.2f} e⁻")
    catalog = open_cosmos2025(max_mag=args.max_mag)
    print(f"      {len(catalog):,} catalog galaxies after quality cuts "
          f"(VIS mag ≤ {args.max_mag})")
    # Catalog-derived "typical galaxy" colours per NISP band — used to
    # paint NISP HR channels from the single-band HST cutout (which
    # only carries F814W ≈ VIS morphology). One scale factor per band,
    # applied per-pixel; per-source colours are lost but the absolute
    # NISP brightness is correct on average.
    typical_band_ratios = catalog.typical_band_electron_ratios()
    print(f"      typical e_band / e_VIS (median over catalog): "
          + ", ".join(
              f"{name}={typical_band_ratios[k]:.3g}"
              for k, name in enumerate(Config.LR_INPUT_BAND_NAMES)
          ))

    if args.dry_run:
        print("\nDRY RUN — would synthesise "
              f"{args.n_train + args.n_valid} pairs at {H_hr}² HR.")
        runtime = time.time() - t0
        print(f"\nRUNTIME_SECONDS={runtime:.1f}")
        return 0

    reporter.set_stage("selecting galaxies on HLSP coverage")
    print(f"[3/4] selecting galaxies that fall on the HLSP coverage ...")
    # Walk catalog row-by-row, keep galaxies whose RA/Dec lands on a tile.
    rng = np.random.default_rng()
    # Oversample the catalog so bright-stamp rejection has spares.
    target_total   = args.n_train + args.n_valid
    candidate_pool = min(len(catalog),
                         max(target_total,
                             int(np.ceil(args.oversample * target_total))))
    catalog_indices = np.arange(len(catalog))
    rng.shuffle(catalog_indices)
    catalog_indices = catalog_indices[:candidate_pool]
    print(f"      shuffled candidate pool = {candidate_pool:,} "
          f"(oversample = {args.oversample:g}× target {target_total})")

    n_workers = max(0, int(args.n_workers))
    reporter.set_stage(f"streaming pairs (pool size = {n_workers})")
    print(f"[4/4] streaming pairs to {args.output_dir} "
          f"(pool size = {n_workers}) ...")
    pairs_written          = 0
    pairs_attempted        = 0
    pairs_skipped_no_tile  = 0
    pairs_skipped_cutout   = 0
    pairs_skipped_bright   = 0
    pairs_skipped_other    = 0
    pairs_per_subset = {"train": args.n_train, "validate": args.n_valid}

    summary: Dict[str, Any] = {"subsets": {}}
    catalog_iter = iter(catalog_indices)
    base_seed = int(rng.integers(0, 2**31))

    def _next_task() -> Optional[Tuple]:
        """Pull the next candidate sky chunk off the shuffled catalog,
        skipping centres whose RA/Dec doesn't land on any HLSP tile.

        Tile lookup runs in the *main* process because the index is
        ~MB-scale shared state; cheaper than re-loading it per worker.
        The catalog only selects the centre RA/Dec — its per-band
        catalog fluxes are no longer used (HST's native photometry is
        what scales the cube). Returns the task tuple ready for
        ``_process_one_galaxy`` or ``None`` if the catalog is exhausted.
        """
        nonlocal pairs_skipped_no_tile
        for i in catalog_iter:
            ra  = float(catalog.ra_deg[i])
            dec = float(catalog.dec_deg[i])
            tile_path = tiles.find_tile(ra, dec)
            if tile_path is None:
                pairs_skipped_no_tile += 1
                continue
            seed = (base_seed * 1_000_003 + int(i)) & 0x7FFFFFFF
            return (int(i), ra, dec, tile_path, hlsp_side_pix, seed)
        return None

    def _classify_skip(reason: Optional[str]) -> None:
        nonlocal pairs_skipped_cutout, pairs_skipped_bright, pairs_skipped_other
        if reason == "rejected-bright":
            pairs_skipped_bright += 1
        elif reason in ("cutout-too-small", "cutout-failed", "no-image-hdu"):
            pairs_skipped_cutout += 1
        else:
            pairs_skipped_other += 1

    def _write_result(result, subset_writers, subset_done):
        """Wrap one worker result as MultiBandSkyImages and write."""
        catalog_idx, hr_cube, lr_cube = result
        cw, dw, hw = subset_writers
        try:
            cosmos_id = int(catalog.catalog_id[catalog_idx])
        except (IndexError, TypeError, ValueError):
            cosmos_id = catalog_idx
        meta = {"source": "hst_hlsp", "cosmos_id": cosmos_id}
        clean_img = MultiBandSkyImage(
            data=hr_cube, pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
            band_names=Config.LR_INPUT_BAND_NAMES, is_clean=True,
            metadata=meta,
        )
        dirty_img = MultiBandSkyImage(
            data=lr_cube,
            pixel_scale_arcsec=Config.VIS_PIXEL_SCALE_ARCSEC,
            band_names=Config.LR_INPUT_BAND_NAMES, is_clean=False,
            metadata=meta,
        )
        hr_vis_only = MultiBandSkyImage(
            data=hr_cube[..., :1].copy(),
            pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
            band_names=("VIS",), is_clean=True,
            metadata=meta,
        )
        cw.write(clean_img,   index=subset_done)
        dw.write(dirty_img,   index=subset_done)
        hw.write(hr_vis_only, index=subset_done)

    init_args = (
        args.kernel,
        H_hr,
        typical_band_ratios,
        sigma_lr_band,
        float(args.max_relative_noise),
    )

    if n_workers == 0:
        # Single-process fallback — useful for debugging / profiling.
        # Calls _init_worker in-process so the worker fn sees the kernel.
        _init_worker(*init_args)

    for subset, target_n in pairs_per_subset.items():
        sub_done = 0
        with open_multiband_writer(
            f"clean_{subset}", records_dir=args.output_dir,
        ) as cw, open_multiband_writer(
            f"dirty_{subset}", records_dir=args.output_dir,
        ) as dw, open_multiband_writer(
            f"hr_{subset}", records_dir=args.output_dir,
        ) as hw:
            writers = (cw, dw, hw)

            if n_workers == 0:
                # No pool — sequential same as before, but routed
                # through the same _process_one_galaxy helper so the
                # two code paths can't diverge.
                while sub_done < target_n:
                    task = _next_task()
                    if task is None:
                        break
                    pairs_attempted += 1
                    result, reason = _process_one_galaxy(task)
                    if result is None:
                        _classify_skip(reason)
                        continue
                    _write_result(result, writers, sub_done)
                    sub_done += 1
                    pairs_written += 1
                    reporter.set_step(sub_done, target_n, subset)
                    if sub_done <= 5 or sub_done % 10 == 0:
                        print(f"      {subset}: {sub_done}/{target_n} written",
                              flush=True)
            else:
                # ProcessPoolExecutor: keep ~2*n_workers in flight so
                # workers stay busy while one finishes + the main
                # process writes. Pool initialiser loads the kernel
                # once per process.

                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_init_worker,
                    initargs=init_args,
                ) as pool:
                    in_flight: set = set()
                    target_in_flight = max(n_workers * 2, 4)

                    # Prime the pool with the first batch.
                    while len(in_flight) < target_in_flight:
                        task = _next_task()
                        if task is None:
                            break
                        pairs_attempted += 1
                        in_flight.add(pool.submit(_process_one_galaxy, task))

                    while in_flight and sub_done < target_n:
                        done, in_flight = wait(
                            in_flight, return_when=FIRST_COMPLETED,
                        )
                        for fut in done:
                            try:
                                result, reason = fut.result()
                            except Exception as e:
                                reporter.warn(
                                    f"worker exception: {type(e).__name__}: {e}"
                                )
                                print(f"      worker exception: "
                                      f"{type(e).__name__}: {e}")
                                pairs_skipped_other += 1
                                result, reason = None, "exception"
                            if result is None:
                                _classify_skip(reason)
                            else:
                                _write_result(result, writers, sub_done)
                                sub_done += 1
                                pairs_written += 1
                                reporter.set_step(sub_done, target_n, subset)
                                if sub_done <= 5 or sub_done % 10 == 0:
                                    print(f"      {subset}: {sub_done}/"
                                          f"{target_n} written", flush=True)
                            if sub_done >= target_n:
                                break
                            # Replenish so the pool stays warm.
                            task = _next_task()
                            if task is not None:
                                pairs_attempted += 1
                                in_flight.add(
                                    pool.submit(_process_one_galaxy, task)
                                )

                    # Cancel any over-submitted tasks once we have
                    # enough successes — saves a few seconds on a
                    # large run.
                    for fut in in_flight:
                        fut.cancel()

        summary["subsets"][subset] = {
            "written": sub_done,
            "target":  target_n,
        }
        print(f"      ✓ {subset}: {sub_done} pairs written")

    runtime = time.time() - t0
    summary["pairs_written"]         = pairs_written
    summary["pairs_attempted"]       = pairs_attempted
    summary["pairs_skipped_no_tile"] = pairs_skipped_no_tile
    summary["pairs_skipped_cutout"]  = pairs_skipped_cutout
    summary["pairs_skipped_bright"]  = pairs_skipped_bright
    summary["pairs_skipped_other"]   = pairs_skipped_other
    summary["max_relative_noise"]    = float(args.max_relative_noise)
    summary["sigma_lr_band"]         = float(sigma_lr_band)
    summary["elapsed_s"]             = round(runtime, 1)
    with open(os.path.join(args.output_dir, "generation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"  wrote      {pairs_written} pairs")
    print(f"  attempted  {pairs_attempted}")
    print(f"  skipped    no-tile={pairs_skipped_no_tile} "
          f"cutout={pairs_skipped_cutout} "
          f"bright={pairs_skipped_bright} "
          f"other={pairs_skipped_other}")
    reporter.set_stage(
        f"hst-tfrecords done: written={pairs_written} "
        f"attempted={pairs_attempted} "
        f"rejected_bright={pairs_skipped_bright} "
        f"skipped_cutout={pairs_skipped_cutout} "
        f"skipped_no_tile={pairs_skipped_no_tile}"
    )
    print(f"\nRUNTIME_SECONDS={runtime:.1f}")
    print(f"PAIRS_WRITTEN={pairs_written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
