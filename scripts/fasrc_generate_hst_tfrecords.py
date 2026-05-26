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
  4. Apply the pre-computed differential kernel A to get the Euclid-PSF
     view, sum-rebin to LR scale (0.10″/pix), add per-band Euclid noise.
  5. Write the (HR, LR) pair into the standard ``MultiBandSkyImage``
     TFRecord layout — same schema the synthetic generator uses.

Output records land under ``$DATA_DIR/images/records_v2_hst/`` so the
trainer can mix them with the existing synthetic records via the
``hst_fraction`` knob.
"""

from __future__ import annotations

import os

# Force-hide GPUs BEFORE TF is imported, even if SLURM allocated one.
#
# Why this script is CPU-only by design:
#
#  * Heavy work = scipy.signal.fftconvolve (FFT, CPU only) over a 511²
#    baseline kernel × 4 bands per cutout. No GPU path exists.
#  * The trainable CNN residual is only ~5 k params — its forward
#    pass takes ~1 ms on CPU; a GPU offers no measurable speedup.
#  * Parallelism is process-pool-based (fork), and ``fork`` after the
#    parent has touched CUDA puts every child in an invalid CUDA
#    state. The first TF call in any worker then aborts with
#    ``CUDA_ERROR_NOT_INITIALIZED`` and ``ProcessPoolExecutor`` raises
#    ``BrokenProcessPool``. (SLURM job 15889067, May 2026: 8 child
#    crashes within seconds of pool startup.)
#
# ``setdefault`` would be a no-op when SLURM has already set
# ``CUDA_VISIBLE_DEVICES`` (which it does on every ``--gres=gpu:N``
# job), so we use a hard assignment to override SLURM's value. If you
# explicitly want this script to use a GPU anyway (e.g. you've also
# disabled the worker pool with ``--n-workers 0``), set the env var
# back yourself in the sbatch script *after* the Python launch:
#     CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS python fasrc_generate_hst_tfrecords.py ...
# But note: a GPU does not speed this script up. For *training* the
# transition model, request and keep a GPU normally — that's a
# different script (``fasrc_train_transition_model.py``).
_prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if _prev_cuda:
    print(f"[fasrc_generate_hst_tfrecords] SLURM set "
          f"CUDA_VISIBLE_DEVICES={_prev_cuda!r}; overriding to '' because "
          f"this pipeline is CPU-only by design (FFT + process pool). "
          f"Submit without --gres=gpu next time to free the device "
          f"for someone else.", flush=True)

# Cap per-process thread counts BEFORE numpy/scipy/TF are imported.
# With a 16-worker pool on 16 SLURM CPUs, the default lets each
# worker spawn 16 intra-op threads → 256 OS threads contending for
# 16 cores → all per-cutout work crawls. Pinning to 1 thread per
# process restores 1:1 worker-to-core mapping. setdefault honours an
# explicit override from the sbatch script if the caller really wants
# multi-threaded BLAS.
os.environ.setdefault("OMP_NUM_THREADS",         "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",    "1")
os.environ.setdefault("MKL_NUM_THREADS",         "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS",  "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS",  "1")

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
from euclid_polish.sky.cosmos2025 import open_cosmos2025
from euclid_polish.sky.differential_kernel import DifferentialKernel
from euclid_polish.sky.multiband_forward import apply_band_noise
from euclid_polish.sky.tfrecord import open_multiband_writer
from euclid_polish.sky.types import MultiBandSkyImage
from euclid_polish.training.transition_model import (
    HSTtoEuclidTransition,
    apply_transition_numpy,
    load_model_weights,
)


HLSP_DIR     = os.path.join(Config.DATA_DIR, "hst_hlsp")
KERNEL_PATH  = os.path.join(Config.DATA_DIR, "hst_psf", "diff_kernel_VIS.fits")
TRANSITION_MODEL_PATH = os.path.join(
    Config.DATA_DIR, "hst_psf", "transition_model.weights.h5",
)
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
                   help="Analytic Wiener differential kernel FITS. "
                        "Used only when --transition-model is not provided "
                        "(or its weights file is missing). Kept as a "
                        "fallback so the old pipeline still works.")
    p.add_argument("--transition-model", default=TRANSITION_MODEL_PATH,
                   help="Trained CNN transition model "
                        "(``transition_model.weights.h5``). When present, "
                        "replaces the analytic kernel as the HST→Euclid "
                        "PSF transition. Train via "
                        "scripts/fasrc_train_transition_model.py. Pass "
                        "an empty string or non-existent path to force "
                        "the analytic-kernel fallback.")
    p.add_argument("--transition-channels", type=int, default=12,
                   help="Hidden channel width C of the transition model. "
                        "Must match the value used at training time.")
    p.add_argument("--transition-inner-layers", type=int, default=3,
                   help="Number of inner Conv(C→C) layers. Must match "
                        "the value used at training time.")
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


def _apply_transition(
    hr_band: np.ndarray,
    *,
    transition_model,
    diff_kernel: Optional[np.ndarray],
) -> np.ndarray:
    """Map an HR band image from HST-PSF space to Euclid-PSF space.

    Picks the trained CNN (``transition_model``) when available;
    otherwise falls back to FFT-convolution with the analytic
    differential kernel ``diff_kernel``. Exactly one of the two must
    be non-None.

    Returns an array of the same shape as ``hr_band`` (no resampling
    here — sum-rebin is applied separately downstream).
    """
    if transition_model is not None:
        return apply_transition_numpy(transition_model, hr_band)
    if diff_kernel is None:
        raise RuntimeError(
            "neither transition_model nor diff_kernel is set — "
            "_init_worker must populate one of them"
        )
    return scipy_signal.fftconvolve(hr_band, diff_kernel, mode="same")


def _make_pair(
    hr_cutout_4ch: np.ndarray,            # (H, W, 4) HR clean
    *,
    transition_model,
    diff_kernel: Optional[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Forward-model the HR cube to a 4-band LR cube at 0.10″/pix.

    For each band: (transition or A ⊛ ·) → sum-rebin ×2
    → Poisson + read + sky + dark.

    The transition step is what makes the chain photometrically
    Euclid-equivalent while accounting for the HST-baked-in PSF:

      * trained CNN (``transition_model`` ≠ None): preferred path,
        well-behaved on stars/galaxies + free of the Wiener inverse's
        checkerboard / ring artefacts.
      * analytic Wiener kernel (``diff_kernel`` ≠ None): fallback
        when the CNN weights file isn't available yet.
    """

    H, W, C = hr_cutout_4ch.shape
    assert H % 2 == 0 and W % 2 == 0
    lr_cube = np.zeros((H // 2, W // 2, C), dtype=np.float32)
    for k, band_name in enumerate(Config.LR_INPUT_BAND_NAMES):
        band = Config.get_band(band_name)
        convolved = _apply_transition(
            hr_cutout_4ch[..., k],
            transition_model=transition_model,
            diff_kernel=diff_kernel,
        )
        # Sum-rebin ×2 (photometric: conserves total electrons per LR pixel).
        rebinned = convolved.reshape(H // 2, 2, W // 2, 2).sum(axis=(1, 3))
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
_WORKER_KERNEL:           Optional[np.ndarray] = None
_WORKER_TRANSITION_MODEL                       = None    # tf.keras.Model | None
_WORKER_IMAGE_SIZE:       int                  = 0
_WORKER_TYPICAL_RATIOS:   Optional[np.ndarray] = None


def _load_summary_for_weights(weights_path: str) -> Dict[str, Any]:
    """Read the ``transition_model_summary.json`` next to a weights file.

    The summary records the exact hyperparameters used at training time
    (channels, n_inner_layers, kernel_size, baseline kernel path, etc.).
    Without it, the worker has to guess — and guessing wrong means
    Keras' ``load_weights`` fails with ``"expected N variables, got M"``
    because the reconstructed model has different shape than what's in
    the file.

    Returns an empty dict if the summary doesn't exist (legacy models
    pre-summary). Caller must fall back to CLI defaults in that case.
    """
    summary_path = weights_path[:-len(".weights.h5")] + "_summary.json" \
        if weights_path.endswith(".weights.h5") else weights_path + ".summary.json"
    if not os.path.isfile(summary_path):
        # Also try the legacy name produced by the training script.
        dirpath = os.path.dirname(weights_path)
        alt = os.path.join(dirpath, "transition_model_summary.json")
        if os.path.isfile(alt):
            summary_path = alt
        else:
            return {}
    try:
        with open(summary_path) as f:
            return json.load(f)
    except Exception:
        return {}


def _init_worker(
    kernel_path: str,
    transition_model_path: Optional[str],
    transition_channels: int,
    transition_inner_layers: int,
    image_size: int,
    typical_band_ratios: np.ndarray,
) -> None:
    """Called once per worker process at pool startup.

    Loads the HST→Euclid transition operator and pins the per-band
    electron ratios into module-level globals so each worker pays the
    I/O + parse cost exactly once — not per-galaxy. Pickling the
    kernel/model through every task argument would be ~1 MB per
    submission × 6400 galaxies = 6 GB of pickle churn for nothing.

    Selection rules:

    * If ``transition_model_path`` is provided and the weights file
      exists, the trained CNN is loaded and is the operator the worker
      uses. **Hyperparameters come from the
      ``transition_model_summary.json`` sidecar** when present (only
      the CLI fallbacks ``transition_channels`` /
      ``transition_inner_layers`` are used if the summary is missing).
      The summary also tells us whether the model has an analytic
      baseline pinned as a non-trainable layer; if so we load the
      same baseline FITS so the layer shapes match the saved weights.
    * Otherwise the analytic Wiener kernel is loaded from
      ``kernel_path``. This is the legacy path; we keep it so workflows
      that haven't run the CNN-training step yet still produce
      TFRecords.
    """
    global _WORKER_KERNEL, _WORKER_TRANSITION_MODEL
    global _WORKER_IMAGE_SIZE, _WORKER_TYPICAL_RATIOS
    _WORKER_IMAGE_SIZE     = int(image_size)
    _WORKER_TYPICAL_RATIOS = np.asarray(typical_band_ratios, dtype=np.float32)

    use_cnn = (
        transition_model_path
        and os.path.isfile(transition_model_path)
    )
    if use_cnn:
        # Read training-time hyperparameters from the summary sidecar
        # so the model is reconstructed with the right shape.
        summary = _load_summary_for_weights(transition_model_path)
        channels       = int(summary.get("channels",       transition_channels))
        n_inner_layers = int(summary.get("n_inner_layers", transition_inner_layers))
        kernel_size    = int(summary.get("kernel_size",    3))
        baseline_path  = str(summary.get("analytic_baseline_kernel", ""))
        baseline_crop  = int(summary.get("baseline_crop",  0))

        # Load + crop the analytic baseline kernel if the model used one.
        # Resolve it relative to the diff-kernel FITS path if the saved
        # path is a FASRC-style absolute path that doesn't exist locally.
        baseline_kernel = None
        if baseline_path:
            candidates = [baseline_path, kernel_path]
            for cand in candidates:
                if cand and os.path.isfile(cand):
                    with fits.open(cand) as hdul:
                        full = np.asarray(hdul[0].data, dtype=np.float32)
                    if baseline_crop > 0 and baseline_crop < full.shape[0]:
                        side = baseline_crop | 1
                        i0 = (full.shape[0] - side) // 2
                        baseline_kernel = full[i0:i0 + side, i0:i0 + side]
                    else:
                        baseline_kernel = full
                    break
            if baseline_kernel is None:
                # Summary says baseline was used but we can't find the
                # kernel file — fail loudly. Loading without the baseline
                # would give the "expected 0 variables, received 1"
                # Keras error.
                raise RuntimeError(
                    f"transition model summary records "
                    f"analytic_baseline_kernel={baseline_path!r} but no "
                    f"matching FITS was found locally. Tried: {candidates}. "
                    f"Re-sync the kernel or remove the baseline reference."
                )

        model = HSTtoEuclidTransition(
            channels=channels,
            n_inner_layers=n_inner_layers,
            kernel_size=kernel_size,
            baseline_kernel=baseline_kernel,
        )
        load_model_weights(model, transition_model_path)
        _WORKER_TRANSITION_MODEL = model
        _WORKER_KERNEL = None
    else:
        _WORKER_KERNEL           = DifferentialKernel.from_fits(kernel_path).data
        _WORKER_TRANSITION_MODEL = None

    # Heartbeat so a future stall in worker init (e.g. another slow
    # tf.nn.conv2d during build) is visible from the SLURM log instead
    # of looking identical to a healthy "just hasn't finished cutout #1
    # yet" state.
    print(f"      [worker {os.getpid()}] init complete "
          f"(use_cnn={bool(use_cnn)})", flush=True)


def _process_one_galaxy(
    task: Tuple[int, float, float, str, int, int],
) -> Optional[Tuple[int, np.ndarray, np.ndarray]]:
    """Cut + resample + forward-model one HST chunk of sky.

    Pure function — no shared state beyond the module globals set by
    ``_init_worker``. The kernel is the only non-trivial state and is
    immutable across calls. The RNG is seeded per task so two workers
    processing the same catalog_idx would produce identical output.

    Note: ``catalog_idx`` only selects *which RA/Dec to centre on* —
    its per-band catalog fluxes are no longer used. The HST cutout's
    own photometry sets the brightness of every source in the field.

    Returns ``(catalog_idx, hr_cube, lr_cube)`` on success, ``None``
    when the chunk can't be cut (off-tile, malformed FITS, cutout
    smaller than the target HR size, or HST data all-NaN).
    """
    catalog_idx, ra, dec, tile_path, hlsp_side_pix, seed = task
    if _WORKER_TYPICAL_RATIOS is None or (
        _WORKER_KERNEL is None and _WORKER_TRANSITION_MODEL is None
    ):
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
                return None
            wcs = WCS(sci.header)
            cutout = Cutout2D(
                sci.data,
                SkyCoord(ra * u.deg, dec * u.deg),
                size=(hlsp_side_pix, hlsp_side_pix),
                wcs=wcs, mode="strict",
            )
    except Exception:
        return None

    hr_resampled = _resample_hlsp_to_hr(
        np.asarray(cutout.data, dtype=np.float32),
    )
    Hh, Wh = hr_resampled.shape
    H_hr = _WORKER_IMAGE_SIZE
    if Hh < H_hr or Wh < H_hr:
        return None
    i0 = (Hh - H_hr) // 2
    j0 = (Wh - H_hr) // 2
    hr_clean_rate = hr_resampled[i0:i0 + H_hr, j0:j0 + H_hr]

    # No sky-offset subtraction here. HLSP COSMOS F814W mosaics are
    # already sky-subtracted by HAP, so the cutout median is ≈ 0 by
    # design. A multi-source 25″ chunk also has real galaxies leaking
    # into any outer annulus we'd use as a "sky" reference, so the
    # local-median estimate is biased upward and subtracting it
    # uniformly down-shifts every real source. Finally, the Euclid
    # forward model already does its own symmetric sky add/subtract
    # in ``apply_band_noise`` — pre-subtracting on the HST side is
    # redundant.

    hr_cube = _hst_to_euclid_hr_cube(hr_clean_rate, _WORKER_TYPICAL_RATIOS)
    if hr_cube is None:
        return None

    # Per-task RNG so noise is reproducible across runs at the same seed
    # but independent across galaxies in a single run.
    rng = np.random.default_rng(int(seed))
    lr_cube = _make_pair(
        hr_cube,
        transition_model=_WORKER_TRANSITION_MODEL,
        diff_kernel=_WORKER_KERNEL,
        rng=rng,
    )
    return catalog_idx, hr_cube, lr_cube


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

    The catalog's ``total_flux_e`` for the central galaxy is **not used
    here** — the old chain multiplied the (unit-normalised) cube by one
    galaxy's catalog flux, allocating a single source's electron budget
    across every visible source in the cutout. Real HST cutouts contain
    many galaxies; that's the whole point.
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
    os.makedirs(args.output_dir, exist_ok=True)

    use_cnn = bool(
        args.transition_model and os.path.isfile(args.transition_model)
    )

    print("=" * 64)
    print(f"  HST → Euclid TFRecord pair generation")
    print("=" * 64)
    print(f"  HLSP dir         = {args.hlsp_dir}")
    print(f"  transition model = {args.transition_model} "
          f"{'(found)' if use_cnn else '(missing — falling back to kernel)'}")
    if not use_cnn:
        print(f"  fallback kernel  = {args.kernel}")
    print(f"  output           = {args.output_dir}")
    print(f"  n_train          = {args.n_train}")
    print(f"  n_valid          = {args.n_valid}")
    print(f"  image_size       = {args.image_size} (HR pixels)")
    print(f"  dry run          = {args.dry_run}")
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

    print(f"[2/4] loading transition operator + COSMOS catalog ...")
    if not use_cnn and not os.path.isfile(args.kernel):
        print(f"ERROR: neither transition model ({args.transition_model}) "
              f"nor analytic kernel ({args.kernel}) exists. Run either "
              "scripts/fasrc_train_transition_model.py or "
              "scripts/fasrc_compute_differential_kernel.py first.")
        return 1
    if use_cnn:
        print(f"      using trained CNN transition model "
              f"(channels={args.transition_channels}, "
              f"inner_layers={args.transition_inner_layers})")
    else:
        dk = DifferentialKernel.from_fits(args.kernel)
        print(f"      analytic kernel shape = {dk.data.shape} "
              f"DC gain = {dk.dc_gain:.4f}")
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

    print(f"[3/4] selecting galaxies that fall on the HLSP coverage ...")
    # Walk catalog row-by-row, keep galaxies whose RA/Dec lands on a tile.
    rng = np.random.default_rng()
    catalog_indices = np.arange(len(catalog))
    rng.shuffle(catalog_indices)
    target_total = args.n_train + args.n_valid


    n_workers = max(0, int(args.n_workers))
    print(f"[4/4] streaming pairs to {args.output_dir} "
          f"(pool size = {n_workers}) ...")
    pairs_written = 0
    pairs_skipped = 0
    pairs_per_subset = {"train": args.n_train, "validate": args.n_valid}

    summary = {"subsets": {}}
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
        nonlocal pairs_skipped
        for i in catalog_iter:
            ra  = float(catalog.ra_deg[i])
            dec = float(catalog.dec_deg[i])
            tile_path = tiles.find_tile(ra, dec)
            if tile_path is None:
                pairs_skipped += 1
                continue
            seed = (base_seed * 1_000_003 + int(i)) & 0x7FFFFFFF
            return (int(i), ra, dec, tile_path, hlsp_side_pix, seed)
        return None

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

    transition_model_path_arg = (
        args.transition_model if use_cnn else ""
    )
    init_args = (
        args.kernel,
        transition_model_path_arg,
        int(args.transition_channels),
        int(args.transition_inner_layers),
        H_hr,
        typical_band_ratios,
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
                    result = _process_one_galaxy(task)
                    if result is None:
                        pairs_skipped += 1
                        continue
                    _write_result(result, writers, sub_done)
                    sub_done += 1
                    pairs_written += 1
                    if sub_done <= 5 or sub_done % 10 == 0:
                        print(f"      {subset}: {sub_done}/{target_n} written",
                              flush=True)
            else:
                # ProcessPoolExecutor: keep ~2*n_workers in flight so
                # workers stay busy while one finishes + the main
                # process writes. Pool initialiser loads the kernel
                # once per process.
                from concurrent.futures import (
                    FIRST_COMPLETED, ProcessPoolExecutor, wait,
                )

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
                        in_flight.add(pool.submit(_process_one_galaxy, task))

                    while in_flight and sub_done < target_n:
                        done, in_flight = wait(
                            in_flight, return_when=FIRST_COMPLETED,
                        )
                        for fut in done:
                            try:
                                result = fut.result()
                            except Exception as e:
                                print(f"      worker exception: "
                                      f"{type(e).__name__}: {e}")
                                pairs_skipped += 1
                                result = None
                            if result is None:
                                pairs_skipped += 1
                            else:
                                _write_result(result, writers, sub_done)
                                sub_done += 1
                                pairs_written += 1
                                if sub_done <= 5 or sub_done % 10 == 0:
                                    print(f"      {subset}: {sub_done}/"
                                          f"{target_n} written", flush=True)
                            if sub_done >= target_n:
                                break
                            # Replenish so the pool stays warm.
                            task = _next_task()
                            if task is not None:
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
