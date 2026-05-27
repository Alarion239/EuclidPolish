#!/usr/bin/env python
"""Train the HST → Euclid PSF-transition model ``A_θ``.

Loads the transition-pair TFRecords produced by
``scripts/fasrc_generate_transition_pairs.py`` and trains a tiny CNN
(~5k params) to map ``clean ⊛ PSF_HST → clean ⊛ PSF_Euclid`` with L1
loss in linear electron units (no asinh stretch — image-formation
training is best done in the natural unit space).

The trained model replaces the analytic differential kernel
``diff_kernel_VIS.fits``: at HST-cutout serving time the network maps
HR scenes from HST-PSF space to Euclid-PSF space before sum-rebin to LR.

Output: weights file at ``$DATA_DIR/hst_psf/transition_model.weights.h5``,
plus a JSON sidecar with training metadata (param count, final loss,
hyperparameters).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from astropy.io import fits

import signal as _signal

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.sky.tfrecord import (
    parse_record_graph_v2, read_multiband_skyimages, tfrecord_path,
)
from euclid_polish.training.transition_augmentations import (
    add_hlsp_noise, apply_linear_combo_augmentation, build_star_pair_dataset,
    make_psf_identity_pair, sample_hlsp_noise_params, sum_rebin_2d,
)
from euclid_polish.training.transition_model import (
    HSTDenoiser, HSTtoEuclidTransition, load_denoiser_weights,
    save_model_weights, total_parameter_count,
)
from scripts.fasrc_generate_transition_pairs import (
    _load_euclid_vis_psf_on_hr, _load_hst_psf_on_hr,
)


PAIRS_DIR     = os.path.join(Config.DATA_DIR, "images", "records_transition")
HST_PSF_PATH  = os.path.join(Config.DATA_DIR, "hst_psf", "F814W.fits")
DEFAULT_MODEL = os.path.join(Config.DATA_DIR, "hst_psf",
                             "transition_model.weights.h5")
SUMMARY_JSON  = os.path.join(Config.DATA_DIR, "hst_psf",
                             "transition_model_summary.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs-dir", default=PAIRS_DIR,
                   help="Transition-pair TFRecords (output of "
                        "fasrc_generate_transition_pairs.py).")
    p.add_argument("--output", default=DEFAULT_MODEL,
                   help="Weights output path (.weights.h5).")
    p.add_argument("--summary-output", default=SUMMARY_JSON,
                   help="JSON sidecar with training summary stats.")
    p.add_argument("--hst-psf", default=HST_PSF_PATH,
                   help="HST F814W ePSF FITS — used for the diagnostic "
                        "'PSF-to-PSF identity' check at end of training.")
    p.add_argument("--steps", type=int, default=20_000,
                   help="Total training steps.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Examples per gradient step. Pairs are 256² float32 "
                        "→ ~1 MB / example; 8 fits comfortably on a 16 GB "
                        "GPU and is more than enough for a 5k-param model.")
    p.add_argument("--learning-rate", type=float, default=2e-3,
                   help="Adam learning rate. The model is tiny so a "
                        "relatively high LR converges in O(10k) steps.")
    p.add_argument("--validate-every", type=int, default=500,
                   help="Run validation + log every N steps.")
    p.add_argument("--channels", type=int, default=12,
                   help="Hidden channel width C. With defaults (K=3, "
                        "n_inner=3): C=12 → 4153 params (under 5k cap).")
    p.add_argument("--n-inner-layers", type=int, default=3,
                   help="Number of inner ``Conv(C→C)`` layers. Default 3 "
                        "→ 5 total convs. Set to 0 for a 2-layer model "
                        "(``Conv(1→C) → Conv(C→1)``), useful when paired "
                        "with a large kernel for a big receptive field "
                        "at low param count.")
    p.add_argument("--kernel-size", type=int, default=3,
                   help="Side length of every Conv2D kernel (odd). "
                        "Default 3 with 5 layers gives RF=11 px. For a "
                        "bigger receptive field at lower depth, pair "
                        "K with n_inner_layers=0:\n"
                        "  K=21, C=5  → RF=41, ~4400 params\n"
                        "  K=31, C=2  → RF=61, ~3800 params\n"
                        "  K=41, C=1  → RF=81, ~3400 params\n"
                        "Param count = 2·K²·C + n_inner·K²·C² (no biases).")
    p.add_argument("--max-params", type=int, default=5000,
                   help="Hard upper bound on TRAINABLE parameter count. "
                        "The non-trainable analytic baseline kernel (if "
                        "any) does NOT count against this. Default 5000 "
                        "enforces the tiny-model contract; raise to "
                        "e.g. 50000 to experiment with wider channels.")
    p.add_argument("--analytic-baseline-kernel", type=str, default="",
                   help="Optional path to ``diff_kernel_VIS.fits`` (the "
                        "analytic Wiener differential kernel). When set, "
                        "the model becomes A_θ(x) = (1+f)(A_analytic⊛x) "
                        "— the analytic kernel is applied first, then "
                        "the CNN ``f`` learns to clean up its known "
                        "Wiener artifacts (checkerboard, rings, "
                        "spike-mismatch residuals). At init A_θ ≈ "
                        "A_analytic so training starts near the right "
                        "answer. Empty (default) uses the identity "
                        "baseline A_θ(x) = x + f(x).")
    p.add_argument("--baseline-crop", type=int, default=0,
                   help="Optional centre-crop of the analytic baseline "
                        "kernel before pinning it as a non-trainable "
                        "conv. Default 0 keeps the full kernel — "
                        "preferred so no kernel structure is lost. "
                        "Pass an odd integer < kernel_side to enable "
                        "cropping (saves per-step compute at the cost "
                        "of dropping the wings).")
    p.add_argument("--weight-decay", type=float, default=1e-5,
                   help="L2 regularisation strength applied to conv "
                        "weights. Small — the residual structure already "
                        "discourages large updates.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for shuffling + initialiser.")
    p.add_argument("--rebin-factor", type=int, default=2,
                   help="Spatial sum-rebin applied to ``A_θ(input)`` "
                        "before comparing to target. Must match the "
                        "rebin used at pair-generation time (which "
                        "wrote the LR target). Default 2 matches "
                        "Euclid HR→LR. Set to 1 only if you generated "
                        "pairs with --rebin-factor=1 (legacy mode).")
    p.add_argument("--star-injection-fraction", type=float, default=0.2,
                   help="Fraction of training examples replaced by "
                        "synthetic star-field pairs (sum of PSF stamps at "
                        "random positions/amplitudes — input uses PSF_HST, "
                        "target uses PSF_Euclid). The synthetic clean-scene "
                        "generator is galaxy-dominated; stars are the "
                        "single most important sample for ``A_θ`` (its "
                        "core diagnostic is A_θ(PSF_HST) ≈ PSF_Euclid), so "
                        "we inject them explicitly. 0 disables.")
    p.add_argument("--max-stars-per-image", type=int, default=8,
                   help="Upper bound on stars per injected synthetic field. "
                        "Each field draws ``Uniform[1, max]`` stars; "
                        "amplitudes are uniform in [10, 1000] electrons.")
    p.add_argument("--linear-combo-fraction", type=float, default=0.3,
                   help="Per-example probability of emitting a linear "
                        "combination ``α·pair_A + β·pair_B`` instead of "
                        "the raw pair. The pair stays valid because "
                        "convolution is linear. ``α ~ U[0.3, 1.5]``, "
                        "``β ~ U[-0.5, 0.5]``. 0 disables.")
    p.add_argument("--add-psf-pair-to-validation",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Prepend a single (PSF_HST, PSF_Euclid) pair "
                        "to the validation set — both PSFs centred in "
                        "the validate-shard canvas. The pair is the "
                        "literal PSF identity case A_θ(PSF_HST) ≈ "
                        "PSF_Euclid that the model must satisfy. With "
                        "this on, the reported val_L1 is bounded below "
                        "by the error on that one case, so a model that "
                        "passes synthetic-galaxy validation but flunks "
                        "point sources can't slip past unnoticed. "
                        "Pass --no-add-psf-pair-to-validation to "
                        "disable.")
    # ── Two-stage chain (optional): freeze a pretrained denoiser before
    #    the deconvolver. See ``fasrc_train_hst_denoiser.py`` (Phase 1).
    p.add_argument("--frozen-denoiser", type=str, default="",
                   help="Optional path to a pretrained ``hst_denoiser."
                        "weights.h5`` (output of "
                        "fasrc_train_hst_denoiser.py). When set, the "
                        "forward pass becomes "
                        "``CNN_2((1+f)(A_analytic ⊛ CNN_1_frozen(x)))`` "
                        "— Phase 1 model is loaded, frozen (no "
                        "gradient updates), and prepended to the "
                        "deconvolver's forward graph. Pair this with "
                        "``--input-noise-sigma`` (or the explicit "
                        "α/σ ranges below) so CNN_1 sees the noise "
                        "distribution it was trained on.")
    p.add_argument("--frozen-denoiser-summary", type=str, default="",
                   help="Path to the denoiser's summary JSON (used to "
                        "reconstruct architecture). Defaults to the "
                        "weights path with ``.weights.h5`` replaced by "
                        "``_summary.json``.")
    # ── Noise augmentation on the input side ──
    # The deconvolver in Phase-2 mode needs the frozen denoiser to see
    # realistic input. Default 0 means no noise (legacy clean-train
    # behaviour); typical Phase-2 use sets the α/σ ranges to match
    # what Phase-1 was trained on.
    p.add_argument("--alpha-min", type=float, default=0.0,
                   help="HLSP shot-noise α lower bound (set both α and "
                        "σ_floor ranges to 0 to disable noise — the "
                        "legacy single-stage default).")
    p.add_argument("--alpha-max", type=float, default=0.0,
                   help="HLSP shot-noise α upper bound.")
    p.add_argument("--sigma-floor-min", type=float, default=0.0,
                   help="HLSP sky/read floor lower bound (e⁻).")
    p.add_argument("--sigma-floor-max", type=float, default=0.0,
                   help="HLSP sky/read floor upper bound (e⁻).")
    p.add_argument("--dry-run", action="store_true",
                   help="Build model + dataset, print sizes, then exit.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pair TFRecord dataset (input, target) → tf.data
# ---------------------------------------------------------------------------

def _probe_image_size(pairs_dir: str, subset: str = "train") -> int:
    """Spatial side length of the on-disk pair shards.

    Augmentations (star injection in particular) need to know the
    image grid the pair generator wrote, otherwise the synthetic-star
    stream would emit shape-mismatched tensors. Probe by reading one
    record. Assumes square.

    The most common failure here is "shard exists but has zero
    records" — symptom of step 3a (pair generation) being run with
    a ``--crop-size`` larger than the synthetic-scene side
    (``Config.DEFAULT_IMAGE_SIZE`` = 256 by default), which skips
    every scene and emits an empty file. Spell that out in the error
    so the user doesn't have to dig through the pair-gen log to find
    the actual cause.
    """
    path = tfrecord_path(pairs_dir, f"input_{subset}")
    if not os.path.exists(path):
        raise RuntimeError(
            f"transition pair shard not found: {path}\n"
            f"Run step 3a (generate transition-model training pairs) first."
        )
    if os.path.getsize(path) == 0:
        raise RuntimeError(
            f"transition pair shard is 0 bytes: {path}\n"
            f"The pair-generation job (step 3a) likely failed silently — "
            f"check its log."
        )
    records = read_multiband_skyimages(path, num_images=1)
    if not records:
        raise RuntimeError(
            f"transition pair shard has no records: {path}\n"
            f"\n"
            f"Most likely cause: step 3a was run with --crop-size larger "
            f"than the synthetic scene side. Default synthetic image_size "
            f"is 256 px (Config.DEFAULT_IMAGE_SIZE). Look for "
            f"'skipped N scenes smaller than crop' lines in the step 3a "
            f"log — if N == total scenes, re-run step 3a with a smaller "
            f"crop_size (try 256 or 128).\n"
        )
    H, W, _ = records[0].data.shape
    if H != W:
        raise RuntimeError(
            f"non-square pair shape {records[0].data.shape}; "
            "the augmentation pipeline assumes square fields"
        )
    return int(H)


def _make_pair_dataset(
    pairs_dir: str, subset: str, *, batch_size: int, seed: int, shuffle: bool,
    rebin_factor: int = 2,
    star_injection_fraction: float = 0.0,
    max_stars_per_image: int = 8,
    linear_combo_fraction: float = 0.0,
    psf_hst: Optional[np.ndarray] = None,
    psf_euclid: Optional[np.ndarray] = None,
    add_psf_pair_to_validation: bool = False,
):
    """Build a tf.data.Dataset of ``(input, target)`` batches.

    Each side is parsed independently from its own TFRecord file and the
    two streams are zipped — same trick the multi-band loader uses for
    ``(clean, dirty)`` pairs. The pairs are written in matched order by
    the generator script, so zipping preserves correspondence.

    Optional training-time augmentations (typically only enabled on the
    train split):

    * ``star_injection_fraction > 0`` — mix in synthetic star fields
      from :func:`build_star_pair_dataset` via ``sample_from_datasets``.
      Requires both PSFs to be passed in.
    * ``linear_combo_fraction > 0`` — apply the linear-combination
      augmentation from :func:`apply_linear_combo_augmentation`.

    Both fractions are independent: stars are mixed in *first*, then
    the combination augmentation operates on the joint distribution
    (so a "linear combination of two stars" is a valid output).
    """
    inp_path = tfrecord_path(pairs_dir, f"input_{subset}")
    tgt_path = tfrecord_path(pairs_dir, f"target_{subset}")
    for p in (inp_path, tgt_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"transition-pair TFRecord not found: {p}. "
                "Run scripts/fasrc_generate_transition_pairs.py first."
            )

    def _parse_one(raw):
        # Single-band VIS-only pairs → num_channels=1.
        return parse_record_graph_v2(raw, num_channels=1)

    ds_inp = tf.data.TFRecordDataset([inp_path]).map(
        _parse_one, num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds_tgt = tf.data.TFRecordDataset([tgt_path]).map(
        _parse_one, num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = tf.data.Dataset.zip((ds_inp, ds_tgt))
    if shuffle:
        ds = ds.shuffle(buffer_size=512, seed=seed,
                        reshuffle_each_iteration=True)
    ds = ds.repeat() if subset == "train" else ds

    # ── Validation: prepend the literal PSF identity pair ──
    # This makes val_L1 explicitly include the model's behavior on the
    # most important diagnostic sample (clean centred point source →
    # clean centred point source through the broader PSF). Train split
    # gets the same effect via random star injection, so prepend only
    # here.
    if (subset == "validate"
            and add_psf_pair_to_validation
            and psf_hst is not None
            and psf_euclid is not None):
        image_size = _probe_image_size(pairs_dir, subset)
        psf_inp, psf_tgt = make_psf_identity_pair(
            psf_hst, psf_euclid,
            image_size=image_size, rebin_factor=int(rebin_factor),
        )
        psf_pair_ds = tf.data.Dataset.from_tensors(
            (tf.constant(psf_inp, dtype=tf.float32),
             tf.constant(psf_tgt, dtype=tf.float32)),
        )
        ds = psf_pair_ds.concatenate(ds)

    # ── Star injection (real pairs + synthetic stars at sample_from_datasets) ──
    if star_injection_fraction > 0.0:
        if psf_hst is None or psf_euclid is None:
            raise ValueError(
                "star_injection_fraction > 0 requires both psf_hst and "
                "psf_euclid; pass them in to _make_pair_dataset."
            )
        image_size = _probe_image_size(pairs_dir, subset)
        star_ds = build_star_pair_dataset(
            psf_hst, psf_euclid,
            image_size=image_size,
            rebin_factor=int(rebin_factor),
            n_stars_min=1, n_stars_max=max_stars_per_image,
            seed=seed + 7919,        # decorrelate from pair-stream shuffle
        )
        f = float(star_injection_fraction)
        ds = tf.data.Dataset.sample_from_datasets(
            [ds, star_ds], weights=[1.0 - f, f], seed=seed,
        )

    # ── Linear-combination augmentation (operates on the joint stream) ──
    if linear_combo_fraction > 0.0:
        ds = apply_linear_combo_augmentation(
            ds, fraction=linear_combo_fraction,
            seed=seed + 31337,
        )

    ds = ds.batch(batch_size, drop_remainder=(subset == "train"))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# PSF-to-PSF identity probe (the load-bearing diagnostic)
# ---------------------------------------------------------------------------

def _psf_identity_residual(
    model, psf_hst_on_hr: np.ndarray, psf_euclid_on_hr: np.ndarray,
    *, rebin_factor: int = 2,
) -> float:
    """``‖sum_rebin(A_θ(PSF_HST)) − sum_rebin(PSF_Euclid)‖₁`` /
    ``‖sum_rebin(PSF_Euclid)‖₁``.

    Comparison happens in **LR space** because that's what the trainer
    optimises (loss compares ``sum_rebin(A_θ(input))`` to the LR
    target). Reporting the diagnostic at HR would give a misleading
    "this is bad" signal even when the trained model is perfectly
    correct at the LR scale the downstream pipeline cares about.

    Both PSF inputs must be on the HR grid with side divisible by
    ``rebin_factor``. The model is fed the HST PSF at HR, its output
    is sum-rebinned, then compared to the (also sum-rebinned) Euclid
    PSF.

    Returns a relative L1 (mean absolute error scaled by the PSF's
    L1 norm). Empirically the analytic Wiener inverse hits ~0.02 in
    HR space; the LR-space version of the same metric tends to be a
    bit smaller because the rebin averages away high-frequency
    residuals.
    """
    # Pad-or-crop the HR PSFs to a side divisible by rebin_factor.
    H = psf_hst_on_hr.shape[0]
    if H % rebin_factor != 0:
        # Centre-crop the HR PSF so sum-rebin is well-defined. The
        # PSF wings beyond this crop carry << 1% flux for the support
        # sizes we use.
        new_H = (H // rebin_factor) * rebin_factor
        cy = (H - new_H) // 2
        psf_hst_on_hr    = psf_hst_on_hr[cy:cy + new_H, cy:cy + new_H]
        psf_euclid_on_hr = psf_euclid_on_hr[cy:cy + new_H, cy:cy + new_H]

    # Reshape to single-sample batch with channel axis.
    x = psf_hst_on_hr[np.newaxis, ..., np.newaxis].astype(np.float32)
    y_pred_hr = model(x, training=False).numpy()[0, :, :, 0]
    y_pred_lr = sum_rebin_2d(y_pred_hr, int(rebin_factor))
    y_true_lr = sum_rebin_2d(
        psf_euclid_on_hr.astype(np.float32), int(rebin_factor),
    )
    err = float(np.abs(y_pred_lr - y_true_lr).sum())
    norm = float(np.abs(y_true_lr).sum()) + 1e-12
    return err / norm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    print("=" * 64)
    print(f"  Train transition model A_θ (HST PSF → Euclid PSF)")
    print("=" * 64)
    print(f"  pairs       = {args.pairs_dir}")
    print(f"  output      = {args.output}")
    print(f"  channels    = {args.channels}")
    print(f"  inner layers = {args.n_inner_layers}")
    print(f"  steps       = {args.steps}")
    print(f"  batch size  = {args.batch_size}")
    print(f"  LR          = {args.learning_rate:g}")
    print(f"  validate_every = {args.validate_every}")
    print(f"  weight_decay   = {args.weight_decay:g}")
    print(f"  star_injection_fraction = {args.star_injection_fraction:g} "
          f"(max {args.max_stars_per_image} stars/field)")
    print(f"  linear_combo_fraction   = {args.linear_combo_fraction:g}")
    print(f"  rebin_factor            = {args.rebin_factor} "
          f"(loss = ||sum_rebin(A_θ(HR)) − target_LR||)")
    print(f"  add PSF pair to val     = {args.add_psf_pair_to_validation}")
    if args.analytic_baseline_kernel:
        print(f"  analytic baseline       = {args.analytic_baseline_kernel}")
        print(f"    crop to side          = {args.baseline_crop}")
    else:
        print(f"  analytic baseline       = (none — identity residual)")
    print()

    t0 = time.time()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    print("[1/4] building model ...")

    # Optional analytic-baseline residual: load + centre-crop the
    # analytic Wiener differential kernel and pass it to the model.
    # ``f`` then sees the (artifact-laden) analytic kernel output and
    # learns to clean it up, instead of learning the full transition
    # from scratch.
    baseline_kernel = None
    if args.analytic_baseline_kernel:
        if not os.path.isfile(args.analytic_baseline_kernel):
            print(f"ERROR: --analytic-baseline-kernel path does not exist: "
                  f"{args.analytic_baseline_kernel}")
            return 1
        with fits.open(args.analytic_baseline_kernel) as hdul:
            full = np.asarray(hdul[0].data, dtype=np.float32)
        if full.ndim != 2 or full.shape[0] != full.shape[1]:
            print(f"ERROR: baseline kernel must be a 2-D square array; "
                  f"got shape {full.shape}")
            return 1
        crop = int(args.baseline_crop)
        if crop > 0 and crop < full.shape[0]:
            side = crop | 1            # force odd
            i0 = (full.shape[0] - side) // 2
            baseline_kernel = full[i0:i0 + side, i0:i0 + side]
            cropped_flux = float(baseline_kernel.sum())
            full_flux = float(full.sum())
            print(f"      analytic baseline: {full.shape} → "
                  f"{baseline_kernel.shape} (centre crop)")
            print(f"      cropped flux / full flux = "
                  f"{cropped_flux / max(full_flux, 1e-12):.4f}")
        else:
            baseline_kernel = full
            print(f"      analytic baseline: {full.shape} "
                  f"(no crop)  sum={float(full.sum()):.4f}")

    model = HSTtoEuclidTransition(
        channels=args.channels,
        n_inner_layers=args.n_inner_layers,
        kernel_size=args.kernel_size,
        baseline_kernel=baseline_kernel,
    )
    n_params = total_parameter_count(model)
    max_params = int(args.max_params)
    print(f"      params = {n_params:,}  (cap = {max_params:,})")
    if n_params > max_params:
        print(f"      ERROR: model exceeds --max-params={max_params:,} budget. "
              "Reduce --channels / --n-inner-layers / --kernel-size, "
              "or raise --max-params if you intentionally want a bigger model.")
        return 1
    print(f"      kernel_size = {args.kernel_size}, "
          f"n_inner_layers = {args.n_inner_layers}, "
          f"channels = {args.channels}")
    print(f"      receptive field = {model.receptive_field} px "
          f"({model.receptive_field * Config.DEFAULT_PIXEL_SCALE:.3f}\" "
          f"at HR scale)")

    # ── Optional: load frozen Phase-1 denoiser ──
    frozen_denoiser = None
    denoiser_summary_meta = None
    if args.frozen_denoiser:
        if not os.path.isfile(args.frozen_denoiser):
            print(f"ERROR: --frozen-denoiser path does not exist: "
                  f"{args.frozen_denoiser}")
            return 1
        # Resolve the summary JSON path (defaults to a sibling of the
        # weights file).
        summary_path = args.frozen_denoiser_summary
        if not summary_path:
            if args.frozen_denoiser.endswith(".weights.h5"):
                summary_path = (
                    args.frozen_denoiser[:-len(".weights.h5")]
                    + "_summary.json"
                )
            else:
                summary_path = args.frozen_denoiser + ".summary.json"
        # Try the alternative legacy name too.
        if not os.path.isfile(summary_path):
            alt = os.path.join(
                os.path.dirname(args.frozen_denoiser),
                "hst_denoiser_summary.json",
            )
            if os.path.isfile(alt):
                summary_path = alt
        if not os.path.isfile(summary_path):
            print(f"ERROR: frozen denoiser summary not found at "
                  f"{summary_path}. Pass --frozen-denoiser-summary "
                  f"explicitly or re-train with "
                  f"fasrc_train_hst_denoiser.py to regenerate.")
            return 1
        with open(summary_path) as f:
            den_summary = json.load(f)
        denoiser_summary_meta = {
            "weights_path": args.frozen_denoiser,
            "summary_path": summary_path,
            "channels":     int(den_summary.get("channels", 8)),
            "n_inner_layers": int(den_summary.get("n_inner_layers", 2)),
            "kernel_size":  int(den_summary.get("kernel_size", 7)),
        }
        frozen_denoiser = HSTDenoiser(
            channels=denoiser_summary_meta["channels"],
            n_inner_layers=denoiser_summary_meta["n_inner_layers"],
            kernel_size=denoiser_summary_meta["kernel_size"],
        )
        load_denoiser_weights(frozen_denoiser, args.frozen_denoiser)
        frozen_denoiser.trainable = False
        for layer in frozen_denoiser.layers:
            layer.trainable = False
        print(f"      loaded frozen denoiser: "
              f"channels={denoiser_summary_meta['channels']} "
              f"n_inner={denoiser_summary_meta['n_inner_layers']} "
              f"K={denoiser_summary_meta['kernel_size']}")
        # Sanity: warn if Phase-2 user forgot to set the noise ranges
        # — without noise the denoiser sees clean input it was never
        # trained on, and its identity error will leak through.
        if (args.alpha_max <= 0.0 and args.sigma_floor_max <= 0.0):
            print("      WARN: --frozen-denoiser set but noise ranges "
                  "are zero. The denoiser will see clean input, "
                  "outside its training distribution. Set "
                  "--alpha-min/--alpha-max and "
                  "--sigma-floor-min/--sigma-floor-max to match the "
                  "denoiser's Phase-1 ranges (typically α∈[0.5,1.0], "
                  "σ_floor∈[8,22]).")

    print("[2/4] loading PSFs for diagnostic ...")
    # Reuse the resampler used by the pair generator so the probe is
    # bit-equivalent to the training data. (Both helpers imported at
    # module scope.)
    psf_side = 421       # same as in fasrc_generate_transition_pairs.py
    psf_hst    = _load_hst_psf_on_hr(args.hst_psf, psf_side)
    psf_euclid = _load_euclid_vis_psf_on_hr(psf_side)
    print(f"      PSFs loaded (HR grid, {psf_side}² each)")

    print("[3/4] building datasets ...")
    # Train split: augmentations on. Validation split stays raw so the
    # reported val_L1 measures the model's behavior on the real (HR
    # synthetic) distribution, not the augmented one.
    train_ds = _make_pair_dataset(
        args.pairs_dir, "train", batch_size=args.batch_size,
        seed=args.seed, shuffle=True,
        rebin_factor=int(args.rebin_factor),
        star_injection_fraction=float(args.star_injection_fraction),
        max_stars_per_image=int(args.max_stars_per_image),
        linear_combo_fraction=float(args.linear_combo_fraction),
        psf_hst=psf_hst, psf_euclid=psf_euclid,
    )
    valid_ds = _make_pair_dataset(
        args.pairs_dir, "validate", batch_size=args.batch_size,
        seed=args.seed, shuffle=False,
        rebin_factor=int(args.rebin_factor),
        psf_hst=psf_hst, psf_euclid=psf_euclid,
        add_psf_pair_to_validation=bool(args.add_psf_pair_to_validation),
    )
    if args.dry_run:
        # Pull one batch to verify shapes.
        for inp_b, tgt_b in train_ds.take(1):
            print(f"      sample batch: input={tuple(inp_b.shape)} "
                  f"target={tuple(tgt_b.shape)}")
        runtime = time.time() - t0
        print(f"\nDRY RUN — no training. RUNTIME_SECONDS={runtime:.1f}")
        return 0

    optimiser = tf.keras.optimizers.Adam(args.learning_rate)
    loss_fn   = tf.keras.losses.MeanAbsoluteError()
    wd        = float(args.weight_decay)
    rebin     = int(args.rebin_factor)
    noise_enabled = (args.alpha_max > 0.0 or args.sigma_floor_max > 0.0)
    alpha_min       = tf.constant(float(args.alpha_min),       dtype=tf.float32)
    alpha_max       = tf.constant(float(args.alpha_max),       dtype=tf.float32)
    sigma_floor_min = tf.constant(float(args.sigma_floor_min), dtype=tf.float32)
    sigma_floor_max = tf.constant(float(args.sigma_floor_max), dtype=tf.float32)

    def _sum_rebin_tf(x):
        """Photometric sum-rebin via avg-pool × area. Matches the numpy
        ``sum_rebin_2d`` used at pair-generation + augmentation time."""
        if rebin == 1:
            return x
        return tf.nn.avg_pool2d(
            x, ksize=rebin, strides=rebin, padding="VALID",
        ) * (rebin ** 2)

    def _apply_noise_and_denoise(inp):
        """Add HLSP-style shot+sky noise per image, then optionally run
        through the frozen denoiser.

        Implements the same noise model as
        :func:`add_hlsp_noise` (Gaussian-approximated shot + sky/read
        floor) directly in the training graph: each example draws its
        own ``α`` and ``σ_floor`` so a single epoch covers the full
        spatial-variation distribution of t_eff and depth across an
        HLSP tile.

        ``tf.stop_gradient`` after the (frozen) denoiser ensures
        gradients only flow into the trainable deconvolver, even
        though the denoiser sits in the forward graph.
        """
        if not noise_enabled:
            return inp
        # Per-image α and σ_floor, broadcast across spatial+channel dims.
        B = tf.shape(inp)[0]
        alpha = tf.random.uniform(
            (B, 1, 1, 1), minval=alpha_min, maxval=alpha_max,
            dtype=tf.float32,
        )
        sigma_floor = tf.random.uniform(
            (B, 1, 1, 1), minval=sigma_floor_min, maxval=sigma_floor_max,
            dtype=tf.float32,
        )
        shot_var  = alpha * tf.maximum(inp, 0.0)
        total_var = shot_var + tf.square(sigma_floor)
        eps = tf.random.normal(tf.shape(inp), dtype=tf.float32) \
            * tf.sqrt(total_var)
        noisy = inp + eps
        if frozen_denoiser is not None:
            noisy = frozen_denoiser(noisy, training=False)
            noisy = tf.stop_gradient(noisy)
        return noisy

    @tf.function
    def train_step(inp, tgt):
        """L1 loss in LR space.

        Model produces HR output (same shape as input). The loss
        sum-rebins that HR output to LR and compares against the LR
        ``tgt``. This matches the downstream pipeline:
        ``HST_HR → A_θ → HR → sum_rebin → LR``. The model only has to
        be correct at LR — high-frequency spike-mismatch residuals
        between HST and Euclid PSFs get averaged away by the rebin.

        Optional Phase-2 mode: ``inp`` gets HLSP-style noise added per
        image, then passes through the (frozen) Phase-1 denoiser
        before the deconvolver. Both steps are no-ops when
        ``--alpha-max`` and ``--sigma-floor-max`` are 0 (legacy path).
        """
        inp_for_model = _apply_noise_and_denoise(inp)
        with tf.GradientTape() as tape:
            pred_hr = model(inp_for_model, training=True)
            pred_lr = _sum_rebin_tf(pred_hr)
            loss = loss_fn(tgt, pred_lr)
            if wd > 0:
                l2 = tf.add_n([
                    tf.nn.l2_loss(v) for v in model.trainable_variables
                    if "kernel" in v.name
                ])
                loss = loss + wd * l2
        grads = tape.gradient(loss, model.trainable_variables)
        optimiser.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def eval_step(inp, tgt):
        inp_for_model = _apply_noise_and_denoise(inp)
        pred_hr = model(inp_for_model, training=False)
        pred_lr = _sum_rebin_tf(pred_hr)
        return loss_fn(tgt, pred_lr)

    print(f"[4/4] training for {args.steps:,} steps ...")
    step = 0
    train_iter = iter(train_ds)
    log: list[dict] = []

    # ── Checkpoint paths. ──────────────────────────────────────────────
    # We save twice every validation:
    #   * ``args.output`` (e.g. ``transition_model.weights.h5``) — the
    #     LATEST weights. Always overwritten, so if SIGTERM hits at
    #     wall-time the most recent checkpoint is durable on disk.
    #   * ``…best.weights.h5`` — the weights with the LOWEST val_L1
    #     seen so far. Only overwritten when val_L1 improves. This is
    #     the file you'll usually want at inference time.
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if args.output.endswith(".weights.h5"):
        best_path = args.output[:-len(".weights.h5")] + ".best.weights.h5"
    else:
        best_path = args.output + ".best"
    best_val_loss = float("inf")
    best_step     = -1

    # ── SIGTERM handler — final save on graceful shutdown. ────────────
    # SLURM sends SIGTERM at wall-time before the hard SIGKILL; we
    # catch it, save the current weights to ``args.output``, then exit.
    # The per-validation save above already keeps the disk current, but
    # catching SIGTERM means we also capture any progress between the
    # last validation and the signal.
    _sigterm_received = {"flag": False}

    def _handle_sigterm(signum, frame):
        _sigterm_received["flag"] = True
        try:
            print(f"\n[SIGTERM caught] saving weights to {args.output} "
                  f"before exit ...", flush=True)
            save_model_weights(model, args.output)
            print(f"[SIGTERM caught] save complete.", flush=True)
        except Exception as e:
            print(f"[SIGTERM caught] save FAILED: {e}", flush=True)
        # Re-raise default behavior so the job actually terminates.
        raise SystemExit(0)

    _signal.signal(_signal.SIGTERM, _handle_sigterm)

    # Initial PSF probe (before any training — should be ~1 because
    # residual init → A_θ(x) ≈ x, so A_θ(PSF_HST) ≈ PSF_HST and the
    # LR-space comparison still reflects the PSF-shape difference).
    psf_err_init = _psf_identity_residual(
        model, psf_hst, psf_euclid, rebin_factor=rebin,
    )
    print(f"      PSF identity residual @ step 0: {psf_err_init:.4f} "
          f"(LR-space, rebin={rebin})")

    while step < args.steps:
        inp_b, tgt_b = next(train_iter)
        loss_v = train_step(inp_b, tgt_b)
        step += 1

        if step % args.validate_every == 0 or step == args.steps:
            # Compute validation L1 (no weight decay term).
            val_losses = []
            for inp_b, tgt_b in valid_ds:
                val_losses.append(float(eval_step(inp_b, tgt_b).numpy()))
            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            psf_err  = _psf_identity_residual(
                model, psf_hst, psf_euclid, rebin_factor=rebin,
            )
            elapsed  = time.time() - t0

            # ── Save latest weights every validation. ─────────────────
            # Cheap (the model is < 50 KB) and guarantees that any
            # SIGTERM after this point leaves usable weights on disk.
            try:
                save_model_weights(model, args.output)
            except Exception as e:
                print(f"      WARN: latest-save failed at step {step}: {e}")

            # ── Save-best on val_L1. ──────────────────────────────────
            best_tag = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step     = step
                try:
                    save_model_weights(model, best_path)
                    best_tag = "  [BEST so far → saved]"
                except Exception as e:
                    print(f"      WARN: best-save failed at step {step}: {e}")

            print(f"      step {step:6d}/{args.steps} "
                  f"train_L1={float(loss_v.numpy()):.4e} "
                  f"val_L1={val_loss:.4e} "
                  f"PSF_id_err={psf_err:.4f} "
                  f"({elapsed:.1f}s){best_tag}")
            log.append({
                "step":       int(step),
                "train_l1":   float(loss_v.numpy()),
                "val_l1":     val_loss,
                "psf_id_err": psf_err,
                "elapsed_s":  elapsed,
            })

    # Final save after the loop (idempotent — overwrites the last
    # per-validation checkpoint at exactly the same step).
    save_model_weights(model, args.output)
    print(f"  wrote latest weights → {args.output}")
    if best_step >= 0:
        print(f"  best weights → {best_path}  "
              f"(step {best_step}, val_L1 = {best_val_loss:.4e})")

    # Save summary sidecar.
    psf_err_final = _psf_identity_residual(
        model, psf_hst, psf_euclid, rebin_factor=rebin,
    )
    summary = {
        "model_path":     args.output,
        "best_model_path": best_path,
        "best_val_l1":    float(best_val_loss),
        "best_step":      int(best_step),
        "params":         int(n_params),
        "channels":       int(args.channels),
        "n_inner_layers": int(args.n_inner_layers),
        "kernel_size":    int(args.kernel_size),
        "max_params":     int(args.max_params),
        "analytic_baseline_kernel": str(args.analytic_baseline_kernel or ""),
        "baseline_crop":  int(args.baseline_crop),
        "steps":          int(args.steps),
        "batch_size":     int(args.batch_size),
        "learning_rate":  float(args.learning_rate),
        "weight_decay":   float(args.weight_decay),
        "star_injection_fraction": float(args.star_injection_fraction),
        "max_stars_per_image":     int(args.max_stars_per_image),
        "linear_combo_fraction":   float(args.linear_combo_fraction),
        "rebin_factor":            int(args.rebin_factor),
        "add_psf_pair_to_validation": bool(args.add_psf_pair_to_validation),
        "psf_id_err_init":  float(psf_err_init),
        "psf_id_err_final": float(psf_err_final),
        # Two-stage Phase-2 record: present iff this run used a frozen
        # denoiser. The pipeline worker checks for this key to decide
        # whether to load the denoiser at inference time.
        "denoiser":          denoiser_summary_meta,
        "noise":             {
            "alpha_min":       float(args.alpha_min),
            "alpha_max":       float(args.alpha_max),
            "sigma_floor_min": float(args.sigma_floor_min),
            "sigma_floor_max": float(args.sigma_floor_max),
        },
        "elapsed_s":      round(time.time() - t0, 1),
        "log":            log,
    }
    os.makedirs(os.path.dirname(args.summary_output), exist_ok=True)
    with open(args.summary_output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote summary → {args.summary_output}")
    print(f"\nRUNTIME_SECONDS={summary['elapsed_s']}")
    print(f"PSF_ID_ERR_FINAL={psf_err_final:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
