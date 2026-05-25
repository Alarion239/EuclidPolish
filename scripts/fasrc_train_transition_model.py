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

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config


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
                   help="Hidden channel width C. Default 12 → ~4150 "
                        "params, inside the 5k budget.")
    p.add_argument("--n-inner-layers", type=int, default=3,
                   help="Number of inner ``Conv(C→C)`` layers. Default 3 "
                        "→ 5 total convs, 11px receptive field.")
    p.add_argument("--weight-decay", type=float, default=1e-5,
                   help="L2 regularisation strength applied to conv "
                        "weights. Small — the residual structure already "
                        "discourages large updates.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for shuffling + initialiser.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build model + dataset, print sizes, then exit.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pair TFRecord dataset (input, target) → tf.data
# ---------------------------------------------------------------------------

def _make_pair_dataset(
    pairs_dir: str, subset: str, *, batch_size: int, seed: int, shuffle: bool,
):
    """Build a tf.data.Dataset of ``(input, target)`` batches.

    Each side is parsed independently from its own TFRecord file and the
    two streams are zipped — same trick the multi-band loader uses for
    ``(clean, dirty)`` pairs. The pairs are written in matched order by
    the generator script, so zipping preserves correspondence.
    """
    import tensorflow as tf
    from euclid_polish.sky.tfrecord import parse_record_graph_v2, tfrecord_path

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
    ds = ds.batch(batch_size, drop_remainder=(subset == "train"))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# PSF-to-PSF identity probe (the load-bearing diagnostic)
# ---------------------------------------------------------------------------

def _psf_identity_residual(
    model, psf_hst_on_hr: np.ndarray, psf_euclid_on_hr: np.ndarray,
) -> float:
    """``‖A_θ(PSF_HST) − PSF_Euclid‖₁`` / ``‖PSF_Euclid‖₁``.

    The cleanest single-number sanity check that A_θ has learned the
    transition. If A_θ is doing its job, applying it to the HST PSF
    itself should produce something close to the Euclid PSF. We compute
    this on the same HR-grid PSFs the training-pair generator
    convolved with.

    Returns a relative L1 (= mean absolute error scaled by PSF L1 norm).
    Empirically the analytic Wiener inverse hits ~0.02 here; a well-
    trained A_θ should be in the same ballpark or better.
    """
    import tensorflow as tf
    # Reshape to a single-sample batch with channel axis.
    x = psf_hst_on_hr[np.newaxis, ..., np.newaxis].astype(np.float32)
    y_pred = model(x, training=False).numpy()[0, :, :, 0]
    err = float(np.abs(y_pred - psf_euclid_on_hr).sum())
    norm = float(np.abs(psf_euclid_on_hr).sum()) + 1e-12
    return err / norm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # tf import is gated behind argparse so --dry-run prints fast.
    import tensorflow as tf

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
    print()

    t0 = time.time()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    from euclid_polish.training.transition_model import (
        HSTtoEuclidTransition, save_model_weights, total_parameter_count,
    )

    print("[1/4] building model ...")
    model = HSTtoEuclidTransition(
        channels=args.channels,
        n_inner_layers=args.n_inner_layers,
    )
    n_params = total_parameter_count(model)
    print(f"      params = {n_params:,}  (cap = 5,000)")
    if n_params > 5000:
        print(f"      ERROR: model exceeds 5k-param budget. "
              "Reduce --channels or --n-inner-layers.")
        return 1
    print(f"      receptive field = {model.receptive_field} px "
          f"({model.receptive_field * Config.DEFAULT_PIXEL_SCALE:.3f}\" "
          f"at HR scale)")

    print("[2/4] loading PSFs for diagnostic ...")
    # Reuse the resampler used by the pair generator so the probe is
    # bit-equivalent to the training data.
    from scripts.fasrc_generate_transition_pairs import (
        _load_hst_psf_on_hr, _load_euclid_vis_psf_on_hr,
    )
    psf_side = 421       # same as in fasrc_generate_transition_pairs.py
    psf_hst    = _load_hst_psf_on_hr(args.hst_psf, psf_side)
    psf_euclid = _load_euclid_vis_psf_on_hr(psf_side)
    print(f"      PSFs loaded (HR grid, {psf_side}² each)")

    print("[3/4] building datasets ...")
    train_ds = _make_pair_dataset(
        args.pairs_dir, "train", batch_size=args.batch_size,
        seed=args.seed, shuffle=True,
    )
    valid_ds = _make_pair_dataset(
        args.pairs_dir, "validate", batch_size=args.batch_size,
        seed=args.seed, shuffle=False,
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

    @tf.function
    def train_step(inp, tgt):
        with tf.GradientTape() as tape:
            pred = model(inp, training=True)
            loss = loss_fn(tgt, pred)
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
        pred = model(inp, training=False)
        return loss_fn(tgt, pred)

    print(f"[4/4] training for {args.steps:,} steps ...")
    step = 0
    train_iter = iter(train_ds)
    log: list[dict] = []

    # Initial PSF probe (before any training — should be ~1 because
    # residual init → A_θ(x) ≈ x, so A_θ(PSF_HST) ≈ PSF_HST).
    psf_err_init = _psf_identity_residual(model, psf_hst, psf_euclid)
    print(f"      PSF identity residual @ step 0: {psf_err_init:.4f}")

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
            psf_err  = _psf_identity_residual(model, psf_hst, psf_euclid)
            elapsed  = time.time() - t0
            print(f"      step {step:6d}/{args.steps} "
                  f"train_L1={float(loss_v.numpy()):.4e} "
                  f"val_L1={val_loss:.4e} "
                  f"PSF_id_err={psf_err:.4f} "
                  f"({elapsed:.1f}s)")
            log.append({
                "step":       int(step),
                "train_l1":   float(loss_v.numpy()),
                "val_l1":     val_loss,
                "psf_id_err": psf_err,
                "elapsed_s":  elapsed,
            })

    # Save weights.
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_model_weights(model, args.output)
    print(f"  wrote weights → {args.output}")

    # Save summary sidecar.
    psf_err_final = _psf_identity_residual(model, psf_hst, psf_euclid)
    summary = {
        "model_path":     args.output,
        "params":         int(n_params),
        "channels":       int(args.channels),
        "n_inner_layers": int(args.n_inner_layers),
        "steps":          int(args.steps),
        "batch_size":     int(args.batch_size),
        "learning_rate":  float(args.learning_rate),
        "weight_decay":   float(args.weight_decay),
        "psf_id_err_init":  float(psf_err_init),
        "psf_id_err_final": float(psf_err_final),
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
