#!/usr/bin/env python
"""Phase 1 of the two-stage HST→Euclid chain: train the denoiser.

Trains :class:`HSTDenoiser` to map a noisy HST-PSF-blurred HR scene
back to its clean version:

    A_θ_denoiser(  scene ⊛ PSF_HST  +  ε  )  ≈  scene ⊛ PSF_HST

with ``ε`` drawn from :func:`add_hlsp_noise` (Gaussian-approximated
HLSP shot + sky/read noise, see that function's docstring for the
derivation). Training pairs come from the existing
``records_transition/input_{train,validate}.tfrecord`` shards
produced by ``fasrc_generate_transition_pairs.py`` — those shards
*are* the clean HST-blurred scenes, which makes them the target. We
add noise on-the-fly to make the input.

Output: ``$DATA_DIR/hst_psf/hst_denoiser.weights.h5`` plus
``hst_denoiser_summary.json`` sidecar.

The downstream Phase-2 trainer (``fasrc_train_transition_model.py``
with ``--frozen-denoiser``) loads the resulting weights, freezes
them, and prepends the denoiser to the deconvolver's forward pass.
"""

from __future__ import annotations

import argparse
import json
import os
import signal as _signal
import sys
import time
from typing import Optional

import numpy as np
import tensorflow as tf

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.sky.tfrecord import (
    read_multiband_skyimages, tfrecord_path,
)
from euclid_polish.sky.types import MultiBandSkyImage
from euclid_polish.training.transition_augmentations import (
    add_hlsp_noise, sample_hlsp_noise_params,
)
from euclid_polish.training.transition_model import (
    HSTDenoiser, save_denoiser_weights, total_denoiser_parameter_count,
)


PAIRS_DIR     = os.path.join(Config.DATA_DIR, "images", "records_transition")
DEFAULT_MODEL = os.path.join(Config.DATA_DIR, "hst_psf",
                             "hst_denoiser.weights.h5")
SUMMARY_JSON  = os.path.join(Config.DATA_DIR, "hst_psf",
                             "hst_denoiser_summary.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs-dir", default=PAIRS_DIR,
                   help="Source TFRecords (output of "
                        "fasrc_generate_transition_pairs.py). The "
                        "denoiser reads only the ``input_*.tfrecord`` "
                        "files — those are the clean HST-PSF-blurred "
                        "HR scenes. Target side of the transition pair "
                        "is unused here.")
    p.add_argument("--output", default=DEFAULT_MODEL,
                   help="Weights output path (.weights.h5).")
    p.add_argument("--summary-output", default=SUMMARY_JSON,
                   help="JSON sidecar with training summary stats.")
    p.add_argument("--steps", type=int, default=20_000,
                   help="Total training steps.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Examples per gradient step.")
    p.add_argument("--learning-rate", type=float, default=2e-3,
                   help="Adam learning rate.")
    p.add_argument("--validate-every", type=int, default=500,
                   help="Run validation + log every N steps.")
    p.add_argument("--channels", type=int, default=8,
                   help="Hidden channel width C. Default 8.")
    p.add_argument("--n-inner-layers", type=int, default=2,
                   help="Inner Conv(C→C, K) layers. Default 2 → 4 "
                        "total conv layers → 25-px receptive field "
                        "at K=7.")
    p.add_argument("--kernel-size", type=int, default=7,
                   help="Spatial kernel side. Default 7 — wider than "
                        "the deconvolver's K=3 because denoising "
                        "benefits from a larger context window.")
    p.add_argument("--max-params", type=int, default=10_000,
                   help="Hard cap on trainable parameter count. "
                        "Default 10k leaves some headroom over the "
                        "default 7056-param model.")
    p.add_argument("--weight-decay", type=float, default=1e-5,
                   help="L2 regularisation strength.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed.")
    # ── Noise-model hyperparameters ──
    p.add_argument("--alpha-min", type=float, default=0.5,
                   help="Lower bound of shot-noise coefficient α = "
                        "C/t_eff. Default 0.5 corresponds to "
                        "COSMOS-deep t_eff ≈ 3000 s.")
    p.add_argument("--alpha-max", type=float, default=1.0,
                   help="Upper bound of α. Default 1.0 corresponds to "
                        "t_eff ≈ 1500 s.")
    p.add_argument("--sigma-floor-min", type=float, default=8.0,
                   help="Lower bound of sky/read floor in e⁻ per HR "
                        "pixel. Default 8 corresponds to deep "
                        "COSMOS regions.")
    p.add_argument("--sigma-floor-max", type=float, default=22.0,
                   help="Upper bound of σ_floor. Default 22 "
                        "corresponds to shallow regions.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build model + dataset, print sizes, exit.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset: (noisy, clean) HR scenes streamed from the input_*.tfrecord shards
# ---------------------------------------------------------------------------

def _probe_image_size(pairs_dir: str, subset: str = "train") -> int:
    path = tfrecord_path(pairs_dir, f"input_{subset}")
    if not os.path.exists(path):
        raise RuntimeError(
            f"input shard not found: {path}\n"
            f"Run fasrc_generate_transition_pairs.py first."
        )
    if os.path.getsize(path) == 0:
        raise RuntimeError(f"input shard is 0 bytes: {path}")
    records = read_multiband_skyimages(path, num_images=1)
    if not records:
        raise RuntimeError(f"input shard has no records: {path}")
    H, W, C = records[0].data.shape
    if H != W:
        raise RuntimeError(
            f"non-square pair shape {records[0].data.shape}"
        )
    return int(H)


def _make_denoiser_dataset(
    pairs_dir: str, subset: str, *,
    batch_size: int, seed: int, shuffle: bool,
    alpha_min: float, alpha_max: float,
    sigma_floor_min: float, sigma_floor_max: float,
):
    """Yield ``(noisy_HR, clean_HR)`` batches.

    Reads ``input_{subset}.tfrecord`` (the clean HR scenes that were
    written there as the *input* side of the transition pair). For
    each record:

      * ``clean``  = record.data[..., 0]    (VIS-only HR)
      * ``α, σ``   = sample_hlsp_noise_params(rng_per_image)
      * ``noisy``  = add_hlsp_noise(clean, α, σ)
      * emit ``(noisy[..., np.newaxis], clean[..., np.newaxis])``

    The per-image noise sampling means a single epoch through the
    dataset hits a wide range of effective-exposure conditions, which
    teaches the denoiser to generalise across the spatial variation
    of t_eff and depth in real HLSP tiles.
    """
    path = tfrecord_path(pairs_dir, f"input_{subset}")
    if not os.path.exists(path):
        raise RuntimeError(f"input shard not found: {path}")

    side = _probe_image_size(pairs_dir, subset)

    # Per-example noise via a numpy generator — easier than tf.random
    # because we want signal-dependent variance, which is awkward in
    # the tf.data graph. Use a from_generator wrapper.
    rng = np.random.default_rng(seed)

    def _gen():
        ds = tf.data.TFRecordDataset([path])
        for raw in ds:
            try:
                img = MultiBandSkyImage.from_tfrecord(raw)
            except Exception:
                continue
            data = np.asarray(img.data, dtype=np.float32)
            if data.ndim != 3:
                continue
            # input shards are single-channel VIS (see pair-gen).
            clean = data[:, :, 0]
            alpha, sigma_floor = sample_hlsp_noise_params(
                rng,
                alpha_min=alpha_min, alpha_max=alpha_max,
                sigma_floor_min=sigma_floor_min,
                sigma_floor_max=sigma_floor_max,
            )
            noisy = add_hlsp_noise(
                clean, alpha=alpha, sigma_floor=sigma_floor, rng=rng,
            )
            yield (noisy[..., np.newaxis], clean[..., np.newaxis])

    sig = (
        tf.TensorSpec(shape=(side, side, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(side, side, 1), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(_gen, output_signature=sig)
    if shuffle:
        ds = ds.shuffle(buffer_size=256, seed=seed,
                        reshuffle_each_iteration=True)
    ds = ds.repeat() if shuffle else ds
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    print("=" * 64)
    print("  Train HST denoiser (Phase 1 of two-stage chain)")
    print("=" * 64)
    print(f"  pairs                 = {args.pairs_dir}")
    print(f"  output                = {args.output}")
    print(f"  channels              = {args.channels}")
    print(f"  inner layers          = {args.n_inner_layers}")
    print(f"  kernel size           = {args.kernel_size}")
    print(f"  steps                 = {args.steps}")
    print(f"  batch size            = {args.batch_size}")
    print(f"  LR                    = {args.learning_rate:g}")
    print(f"  validate_every        = {args.validate_every}")
    print(f"  noise α range         = [{args.alpha_min}, {args.alpha_max}]")
    print(f"  noise σ_floor range   = [{args.sigma_floor_min}, "
          f"{args.sigma_floor_max}] e⁻")
    print()

    t0 = time.time()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    print("[1/3] building denoiser model ...")
    model = HSTDenoiser(
        channels=args.channels,
        n_inner_layers=args.n_inner_layers,
        kernel_size=args.kernel_size,
    )
    n_params = total_denoiser_parameter_count(model)
    max_params = int(args.max_params)
    print(f"      params = {n_params:,}  (cap = {max_params:,})")
    if n_params > max_params:
        print(f"      ERROR: model exceeds --max-params={max_params:,}. "
              "Reduce channels/n_inner_layers/kernel_size or raise "
              "the cap.")
        return 1
    print(f"      receptive field = {model.receptive_field} px "
          f"({model.receptive_field * Config.DEFAULT_PIXEL_SCALE:.3f}\" "
          f"at HR scale)")

    print("[2/3] building datasets ...")
    side_train = _probe_image_size(args.pairs_dir, "train")
    print(f"      train shard image side = {side_train}")
    train_ds = _make_denoiser_dataset(
        args.pairs_dir, "train",
        batch_size=args.batch_size, seed=args.seed, shuffle=True,
        alpha_min=args.alpha_min, alpha_max=args.alpha_max,
        sigma_floor_min=args.sigma_floor_min,
        sigma_floor_max=args.sigma_floor_max,
    )
    valid_ds = _make_denoiser_dataset(
        args.pairs_dir, "validate",
        batch_size=args.batch_size, seed=args.seed, shuffle=False,
        alpha_min=args.alpha_min, alpha_max=args.alpha_max,
        sigma_floor_min=args.sigma_floor_min,
        sigma_floor_max=args.sigma_floor_max,
    )

    if args.dry_run:
        for noisy_b, clean_b in train_ds.take(1):
            print(f"      sample batch: noisy={tuple(noisy_b.shape)} "
                  f"clean={tuple(clean_b.shape)}")
            print(f"      noise check: input σ ≈ "
                  f"{float(tf.math.reduce_std(noisy_b - clean_b)):.2f} e⁻")
        runtime = time.time() - t0
        print(f"\nDRY RUN. RUNTIME_SECONDS={runtime:.1f}")
        return 0

    optimiser = tf.keras.optimizers.Adam(args.learning_rate)
    loss_fn   = tf.keras.losses.MeanAbsoluteError()
    wd        = float(args.weight_decay)

    @tf.function
    def train_step(noisy, clean):
        with tf.GradientTape() as tape:
            pred = model(noisy, training=True)
            loss = loss_fn(clean, pred)
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
    def eval_step(noisy, clean):
        return loss_fn(clean, model(noisy, training=False))

    print(f"[3/3] training for {args.steps:,} steps ...")

    # ── Checkpoint paths (same convention as the deconvolver trainer) ──
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if args.output.endswith(".weights.h5"):
        best_path = args.output[:-len(".weights.h5")] + ".best.weights.h5"
    else:
        best_path = args.output + ".best"
    best_val_loss = float("inf")
    best_step     = -1

    # SIGTERM → save before exit (SLURM wall-time graceful path).
    _sigterm = {"flag": False}
    def _handle_sigterm(signum, frame):
        _sigterm["flag"] = True
        try:
            print(f"\n[SIGTERM] saving weights to {args.output} ...",
                  flush=True)
            save_denoiser_weights(model, args.output)
            print(f"[SIGTERM] save complete.", flush=True)
        except Exception as e:
            print(f"[SIGTERM] save FAILED: {e}", flush=True)
        raise SystemExit(0)
    _signal.signal(_signal.SIGTERM, _handle_sigterm)

    step = 0
    train_iter = iter(train_ds)
    log: list[dict] = []
    while step < args.steps:
        noisy_b, clean_b = next(train_iter)
        loss_v = train_step(noisy_b, clean_b)
        step += 1

        if step % args.validate_every == 0 or step == args.steps:
            val_losses = []
            for noisy_b, clean_b in valid_ds:
                val_losses.append(float(eval_step(noisy_b, clean_b).numpy()))
            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            elapsed = time.time() - t0

            try:
                save_denoiser_weights(model, args.output)
            except Exception as e:
                print(f"      WARN: latest-save failed at step {step}: {e}")

            best_tag = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step     = step
                try:
                    save_denoiser_weights(model, best_path)
                    best_tag = "  [BEST so far → saved]"
                except Exception as e:
                    print(f"      WARN: best-save failed at step {step}: {e}")

            print(f"      step {step:6d}/{args.steps} "
                  f"train_L1={float(loss_v.numpy()):.4e} "
                  f"val_L1={val_loss:.4e} "
                  f"({elapsed:.1f}s){best_tag}", flush=True)
            log.append({
                "step":      int(step),
                "train_l1":  float(loss_v.numpy()),
                "val_l1":    val_loss,
                "elapsed_s": elapsed,
            })

    save_denoiser_weights(model, args.output)
    print(f"  wrote latest weights → {args.output}")
    if best_step >= 0:
        print(f"  best weights → {best_path}  "
              f"(step {best_step}, val_L1 = {best_val_loss:.4e})")

    summary = {
        "model_path":       args.output,
        "best_model_path":  best_path if best_step >= 0 else None,
        "best_val_l1":      best_val_loss if best_step >= 0 else None,
        "best_step":        best_step,
        "params":           n_params,
        "channels":         int(args.channels),
        "n_inner_layers":   int(args.n_inner_layers),
        "kernel_size":      int(args.kernel_size),
        "max_params":       max_params,
        "steps":            int(args.steps),
        "batch_size":       int(args.batch_size),
        "learning_rate":    float(args.learning_rate),
        "weight_decay":     float(args.weight_decay),
        "noise":            {
            "alpha_min":       float(args.alpha_min),
            "alpha_max":       float(args.alpha_max),
            "sigma_floor_min": float(args.sigma_floor_min),
            "sigma_floor_max": float(args.sigma_floor_max),
        },
        "elapsed_s":        round(time.time() - t0, 1),
        "log":              log,
    }
    with open(args.summary_output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote summary       → {args.summary_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
