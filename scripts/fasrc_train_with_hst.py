#!/usr/bin/env python
"""Thin training entry point that mixes synthetic + HST TFRecords.

Delegates to the existing :class:`Trainer` but pulls batches from two
TFRecord sources (the synthetic ``records_v2`` and the HST-derived
``records_v2_hst``), interleaving them at a user-specified per-batch
fraction.

CLI:

    python scripts/fasrc_train_with_hst.py \\
        --steps 400000 --batch-size 16 --hst-fraction 0.10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tf_keras.losses import MeanAbsoluteError
from tf_keras.optimizers.schedules import PiecewiseConstantDecay

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.observability.reporter import Reporter
from euclid_polish.observability.resource_sampler import ResourceSampler
from euclid_polish.training import Trainer
from euclid_polish.training.data_multiband import MultiBandEuclidDataset
from euclid_polish.training.forward_op import HSTForwardOp
from euclid_polish.training.models.wdsr import wdsr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps",        type=int,   default=Config.DEFAULT_TRAIN_STEPS)
    # Explicit per-lane batch composition for ``Trainer.train_step_sky``.
    # Each step's batch is the fixed contiguous block layout
    # ``[n_syn synthetic | n_hst HST | n_anchor star-anchor]``; the batch
    # size is just their sum. Counts (not fractions) because the trainer
    # slices by these exact integers — no rounding, no "fraction rounds to
    # 0 rows" footgun.
    p.add_argument("--n-syn", type=int, default=Config.DEFAULT_BATCH_SIZE,
                   help="Synthetic examples per batch. Must be ≥ 1 "
                        "(the synthetic lane is always present).")
    p.add_argument("--n-hst", type=int, default=0,
                   help="HST examples per batch (SR=sky lane, "
                        "|asinh(H⊛SR) - HST_image|). 0 disables it. "
                        "Requires the HST records (--records-hst) and the "
                        "F814W ePSF.")
    p.add_argument("--n-anchor", type=int, default=0,
                   help="Star-anchor examples per batch (operator-free: "
                        "masked |SR - delta| at the catalog star pixel). "
                        "0 disables it. Requires the star-anchor records "
                        "(--records-anchor) built by "
                        "fasrc_generate_star_anchor_tfrecords.py.")
    p.add_argument("--records-syn",  default=Config.RECORDS_DIR_V2,
                   help="Synthetic TFRecord directory.")
    p.add_argument("--records-hst",
                   default=os.path.join(Config.DATA_DIR, "images", "records_v2_hst"),
                   help="HST-derived TFRecord directory (built by "
                        "fasrc_generate_hst_tfrecords.py).")
    p.add_argument("--records-anchor",
                   default=os.path.join(
                       Config.DATA_DIR, "images", "records_v2_star_anchor"),
                   help="Star-anchor TFRecord directory (built by "
                        "fasrc_generate_star_anchor_tfrecords.py).")
    p.add_argument("--synthetic-loss-weight", type=float, default=1.0,
                   help="Per-example loss multiplier for synthetic "
                        "records. Default 1.0; set to 0 to keep them in "
                        "the batch mix but zero their gradient (ablation).")
    p.add_argument("--hst-loss-weight", type=float, default=1.0,
                   help="Per-example loss multiplier for HST records. "
                        "Default 1.0; raise to up-weight the HST "
                        "supervised path, 0 to ablate.")
    p.add_argument("--star-anchor-loss-weight", type=float, default=1.0,
                   help="Multiplier on the per-example star-anchor loss "
                        "(masked at the star pixel). Default 1.0; bump to "
                        "up-weight the anchor, set to 0 to ablate (anchor "
                        "data still loaded, loss contribution zeroed out).")
    p.add_argument("--ckpt-dir",     default=Config.DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--num-res-blocks", type=int, default=Config.DEFAULT_NUM_RES_BLOCKS)
    p.add_argument("--evaluate-every", type=int, default=Config.DEFAULT_EVALUATE_EVERY)
    p.add_argument("--learning-rate", type=float, default=0.0,
                   help="Constant Adam learning rate for the whole run "
                        "(e.g. 0.001). 0 (default) uses the two-phase "
                        "PiecewiseConstantDecay schedule 1e-3 → 5e-4 at "
                        "steps//2. A constant LR is simpler to reason about "
                        "for an exploratory restart; the decay gives a "
                        "slightly sharper final result. With a larger batch, "
                        "scale the LR up accordingly.")
    p.add_argument("--nonneg-sr-weight", type=float,
                   default=Config.NONNEG_SR_WEIGHT,
                   help="Weight of the SR non-negativity penalty "
                        "λ·mean(relu(-SR)) added to every step's loss. SR is "
                        "the model's single deconvolved-sky output, so this "
                        "constrains all three lanes toward physical (≥0) "
                        "flux at once. 0 disables. Default from "
                        "Config.NONNEG_SR_WEIGHT.")
    p.add_argument("--resume-baseline", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="On a resumed run, measure the restored checkpoint's "
                        "score as the save-best bar to beat (default). Use "
                        "--no-resume-baseline to ignore the previous best and "
                        "overwrite it — appropriate when the architecture "
                        "changed so the old score is meaningless.")
    p.add_argument("--save-best-w-syn", type=float, default=1.0,
                   help="Weight of synthetic PSNR (dB) in the composite "
                        "save-best score. Default 1.")
    p.add_argument("--save-best-w-hst", type=float, default=1.0,
                   help="Weight of HST PSNR (dB) in the composite "
                        "save-best score. No effect when the HST "
                        "validate split is absent. Default 1.")
    p.add_argument("--save-best-w-anchor", type=float, default=0.0,
                   help="Weight of the star-anchor PSNR (dB) in the "
                        "composite save-best score (ADDED — higher is "
                        "better, like the other two PSNRs). The anchor "
                        "PSNR is masked to the star pixel. Defaults to 0 "
                        "(monitored-only).")
    p.add_argument("--forward-op-crop-half", type=int, default=0,
                   help="Optional central crop of the F814W PSF kernel used "
                        "by the HST forward op → (2·crop+1)² kernel. "
                        "Default 0 = use the FULL saved PSF: the forward op "
                        "convolves via an explicit FFT (O(N log N)), so the "
                        "full PSF — wings and all — is exact and fast. Set a "
                        "positive value only to ablate the PSF wings.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    reporter = Reporter.from_env()
    print("=" * 64)
    print(f"  WDSR training with HST + star-anchor mix")
    print("=" * 64)
    print(f"  steps              = {args.steps}")
    print(f"  batch layout       = syn {args.n_syn} / hst {args.n_hst} / "
          f"anchor {args.n_anchor}  "
          f"(batch {args.n_syn + args.n_hst + args.n_anchor})")
    print(f"  synthetic records  = {args.records_syn}")
    print(f"  HST records        = {args.records_hst}")
    print(f"  star-anchor records= {args.records_anchor}")
    print(f"  loss weights syn/hst/anchor = {args.synthetic_loss_weight}/"
          f"{args.hst_loss_weight}/{args.star_anchor_loss_weight}")
    print(f"  checkpoint dir     = {args.ckpt_dir}")
    print()

    t0 = time.time()
    if args.dry_run:
        print("DRY RUN — would train.")
        print(f"\nRUNTIME_SECONDS={time.time() - t0:.1f}")
        return 0

    # Explicit per-lane batch composition. The batch is the fixed
    # contiguous layout ``[n_syn | n_hst | n_anchor]``; batch_size is the sum.
    n_syn    = int(args.n_syn)
    n_hst    = int(args.n_hst)
    n_anchor = int(args.n_anchor)
    if min(n_syn, n_hst, n_anchor) < 0:
        raise SystemExit("--n-syn / --n-hst / --n-anchor must be ≥ 0")
    if n_syn < 1:
        raise SystemExit("--n-syn must be ≥ 1 (the synthetic lane is always on)")
    use_hst      = n_hst > 0
    use_anchor   = n_anchor > 0
    fixed_layout = use_hst or use_anchor
    batch_size   = n_syn + n_hst + n_anchor
    lane_counts  = (n_syn, n_hst, n_anchor) if fixed_layout else None

    # Two dataset instances — one for training, one for validation. HST
    # and star-anchor sources are only mixed into training; the synthetic
    # validation split stays pure so the metric is comparable across runs.
    reporter.set_stage("building datasets")
    if fixed_layout:
        print(f"      fixed batch layout: syn={n_syn} hst={n_hst} "
              f"anchor={n_anchor} (batch={batch_size})")
        train_dataset = MultiBandEuclidDataset(
            subset="train",
            records_dir=args.records_syn,
            hst_records_dir=args.records_hst if use_hst else None,
            anchor_records_dir=args.records_anchor if use_anchor else None,
        ).dataset_fixed_layout(n_syn, n_hst, n_anchor, random_transform=True)
    else:
        print(f"      pure-synthetic batch: {batch_size}")
        train_dataset = MultiBandEuclidDataset(
            subset="train",
            records_dir=args.records_syn,
        ).dataset(batch_size=batch_size, random_transform=True)
    valid_dataset = MultiBandEuclidDataset(
        subset="validate",
        records_dir=args.records_syn,
    ).dataset(batch_size=batch_size, random_transform=False,
              repeat_count=1)

    # Additive multi-source validation datasets. These never influence
    # save-best (which stays keyed on the synthetic metric); they only
    # surface per-source progress in the training log so a regime change
    # that helps HST / star-anchor while dipping synthetic is visible.
    # Both are built leniently — a missing split logs a note and yields
    # None rather than aborting the run.
    hst_valid_dataset = None
    if use_hst:
        try:
            hst_valid_dataset = MultiBandEuclidDataset(
                subset="validate",
                records_dir=args.records_hst,
            ).dataset(batch_size=batch_size,
                      random_transform=False, repeat_count=1)
        except FileNotFoundError:
            print("  note: no HST validate split found — "
                  "skipping HST validation logging")

    anchor_valid_dataset = None
    if use_anchor:
        anchor_valid_dataset = MultiBandEuclidDataset(
            subset="validate",
            records_dir=args.records_syn,
            anchor_records_dir=args.records_anchor,
        ).anchor_dataset(batch_size=batch_size, random_transform=False,
                         repeat_count=1)
        if anchor_valid_dataset is None:
            print("  note: no star-anchor validate split found — "
                  "skipping anchor validation logging")

    # Model + loss + optimizer (same recipe as the standard trainer).
    reporter.set_stage("building model + optimizer")
    scale = Config.DEFAULT_REBIN_FACTOR
    model = wdsr(
        scale=scale, num_res_blocks=args.num_res_blocks,
        nchan_in=Config.NUM_LR_CHANNELS, nchan_out=Config.NUM_HR_CHANNELS,
    )
    # Learning rate: a constant when ``--learning-rate`` is given, else the
    # two-phase decay (1e-3 → 5e-4 at the half-way step). Adam accepts a
    # bare float or a schedule object interchangeably.
    if args.learning_rate and args.learning_rate > 0:
        schedule = float(args.learning_rate)
        print(f"      learning rate: constant {schedule:g}")
    else:
        schedule = PiecewiseConstantDecay(
            boundaries=[args.steps // 2],
            values=[1e-3, 5e-4],
        )
        print(f"      learning rate: 1e-3 → 5e-4 at step {args.steps // 2}")

    # HST forward op for the SR=sky objective: H ⊛ SR (HST F814W PSF,
    # rebin_factor=1 — HST shares SR's 0.05″ grid). The star-anchor lane
    # is operator-free, so no VIS forward op is built for training.
    crop_half = (args.forward_op_crop_half
                 if args.forward_op_crop_half and args.forward_op_crop_half > 0
                 else None)
    hst_forward_op = (
        HSTForwardOp(crop_half_side=crop_half) if use_hst else None
    )
    if hst_forward_op is not None:
        print(f"      HST forward op: F814W PSF kernel "
              f"{hst_forward_op.psf_kernel_size}×{hst_forward_op.psf_kernel_size} "
              f"(crop_half_side={crop_half})")

    trainer = Trainer(
        model=model,
        loss=MeanAbsoluteError(),
        learning_rate=schedule,
        checkpoint_dir=args.ckpt_dir,
        hst_forward_op=hst_forward_op,
        synthetic_loss_weight=float(args.synthetic_loss_weight),
        hst_loss_weight=float(args.hst_loss_weight),
        star_anchor_loss_weight=float(args.star_anchor_loss_weight),
        nonneg_sr_weight=float(args.nonneg_sr_weight),
    )
    print(f"      total trainable params: "
          f"{sum(int(np.prod(v.shape)) for v in model.trainable_variables):,}")
    reporter.set_stage(f"training {args.steps} steps")
    print(f"\n  training {args.steps} steps ...")

    def _on_train_step(step: int, total: int) -> None:
        reporter.set_step(step, total, "train")

    print(f"      save-best weights: syn={args.save_best_w_syn:g}, "
          f"hst={args.save_best_w_hst:g}, anchor={args.save_best_w_anchor:g}")
    # Sample GPU + CPU utilisation through training so the WebUI shows a
    # live smoothed gauge and the post-mortem records mean/peak.
    with ResourceSampler(reporter):
        trainer.train(
            train_dataset, valid_dataset, steps=int(args.steps),
            evaluate_every=int(args.evaluate_every),
            save_best_only=True,
            step_callback=_on_train_step,
            hst_valid_dataset=hst_valid_dataset,
            anchor_valid_dataset=anchor_valid_dataset,
            save_best_weights=(
                float(args.save_best_w_syn),
                float(args.save_best_w_hst),
                float(args.save_best_w_anchor),
            ),
            lane_counts=lane_counts,
            compute_resume_baseline=bool(args.resume_baseline),
        )

    runtime = time.time() - t0
    print(f"\nRUNTIME_SECONDS={runtime:.1f}")
    print(f"STEPS_TRAINED={args.steps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
