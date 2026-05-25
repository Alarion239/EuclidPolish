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

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps",        type=int,   default=Config.DEFAULT_TRAIN_STEPS)
    p.add_argument("--batch-size",   type=int,   default=Config.DEFAULT_BATCH_SIZE)
    p.add_argument("--hst-fraction", type=float, default=0.10,
                   help="Fraction of each batch drawn from the HST "
                        "TFRecord source (0 → identical to current "
                        "training, 1 → only HST).")
    p.add_argument("--roundtrip-fraction", type=float, default=0.0,
                   help="Fraction of each batch drawn from the real "
                        "Euclid round-trip records "
                        "($DATA_DIR/images/records_v2_euclid_roundtrip "
                        "by default). When > 0, the trainer mixes a "
                        "self-supervised reconstruction loss "
                        "|asinh(Conv(M(lr))/k) - lr_vis| into the "
                        "batch for those examples. Requires the VIS "
                        "PSF FITS at "
                        "$DATA_DIR/euclid_psf/euclid_psf_VIS.fits "
                        "(loaded into a TF-graph forward op). "
                        "``hst_fraction + roundtrip_fraction`` must "
                        "be ≤ 1; the remainder is synthetic. Default "
                        "0 keeps pre-round-trip behaviour bit-for-bit.")
    p.add_argument("--records-syn",  default=Config.RECORDS_DIR_V2,
                   help="Synthetic TFRecord directory.")
    p.add_argument("--records-hst",
                   default=os.path.join(Config.DATA_DIR, "images", "records_v2_hst"),
                   help="HST-derived TFRecord directory (built by "
                        "fasrc_generate_hst_tfrecords.py).")
    p.add_argument("--records-roundtrip",
                   default=os.path.join(
                       Config.DATA_DIR, "images", "records_v2_euclid_roundtrip"),
                   help="Round-trip Euclid TFRecord directory (built "
                        "by fasrc_generate_euclid_roundtrip_tfrecords.py).")
    p.add_argument("--roundtrip-loss-weight", type=float, default=1.0,
                   help="Multiplier on the per-example round-trip "
                        "loss before averaging with the supervised "
                        "loss. Default 1.0; bump to up-weight the "
                        "round-trip path, set to 0 for an ablation "
                        "(round-trip data still loaded, but loss "
                        "contribution zeroed out).")
    p.add_argument("--ckpt-dir",     default=Config.DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--num-res-blocks", type=int, default=Config.DEFAULT_NUM_RES_BLOCKS)
    p.add_argument("--evaluate-every", type=int, default=Config.DEFAULT_EVALUATE_EVERY)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 64)
    print(f"  WDSR training with HST + round-trip mix")
    print("=" * 64)
    print(f"  steps              = {args.steps}")
    print(f"  batch size         = {args.batch_size}")
    print(f"  HST fraction       = {args.hst_fraction}")
    print(f"  round-trip fraction= {args.roundtrip_fraction}")
    print(f"  synthetic records  = {args.records_syn}")
    print(f"  HST records        = {args.records_hst}")
    print(f"  round-trip records = {args.records_roundtrip}")
    print(f"  rt loss weight     = {args.roundtrip_loss_weight}")
    print(f"  checkpoint dir     = {args.ckpt_dir}")
    print()

    t0 = time.time()
    if args.dry_run:
        print("DRY RUN — would train.")
        print(f"\nRUNTIME_SECONDS={time.time() - t0:.1f}")
        return 0

    import numpy as np
    import tensorflow as tf
    from euclid_polish.training import Trainer
    from euclid_polish.training.data_multiband import MultiBandEuclidDataset
    from euclid_polish.training.forward_op import EuclidVISForwardOp
    from euclid_polish.training.models.wdsr import wdsr
    from tf_keras.losses import MeanAbsoluteError
    from tf_keras.optimizers.schedules import PiecewiseConstantDecay

    use_roundtrip = args.roundtrip_fraction > 0
    # The dataset needs source tags whenever the round-trip path is on.
    # When it's off we keep the pre-round-trip 2-tuple API so the
    # validation loop / existing callers don't change behaviour.
    needs_source_tag = use_roundtrip

    # Two dataset instances — one for training, one for validation. HST
    # and round-trip sources are only mixed into training; validation
    # stays pure synthetic so the metric is comparable across runs and
    # round-trip records (which lack HR ground truth) can't slip into
    # the PSNR computation.
    train_dataset = MultiBandEuclidDataset(
        subset="train",
        records_dir=args.records_syn,
        hst_records_dir=args.records_hst if args.hst_fraction > 0 else None,
        hst_fraction=float(args.hst_fraction),
        roundtrip_records_dir=args.records_roundtrip if use_roundtrip else None,
        roundtrip_fraction=float(args.roundtrip_fraction),
    ).dataset(
        batch_size=int(args.batch_size), random_transform=True,
        with_source_tag=needs_source_tag,
    )
    valid_dataset = MultiBandEuclidDataset(
        subset="validate",
        records_dir=args.records_syn,
    ).dataset(batch_size=int(args.batch_size), random_transform=False,
              repeat_count=1)

    # Model + loss + optimizer (same recipe as the standard trainer).
    scale = Config.DEFAULT_REBIN_FACTOR
    model = wdsr(
        scale=scale, num_res_blocks=args.num_res_blocks,
        nchan_in=Config.NUM_LR_CHANNELS, nchan_out=Config.NUM_HR_CHANNELS,
    )
    schedule = PiecewiseConstantDecay(
        boundaries=[args.steps // 2],
        values=[1e-3, 5e-4],
    )

    # Forward op only when the round-trip path is on. Loading + flipping
    # the kernel takes ~tens of ms at startup; nothing if we skip.
    forward_op = EuclidVISForwardOp(rebin_factor=scale) if use_roundtrip else None

    trainer = Trainer(
        model=model,
        loss=MeanAbsoluteError(),
        learning_rate=schedule,
        checkpoint_dir=args.ckpt_dir,
        forward_op=forward_op,
        roundtrip_loss_weight=float(args.roundtrip_loss_weight),
    )
    print(f"      total trainable params: "
          f"{sum(int(np.prod(v.shape)) for v in model.trainable_variables):,}")
    print(f"\n  training {args.steps} steps ...")
    trainer.train(
        train_dataset, valid_dataset, steps=int(args.steps),
        evaluate_every=int(args.evaluate_every),
        save_best_only=True,
    )

    runtime = time.time() - t0
    print(f"\nRUNTIME_SECONDS={runtime:.1f}")
    print(f"STEPS_TRAINED={args.steps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
