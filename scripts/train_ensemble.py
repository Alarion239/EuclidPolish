#!/usr/bin/env python
"""Train a seed-diverse ensemble of WDSR models, then evaluate it on the test set.

Members are trained sequentially into ``<base-dir>/member_NN/`` on the SAME
train/validate TFRecords, each with a distinct seed (member i → ``base_seed+i``;
the seed is recorded on the member's Process.training provenance). The ensemble
mean is the prediction; the per-member spread is the hallucination signal.

Submit on FASRC like the other steps (one GPU, long wall-clock for N members):
    sbatch scripts/fasrc_train_only.sh    # after adapting the python line, or
    python -u scripts/train_ensemble.py --n-members 5 --steps 200000
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config  # noqa: E402
from euclid_polish.ensemble import EnsembleModel, evaluate_on_records  # noqa: E402
from euclid_polish.image.tfio import tfrecord_path  # noqa: E402


def _default_base_dir() -> str:
    parent = os.path.dirname(Config.DEFAULT_CHECKPOINT_DIR.rstrip("/")) or "."
    return os.path.join(parent, "ensemble")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", default=None,
                   help="Ensemble root holding member_NN/ (default: "
                        "<ckpt parent>/ensemble).")
    p.add_argument("--records-dir", default=Config.RECORDS_DIR_V2)
    p.add_argument("--n-members", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=-1,
                   help="Member i is seeded base_seed+i. -1 (default) draws a "
                        "fresh entropy base seed; the value used is recorded on "
                        "each member's provenance for replay.")
    p.add_argument("--steps", type=int, default=Config.DEFAULT_TRAIN_STEPS)
    p.add_argument("--batch-size", type=int, default=Config.DEFAULT_BATCH_SIZE)
    p.add_argument("--evaluate-every", type=int, default=Config.DEFAULT_EVALUATE_EVERY)
    p.add_argument("--num-res-blocks", type=int, default=Config.DEFAULT_NUM_RES_BLOCKS)
    p.add_argument("--eval-images", type=int, default=200,
                   help="Test fields to score after training (0 = skip eval).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = args.base_dir or _default_base_dir()
    lr = tfrecord_path(args.records_dir, "dirty_train")
    hr = tfrecord_path(args.records_dir, "clean_train")
    if not (os.path.exists(lr) and os.path.exists(hr)):
        print(f"✗ training records not found in {args.records_dir} "
              "(dirty_train / clean_train). Generate them first.")
        return 2

    base_seed = None if args.base_seed < 0 else int(args.base_seed)
    print(f"Training {args.n_members}-member ensemble → {base}  "
          f"(base_seed={'entropy' if base_seed is None else base_seed})")

    ens = EnsembleModel(base, scale=Config.DEFAULT_REBIN_FACTOR,
                        num_res_blocks=args.num_res_blocks)
    ens.train(
        lr, hr,
        n_members=args.n_members, base_seed=base_seed,
        steps=args.steps, batch_size=args.batch_size,
        evaluate_every=args.evaluate_every,
        on_member=lambda i, n, _m: print(f"\n=== member {i}/{n} done ===\n"),
    )

    if args.eval_images > 0:
        print("\nEvaluating ensemble on the held-out test set...")
        out = evaluate_on_records(base, args.records_dir,
                                  num_images=args.eval_images)
        print(f"  subset:            {out['subset']}  ({out['n_scored']} scored)")
        print(f"  ensemble PSNR:     {out['ensemble_psnr']:.3f} dB")
        print(f"  mean member PSNR:  {out['mean_member_psnr']:.3f} dB")
        print(f"  ensemble gain:     {out['ensemble_gain_db']:+.3f} dB")
        d = out["disagreement"]
        print(f"  mean disagreement: {d['mean_std_e']:.4g} e⁻  "
              f"({d['frac_flux_hallucinated'] * 100:.1f}% of flux hallucinated)")
    print("\n✓ Ensemble training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
