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
import shutil
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config  # noqa: E402
from euclid_polish.ensemble import EnsembleModel, evaluate_on_records  # noqa: E402
from euclid_polish.image.tfio import tfrecord_path  # noqa: E402
from euclid_polish.observability import Reporter, ResourceSampler  # noqa: E402
from euclid_polish.training.staging import stage_records  # noqa: E402


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
    # LR schedule (warmup → cosine) + reduce-LR-on-plateau guard. Defaults from
    # Config; the WebUI /config page injects these on FASRC submission.
    p.add_argument("--lr-peak", type=float, default=Config.LR_PEAK)
    p.add_argument("--lr-final", type=float, default=Config.LR_FINAL)
    p.add_argument("--lr-warmup-steps", type=int, default=Config.LR_WARMUP_STEPS)
    p.add_argument("--plateau-lr-enabled", type=int,
                   default=int(Config.PLATEAU_LR_ENABLED),
                   help="1/0 — reduce LR when the val metric stalls.")
    p.add_argument("--plateau-lr-factor", type=float,
                   default=Config.PLATEAU_LR_FACTOR)
    p.add_argument("--plateau-lr-patience", type=int,
                   default=Config.PLATEAU_LR_PATIENCE)
    p.add_argument("--plateau-lr-min-delta", type=float,
                   default=Config.PLATEAU_LR_MIN_DELTA)
    p.add_argument("--plateau-lr-cooldown", type=int,
                   default=Config.PLATEAU_LR_COOLDOWN)
    p.add_argument("--plateau-lr-min-lr", type=float,
                   default=Config.PLATEAU_LR_MIN_LR)
    p.add_argument("--plateau-lr-metric", default=Config.PLATEAU_LR_METRIC,
                   choices=["combined_loss", "psnr_stretched"])
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

    # Structured progress for the WebUI (no terminal for tqdm under SLURM).
    # One cumulative bar across all members × steps.
    reporter = Reporter.from_env()
    reporter.set_stage(f"training {args.n_members}-member ensemble")
    # Sample CPU + GPU utilisation every 10 s into the events stream so the job
    # log records gpu_util_mean/peak (the "GPU util" column). Daemon thread;
    # stopped after the eval below.
    sampler = ResourceSampler(reporter).start()

    # Stage the hot train + validate records onto node-local scratch (fast NVMe,
    # no shared-netscratch contention — the input-pipeline stall behind the
    # 48s-vs-100s/1000-step variance). Best-effort: falls back to the original
    # dir if local scratch is missing/too small. Test records stay on netscratch
    # (read once by the post-training eval, not the hot loop).
    def _stage_log(m: str) -> None:
        print(m)
        if m.lstrip().startswith("⚠"):      # surface staging fallbacks to the WebUI
            reporter.warn(m.strip())

    records_dir = stage_records(
        args.records_dir,
        ["dirty_train", "clean_train", "dirty_validate", "clean_validate"],
        on_log=_stage_log,
    )
    staged = records_dir != args.records_dir
    lr = tfrecord_path(records_dir, "dirty_train")
    hr = tfrecord_path(records_dir, "clean_train")

    total = max(1, args.n_members) * max(1, args.steps)
    done_members = [0]                          # members finished so far

    def _step_cb(s: int, _t: int) -> None:
        reporter.set_step(done_members[0] * args.steps + s, total,
                          f"member {done_members[0] + 1}/{args.n_members} · step {s}")

    def _eval_cb(row: dict) -> None:
        # Forward per-member eval metrics to the WebUI's structured stream as
        # ONE cumulative curve: offset the step by members already finished so
        # the validation history is continuous across members (the trainer only
        # knows its own member-local step). Tag the member for the UI.
        m = done_members[0]
        cum = dict(row)
        if cum.get("step") is not None:
            cum["step"] = m * args.steps + int(cum["step"])
        cum["total"] = total
        cum["member"] = m + 1
        reporter.metric(cum)

    def _warn_cb(msg: str) -> None:
        # Surface restore/resume notices, gradient-spike rollbacks and LR
        # halvings (which otherwise only reached stdout) to the WebUI.
        reporter.warn(f"member {done_members[0] + 1}/{args.n_members}: {msg}")

    def _on_member(i: int, n: int, _m) -> None:
        done_members[0] = i                     # ``i`` is 1-indexed (members done)
        print(f"\n=== member {i}/{n} done ===\n")
        reporter.set_step(i * args.steps, total, f"member {i}/{n} done")

    ens = EnsembleModel(base, scale=Config.DEFAULT_REBIN_FACTOR,
                        num_res_blocks=args.num_res_blocks)
    try:
        ens.train(
            lr, hr,
            n_members=args.n_members, base_seed=base_seed,
            steps=args.steps, batch_size=args.batch_size,
            evaluate_every=args.evaluate_every,
            step_callback=_step_cb, eval_callback=_eval_cb, warn_callback=_warn_cb,
            on_member=_on_member,
            # LR schedule + plateau guard (forwarded to each member's Model.train).
            lr_peak=args.lr_peak, lr_final=args.lr_final,
            lr_warmup_steps=args.lr_warmup_steps,
            plateau_lr_enabled=bool(args.plateau_lr_enabled),
            plateau_lr_factor=args.plateau_lr_factor,
            plateau_lr_patience=args.plateau_lr_patience,
            plateau_lr_min_delta=args.plateau_lr_min_delta,
            plateau_lr_cooldown=args.plateau_lr_cooldown,
            plateau_lr_min_lr=args.plateau_lr_min_lr,
            plateau_lr_metric=args.plateau_lr_metric,
        )

        if args.eval_images > 0:
            print("\nEvaluating ensemble on the held-out test set...")
            # Test records read once here — keep them on the original dir.
            out = evaluate_on_records(base, args.records_dir,
                                      num_images=args.eval_images)
            print(f"  subset:            {out['subset']}  ({out['n_scored']} scored)")
            print(f"  ensemble PSNR:     {out['ensemble_psnr']:.3f} dB")
            print(f"  mean member PSNR:  {out['mean_member_psnr']:.3f} dB")
            print(f"  ensemble gain:     {out['ensemble_gain_db']:+.3f} dB")
            d = out["disagreement"]
            print(f"  mean disagreement: {d['mean_std_e']:.4g} e⁻  "
                  f"({d['frac_flux_hallucinated'] * 100:.1f}% of flux hallucinated)")
    finally:
        sampler.stop()
        if staged:                          # remove the node-local staged copy
            shutil.rmtree(records_dir, ignore_errors=True)
    print("\n✓ Ensemble training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
