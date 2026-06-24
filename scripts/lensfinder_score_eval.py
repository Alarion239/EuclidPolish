#!/usr/bin/env python
"""Score evaluation objects with the trained lens-finder heads (torch env).

Runs the per-reconstruction lens classifiers over a shared-store eval run and
writes ``lens_scores.csv`` — ``P(lens)`` for each object's LR (before), SR
(after) and HR renders, using the matching head (the LR head scores the LR
render, etc., as in training). The ``/evaluation`` page then folds these in:
PCA marker opacity ∝ P(lens), plus the lens-identification analysis panel.

Runs in the EuclidPolishZoobot env. Pairs with ``scripts/lensfinder_train.py``
(which produced ``<heads>/{lr,sr,hr}/checkpoints/*.ckpt``).

Usage (EuclidPolishZoobot env)::

    python scripts/lensfinder_score_eval.py \
        --run-dir data/eval_results --heads-dir data/lensfinder/heads
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.eval import zoobot_morph as zm
from euclid_polish.lensfinder import stamps as lf_stamps
from euclid_polish.observability.reporter import Reporter

#: (catalog recon key, discover_objects view) — head ``recon`` scores its render.
_RECON_VIEWS = (("lr", "before"), ("sr", "after"), ("hr", "hr"))
_PRED_COLS = ["p_notlens", "p_lens"]


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None,
                   help="eval run dir (default: Config.EVAL_RESULTS_DIR)")
    p.add_argument("--heads-dir", required=True,
                   help="lens-finder heads dir with {lr,sr,hr}/checkpoints/*.ckpt")
    p.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu", "mps"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--png-size", type=int, default=424)
    p.add_argument("--stamp-m", type=int, default=106,
                   help="HR-grid crop size; MUST match build_stamps (LR uses m//2)")
    p.add_argument("--lupton-q", type=float, default=8.0)
    p.add_argument("--rgb-scale-r", type=float, default=1.0)
    p.add_argument("--rgb-scale-g", type=float, default=1.0)
    p.add_argument("--rgb-scale-b", type=float, default=1.0)
    p.add_argument("--asinh-scale", type=float, default=None)
    p.add_argument("--max-objects", type=int, default=0, help="cap (0 = all)")
    return p.parse_args(argv)


def _newest_ckpt(heads_dir: str, recon: str):
    cks = glob.glob(os.path.join(heads_dir, recon, "checkpoints", "*.ckpt"))
    return max(cks, key=os.path.getmtime) if cks else None


def main(argv=None) -> int:
    args = _parse_args(argv)
    reporter = Reporter.from_env()
    run_dir = args.run_dir or Config.EVAL_RESULTS_DIR
    asinh = float(args.asinh_scale or Config.STRETCH_SCALE_E)
    m = int(args.stamp_m)
    m += m % 2                                  # even → integer LR half-crop

    objects = zm.discover_objects(run_dir)
    if args.max_objects:
        objects = objects[:args.max_objects]
    grade_of = zm.read_grade_map(run_dir)
    print(f"run {run_dir}: {len(objects)} object(s)")
    if not objects:
        print("no objects with SR.fits found")
        return 0

    import pandas as pd
    from galaxy_datasets.transforms import (get_galaxy_transform,
                                            minimal_view_config)
    from zoobot.pytorch.training import finetune
    from zoobot.pytorch.predictions import predict_on_catalog

    cfg = minimal_view_config()
    cfg.output_size = (args.png_size, args.png_size)
    transform = get_galaxy_transform(cfg)

    scores: dict = {recon: {} for recon, _ in _RECON_VIEWS}
    for recon, view in _RECON_VIEWS:
        ckpt = _newest_ckpt(args.heads_dir, recon)
        if ckpt is None:
            print(f"  · {recon}: no checkpoint under {args.heads_dir}/{recon} — skipping")
            continue
        reporter.set_stage(f"scoring {recon} ({view})")
        rows = []
        for i, obj in enumerate(objects):
            src = obj.get(view)
            if not src:
                continue
            png = os.path.join(obj["dir"], "lensfinder", f"{view}.png")
            crop_m = m // 2 if recon == "lr" else m
            lf_stamps.render_eval_stamp(
                src, png, crop_m=crop_m, stretch=asinh, Q=args.lupton_q,
                scale_r=args.rgb_scale_r, scale_g=args.rgb_scale_g,
                scale_b=args.rgb_scale_b, size=args.png_size)
            rows.append({"id_str": obj["id"], "file_loc": png})
            reporter.set_step(i + 1, len(objects), f"{recon} {obj['id']}")
        if not rows:
            continue
        model = finetune.FinetuneableZoobotClassifier.load_from_checkpoint(ckpt)
        preds = predict_on_catalog.predict(
            catalog=pd.DataFrame(rows), model=model, label_cols=_PRED_COLS,
            inference_transform=transform,
            save_loc=os.path.join(run_dir, f"_lensfinder_pred_{recon}.csv"),
            datamodule_kwargs={"batch_size": args.batch_size,
                               "num_workers": args.num_workers},
            trainer_kwargs={"accelerator": args.device, "devices": 1})
        for _, r in preds.iterrows():
            try:
                scores[recon][str(r["id_str"])] = float(r["p_lens"])
            except (TypeError, ValueError, KeyError):
                pass
        print(f"  ✓ {recon}: scored {len(scores[recon])} object(s)")

    reporter.set_stage("writing lens_scores.csv")
    out_csv = os.path.join(run_dir, "lens_scores.csv")
    cols = ["id", "grade", "p_lens_lr", "p_lens_sr", "p_lens_hr"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for obj in objects:
            oid = obj["id"]
            row = {"id": oid, "grade": grade_of.get(oid, "")}
            for recon, _ in _RECON_VIEWS:
                v = scores[recon].get(oid)
                row[f"p_lens_{recon}"] = "" if v is None else v
            w.writerow(row)
    n_sr = sum(1 for o in objects if o["id"] in scores["sr"])
    print(f"\n✓ lens scores for {n_sr}/{len(objects)} objects → {out_csv}")
    reporter.metric({"objects": len(objects), "scored_sr": n_sr})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
