#!/usr/bin/env python
"""Batch super-resolution evaluation over a catalog of sky targets.

Given a normalized ``id,ra,dec[,grade]`` catalog (see
``scripts/fetch_lens_catalog.py`` for the Euclid Q1 strong-lens catalog), this
runs the model on a real 4-band Euclid cutout at every position and writes,
per object, the band FITS + ``SR.fits`` + ``eye.png`` / ``solar.png`` renders
plus a self-consistency residual, then aggregates a ``manifest.csv`` across the
whole run.

The per-object work is exactly the single-position WebUI inference path —
:func:`euclid_polish.web.helpers.jobs_impl.reconstruct_cutout_at` is the shared
body — so a manifest row matches what ``/inference/reconstruct-euclid`` would
produce at the same coordinates.

Positions with no archive coverage (the cutout service finds no covering MER
tile) are logged and skipped rather than aborting the batch.

Designed to run on FASRC under SLURM: progress is emitted via
:class:`euclid_polish.observability.reporter.Reporter` (the same JSONL stream
the WebUI job viewer polls). Locally, with no ``EUCLID_POLISH_EVENTS_PATH`` set,
the Reporter is a no-op and progress just prints.

Usage::

    python scripts/fasrc_eval_catalog.py \
        --catalog data/eval_catalogs/lens_catalog/lenses.csv \
        --out     data/eval_results/lenses_gradeA \
        --grade A --cutout-size 256 --max-n 50
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import traceback

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.euclid.eval_catalog import read_eval_catalog
from euclid_polish.observability.reporter import Reporter

# Columns written to manifest.csv, in order. The residual_* / flux_ratio_*
# keys come straight from reconstruct_cutout_at()'s metrics dict.
_METRIC_KEYS = (
    "residual_std_e",
    "residual_mae_e",
    "residual_rmse_e",
    "residual_max_abs_e",
    "residual_chi",
    "flux_ratio_fwd_over_lr",
)
_MANIFEST_COLS = (
    ["id", "ra", "dec", "grade", "ok", "error", "out_subdir"]
    + list(_METRIC_KEYS)
)

_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_id(obj_id: str) -> str:
    """Filesystem-safe per-object sub-directory name."""
    s = _SAFE_ID.sub("_", obj_id).strip("_")
    return s or "obj"


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--catalog", default=None,
                   help="normalized id,ra,dec[,grade] catalog CSV (default: "
                        "Config.EVAL_CATALOG_DIR/lens_catalog/lenses.csv)")
    p.add_argument("--out", default=None,
                   help="output directory for the run (default: "
                        "Config.EVAL_RESULTS_DIR/<run-name>). Holds one "
                        "subdir per object + manifest.csv")
    p.add_argument("--run-name", default="run",
                   help="run sub-directory name under Config.EVAL_RESULTS_DIR "
                        "when --out is not given")
    p.add_argument("--checkpoint", default=Config.DEFAULT_CHECKPOINT_DIR,
                   help="model checkpoint directory")
    p.add_argument("--num-res-blocks", type=int,
                   default=Config.DEFAULT_NUM_RES_BLOCKS,
                   help="WDSR residual-block count (must match the checkpoint)")
    p.add_argument("--cutout-size", type=int, default=256,
                   help="VIS cutout size in pixels (0.10″/pix)")
    p.add_argument("--grade", default=None,
                   help="keep only catalog rows with this grade (A/B/C)")
    p.add_argument("--max-n", type=int, default=0,
                   help="cap number of objects (0 = all)")
    p.add_argument("--asinh-scale", type=float, default=None,
                   help="asinh stretch knee for the renders (default: "
                        "Config.STRETCH_SCALE_E)")
    p.add_argument("--no-render", action="store_true",
                   help="skip the eye/solar PNG renders (FITS + metrics only)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    # Heavy imports (TF) deferred until after arg parsing so --help is fast.
    import tensorflow as tf
    from euclid_polish.training.inference import load_model_from_checkpoint
    from euclid_polish.web.helpers.jobs_impl import reconstruct_cutout_at

    reporter = Reporter.from_env()

    catalog = args.catalog or os.path.join(
        Config.EVAL_CATALOG_DIR, "lens_catalog", "lenses.csv")
    out_dir = args.out or os.path.join(Config.EVAL_RESULTS_DIR, args.run_name)

    # Make the job self-sufficient on the cluster: if the *default* lens
    # catalog isn't present on this node (it's only fetched into the local
    # data dir by the WebUI button / CLI, which doesn't reach FASRC), pull and
    # normalize it from Zenodo here. An explicitly-passed --catalog that's
    # missing is a user error, so we don't silently substitute the default.
    if not os.path.isfile(catalog):
        if args.catalog:
            raise FileNotFoundError(f"catalog not found: {catalog}")
        from euclid_polish.euclid import lens_catalog
        reporter.set_stage("fetching lens catalog")
        print(f"catalog {catalog} not found — fetching from Zenodo…")
        lens_catalog.fetch(catalog)

    rows = read_eval_catalog(catalog, grade=args.grade,
                             max_n=(args.max_n or None))
    n = len(rows)
    print(f"catalog {catalog}: {n} object(s)"
          + (f" (grade {args.grade})" if args.grade else ""))
    if n == 0:
        print("nothing to evaluate")
        return 0

    if not tf.train.latest_checkpoint(args.checkpoint):
        raise FileNotFoundError(f"no checkpoint in {args.checkpoint}")
    reporter.set_stage("loading model")
    model = load_model_from_checkpoint(
        args.checkpoint, Config.DEFAULT_REBIN_FACTOR, args.num_res_blocks,
        nchan_out=Config.NUM_HR_CHANNELS,   # nchan_in inferred from ckpt
    )

    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.csv")
    reporter.set_stage(f"evaluating {n} object(s)")

    n_ok = 0
    n_skip = 0
    with open(manifest_path, "w", newline="") as fmani:
        writer = csv.DictWriter(fmani, fieldnames=_MANIFEST_COLS)
        writer.writeheader()
        for i, row in enumerate(rows):
            obj_id = row["id"]
            sub = _safe_id(obj_id)
            reporter.set_step(i, n, f"{obj_id} ({i + 1}/{n})")
            print(f"[{i + 1}/{n}] {obj_id}  ra={row['ra']:.5f} "
                  f"dec={row['dec']:.5f}")
            rec: dict = {
                "id": obj_id, "ra": row["ra"], "dec": row["dec"],
                "grade": row["grade"] or "", "ok": False, "error": "",
                "out_subdir": sub,
            }
            rec.update({k: "" for k in _METRIC_KEYS})
            try:
                res = reconstruct_cutout_at(
                    model, row["ra"], row["dec"], args.cutout_size,
                    os.path.join(out_dir, sub),
                    asinh_scale=args.asinh_scale,
                    checkpoint_dir=args.checkpoint,
                    render=not args.no_render,
                )
                metrics = res["metrics"]
                for k in _METRIC_KEYS:
                    rec[k] = metrics.get(k)
                rec["ok"] = True
                n_ok += 1
            except Exception as e:  # noqa: BLE001 — one bad object must not kill the run
                msg = f"{type(e).__name__}: {e}"
                rec["error"] = msg
                n_skip += 1
                reporter.warn(f"{obj_id}: {msg}")
                print(f"  ! skipped: {msg}")
                traceback.print_exc()
            writer.writerow(rec)
            fmani.flush()

    reporter.set_step(n, n, "done")
    reporter.metric({"objects": n, "ok": n_ok, "skipped": n_skip})
    print(f"\n✓ done: {n_ok} ok, {n_skip} skipped → {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
