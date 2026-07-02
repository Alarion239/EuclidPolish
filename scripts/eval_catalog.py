#!/usr/bin/env python
"""Run the SR model over a catalog of real targets (local CLI).

Thin wrapper over :func:`euclid_polish.eval.catalog_runner.run_catalog_eval` —
the same loop the WebUI's local "Run evaluation" job uses. Fetches a 4-band
Euclid cutout at every catalog (RA, Dec), runs the model, and writes per-object
FITS (``SR.fits`` + ``original_stack.fits``) + ``manifest.csv``. PNG rendering
is left to the gallery (local, on demand).

The headline catalog is the Natalie Lines Euclid Q1 strong-lens catalog
(``scripts/fetch_lens_catalog.py``); positions with no archive coverage are
logged and skipped rather than aborting the run.

Usage::

    python scripts/eval_catalog.py --grade A --max-n 20
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.eval import catalog_runner


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=None,
                   help="normalized id,ra,dec[,grade] catalog CSV (default: "
                        "the lens catalog under Config.EVAL_CATALOG_DIR)")
    p.add_argument("--out", default=None,
                   help="output dir (default: Config.EVAL_RESULTS_DIR)")
    p.add_argument("--run-name", default="run",
                   help="deprecated; evaluation now accumulates in Config.EVAL_RESULTS_DIR")
    p.add_argument("--ensemble-dir", default=None,
                   help="ensemble base dir (default: <ckpt parent>/ensemble)")
    p.add_argument("--num-res-blocks", type=int,
                   default=Config.DEFAULT_NUM_RES_BLOCKS)
    p.add_argument("--cutout-size", type=int, default=256)
    p.add_argument("--grade", default=None)
    p.add_argument("--max-n", type=int, default=0)
    p.add_argument("--asinh-scale", type=float, default=None)
    p.add_argument("--render", action="store_true",
                   help="also write eye/solar PNGs (the gallery renders these "
                        "locally on demand, so off by default)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    out_dir = args.out or Config.EVAL_RESULTS_DIR
    catalog_runner.run_catalog_eval(
        out_dir=out_dir,
        catalog_path=args.catalog,
        ensemble_dir=args.ensemble_dir,
        num_res_blocks=args.num_res_blocks,
        cutout_size=args.cutout_size,
        grade=args.grade,
        max_n=(args.max_n or None),
        asinh_scale=args.asinh_scale,
        render=args.render,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
