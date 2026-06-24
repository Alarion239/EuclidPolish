#!/usr/bin/env python
"""Prepare the unified grouped eval dataset locally, with a progress bar.

Thin CLI over :func:`euclid_polish.eval.grouped_runner.run_grouped_analysis` —
the same loop the WebUI's "Run grouped" job uses, but run in the terminal with a
``tqdm`` progress bar (the WebUI drives its own bar; locally there was none).

Prepares ``n`` real lens cutouts per grade (A/B/C) plus source-centered
synthetic stamps (syn-lens / syn-gal) into one run dir with a single
``manifest.csv``. Every object is held at the canonical eval geometry — 53² LR
beside 106² SR/HR — so there are no size knobs; a stamp smaller than that is
dropped. Real cutouts can be reused from an existing run dir (``--lens-source``)
instead of re-downloaded.

Usage (in the EuclidPolishEnv env)::

    python scripts/eval_grouped.py --n 100 \
        --lens-source data/eval_results --out data/eval_zoobot500
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.eval import grouped_runner


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None,
                   help="output run dir (default: Config.EVAL_RESULTS_DIR)")
    p.add_argument("--n", type=int, default=100,
                   help="cutouts per class (per A/B/C grade and per synthetic "
                        "subgroup)")
    p.add_argument("--cutout-size", type=int, default=256,
                   help="real-lens download size in VIS px (center-cropped to "
                        "the canonical 53² LR / 106² SR afterwards)")
    p.add_argument("--lens-source", default=None,
                   help="reuse cached real-lens FITS from this run dir instead "
                        "of re-downloading (they are merely re-cropped)")
    p.add_argument("--no-synthetic", action="store_true",
                   help="skip the syn-lens / syn-gal subgroups")
    p.add_argument("--unique-fields", action="store_true",
                   help="use each synthetic field for at most one subgroup "
                        "(default: allow a field to host both a syn-lens and a "
                        "syn-gal stamp, maximizing yield)")
    p.add_argument("--catalog", default=None,
                   help="lens catalog CSV (default: the bundled lens catalog)")
    p.add_argument("--checkpoint", default=Config.DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    out_dir = args.out or Config.EVAL_RESULTS_DIR

    res = grouped_runner.run_grouped_analysis(
        out_dir, n=args.n,
        cutout_size=args.cutout_size,
        catalog_path=args.catalog,
        checkpoint=args.checkpoint,
        seed=args.seed,
        include_synthetic=not args.no_synthetic,
        lens_source_dir=args.lens_source,
        unique_fields=args.unique_fields,
    )
    print(f"\ngroups: {res.get('groups')}  → {res.get('manifest')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
