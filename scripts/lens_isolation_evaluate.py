#!/usr/bin/env python3
"""Evaluate lens-isolation members on fixed random held-out cutouts."""

from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.experiments.lens_isolation.config import ExperimentPaths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", default=ExperimentPaths().records)
    parser.add_argument("--ensemble-dir", default=ExperimentPaths().ensemble)
    parser.add_argument("--out-dir", default=ExperimentPaths().evaluation)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=Config.DEFAULT_HR_CROP_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.crop_size < 1 or args.crop_size % Config.DEFAULT_REBIN_FACTOR:
        raise ValueError("crop-size must be positive and divisible by the HR/LR scale")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be >= 1")
    plan = {
        "experiment": "lens_isolation",
        "records_dir": os.path.abspath(args.records_dir),
        "ensemble_dir": os.path.abspath(args.ensemble_dir),
        "out_dir": os.path.abspath(args.out_dir),
        "seed": args.seed,
        "crop_size": args.crop_size,
        "limit": args.limit,
    }
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return 0

    from euclid_polish.experiments.lens_isolation.evaluation import evaluate_records, write_report
    from euclid_polish.observability import Reporter, ResourceSampler

    reporter = Reporter.from_env()
    reporter.set_stage("lens isolation: evaluate random held-out cutouts")
    sampler = ResourceSampler(reporter).start()
    try:
        metrics, rows = evaluate_records(
            args.ensemble_dir,
            args.records_dir,
            seed=args.seed,
            crop_size=args.crop_size,
            limit=args.limit,
        )
        reporter.metric({"approaches": len(metrics), "cutouts": len(rows)})
        paths = write_report(args.out_dir, metrics, rows)
    finally:
        sampler.stop()
    print(json.dumps(paths, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
