#!/usr/bin/env python3
"""Evaluate the lens-isolation ensemble on its fixed test split."""

from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.experiments.lens_isolation.config import ExperimentPaths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", default=ExperimentPaths().records)
    parser.add_argument("--ensemble-dir", default=ExperimentPaths().ensemble)
    parser.add_argument("--out-dir", default=ExperimentPaths().evaluation)
    parser.add_argument("--no-source-baselines", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(vars(args), sort_keys=True))
        return 0
    from euclid_polish.experiments.lens_isolation.evaluation import (
        evaluate_records,
        write_report,
    )

    metrics, rows = evaluate_records(
        args.ensemble_dir,
        args.records_dir,
        include_sources=not args.no_source_baselines,
    )
    paths = write_report(args.out_dir, metrics, rows)
    print(json.dumps(paths, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
