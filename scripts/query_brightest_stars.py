#!/usr/bin/env python
"""Query the N brightest Euclid stars + tally cutout integrity.

Non-interactive CLI wrapper around
:meth:`euclid_polish.euclid.catalog.StarCatalog.query_brightest_stars`.
Sorts ``mer_catalogue`` by VIS flux server-side (ESA Euclid archive),
writes the result into ``$DATA_DIR/euclid_stars/stars.csv``, then scans
any cutouts already on disk for per-band valid/corrupted counts.

It is designed to run on the **FASRC login node** (driven over SSH from
the laptop web UI) so the catalog lands on the shared netscratch
``$DATA_DIR`` — the same filesystem the cutout-download / PSF-extraction
SLURM jobs read. It is a quick archive query, not a SLURM job.

Usage::

    python scripts/query_brightest_stars.py --num-stars 200
    python scripts/query_brightest_stars.py --num-stars 500 \\
        --magnitude-min 17 --magnitude-limit 21
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Any, Dict, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.euclid.catalog import StarCatalog
from euclid_polish.euclid.validator import FitsValidator
from euclid_polish.observability.reporter import Reporter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-stars", type=int, default=Config.DEFAULT_BRIGHTEST_N,
                   help="Number of brightest stars to keep (server-side TOP N).")
    p.add_argument("--magnitude-min", type=float, default=None,
                   help="Bright-end cutoff (mag): brighter stars rejected. "
                        "≈ 15 skips stars that saturate the NISP detector.")
    p.add_argument("--magnitude-limit", type=float, default=None,
                   help="Faint-end cutoff (mag): dimmer stars rejected.")
    p.add_argument("--output-dir", default=Config.DEFAULT_OUTPUT_DIR,
                   help="Star catalog root (stars.csv + cutouts/ live here).")
    return p.parse_args()


def _integrity_tally(output_dir: str, reporter: Reporter) -> Dict[str, Any]:
    """Per-band valid/corrupted FITS counts under ``output_dir/cutouts``.

    Mirrors the old web ``_job_check_integrity``: a chunk on disk is the
    only place corruption shows up, so we re-scan whatever cutouts exist.
    A fresh catalog (no downloads yet) reports every band ``absent``.
    """
    validator = FitsValidator()
    cutout_root = os.path.join(output_dir, "cutouts")
    summary: Dict[str, Dict[str, Any]] = {}
    for name in Config.LR_INPUT_BAND_NAMES:
        band_dir = Config.cutout_dir_for_band(name, root=cutout_root)
        if not os.path.isdir(band_dir):
            summary[name] = {"valid": 0, "corrupted": 0, "absent": True}
            print(f"{name}: (no cutouts on disk yet)")
            continue
        files = glob.glob(os.path.join(band_dir, "*.fits"))
        ok = bad = 0
        for f in files:
            is_valid, _ = validator.validate_basic_integrity(f)
            if is_valid:
                ok += 1
            else:
                bad += 1
        summary[name] = {"valid": ok, "corrupted": bad, "absent": False}
        if bad:
            reporter.warn(f"{name}: {bad} corrupted cutout(s)")
        print(f"{name}: valid={ok}, corrupted={bad}, total={len(files)}")
    return summary


def main() -> int:
    args = parse_args()
    reporter = Reporter.from_env()

    window = []
    if args.magnitude_min is not None:   window.append(f"mag>{args.magnitude_min}")
    if args.magnitude_limit is not None: window.append(f"mag<{args.magnitude_limit}")
    win_str = (" [" + ", ".join(window) + "]") if window else ""
    print("=" * 60)
    print(f"  Query brightest {args.num_stars} stars{win_str}")
    print(f"  output dir = {args.output_dir}")
    print("=" * 60)

    t0 = time.perf_counter()
    reporter.set_stage("querying Euclid archive")
    cat = StarCatalog(args.output_dir)
    result = cat.query_brightest_stars(
        num_stars=args.num_stars,
        magnitude_limit=args.magnitude_limit,
        magnitude_min=args.magnitude_min,
    )
    print(result["message"])
    if "Query failed" in str(result.get("message", "")):
        reporter.error(str(result["message"]))

    reporter.set_stage("integrity tally")
    try:
        result["integrity"] = _integrity_tally(args.output_dir, reporter)
    except Exception as e:  # never let an integrity hiccup fail the query
        print(f"integrity tally skipped: {type(e).__name__}: {e}")

    print(f"\nRUNTIME_SECONDS={time.perf_counter() - t0:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
