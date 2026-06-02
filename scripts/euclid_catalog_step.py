#!/usr/bin/env python
"""Unified Euclid-catalog step: query the brightest stars and/or verify the
photometry scale, in ONE SLURM job.

``--mode``:
  * ``query``  — run ``query_brightest_stars.py`` only (writes ``stars.csv``).
  * ``verify`` — run ``verify_star_photometry.py`` only (needs cutouts on disk;
                 reports the measured/catalog electron ratio).
  * ``both``   — query, then verify against whatever cutouts already exist
                 (default; on a fresh catalog with no cutouts the verify pass
                 simply reports nothing and exits 0).

Thin orchestrator: it subprocesses the two existing scripts so each stays the
single source of truth for its own logic. Runs on a FASRC compute node — the
``shared`` partition has outbound internet for the ESA TAP query (the cutout
downloader runs there too), and both the catalog and the report land on the
shared netscratch ``$DATA_DIR`` / the sbatch log.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["query", "verify", "both"], default="both",
                   help="Which phase(s) to run. Default both.")
    # Query phase (forwarded to query_brightest_stars.py).
    p.add_argument("--num-stars", type=int, default=200)
    p.add_argument("--magnitude-min", type=float, default=None)
    p.add_argument("--magnitude-limit", type=float, default=None)
    p.add_argument("--snr-min", type=float, default=None)
    # Verify phase (forwarded to verify_star_photometry.py).
    p.add_argument("--verify-n", type=int, default=40,
                   help="Stars to aperture-measure in the verify pass.")
    p.add_argument("--verify-size", type=int, default=256,
                   help="Cutout side (px) to read in the verify pass; must "
                        "match the downloaded cutouts.")
    return p.parse_args()


def _run(argv: list[str]) -> int:
    """Run one phase script in-process-of-the-job as a subprocess; stream its
    output to our stdout (the sbatch log). Returns its exit code."""
    full = [sys.executable, "-u", *argv]
    print(f"\n{'=' * 60}\n>>> {' '.join(argv)}\n{'=' * 60}", flush=True)
    return subprocess.run(full, cwd=_PROJECT_ROOT).returncode


def main() -> int:
    args = parse_args()

    if args.mode in ("query", "both"):
        argv = ["scripts/query_brightest_stars.py",
                "--num-stars", str(args.num_stars)]
        if args.magnitude_min is not None:
            argv += ["--magnitude-min", f"{args.magnitude_min:g}"]
        if args.magnitude_limit is not None:
            argv += ["--magnitude-limit", f"{args.magnitude_limit:g}"]
        if args.snr_min is not None:
            argv += ["--snr-min", f"{args.snr_min:g}"]
        rc = _run(argv)
        if rc != 0:
            print(f"query phase failed (rc={rc}); skipping verify.")
            return rc

    if args.mode in ("verify", "both"):
        rc = _run(["scripts/verify_star_photometry.py",
                   "--n", str(args.verify_n), "--size", str(args.verify_size)])
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
