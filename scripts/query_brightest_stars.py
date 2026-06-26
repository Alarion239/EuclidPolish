#!/usr/bin/env python
"""Query the N brightest Euclid stars.

Non-interactive CLI wrapper around
:meth:`euclid_polish.catalog.star_catalog.StarCatalog.query_brightest_stars`.
Sorts ``mer_catalogue`` by VIS flux server-side (ESA Euclid archive) and
writes the result into ``$DATA_DIR/euclid_stars/stars.csv``.

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
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.catalog.star_catalog import StarCatalog
from euclid_polish.observability.reporter import Reporter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-stars", type=int, default=Config.DEFAULT_BRIGHTEST_N,
                   help="Number of brightest stars to keep (server-side TOP N).")
    p.add_argument("--magnitude-min", type=float, default=None,
                   help="Bright-end cutoff (AB mag): brighter stars rejected. "
                        "Use to skip stars that saturate (especially NISP). "
                        "Magnitudes are now proper AB from flux_vis_psf "
                        "(µJy), so this differs from the old aperture scale.")
    p.add_argument("--magnitude-limit", type=float, default=None,
                   help="Faint-end cutoff (AB mag): dimmer stars rejected.")
    p.add_argument("--snr-min", type=float, default=None,
                   help="Keep only well-measured stars: PSF "
                        "flux_vis_psf / fluxerr_vis_psf ≥ this (e.g. 50). "
                        "Off by default.")
    p.add_argument("--allow-masked", action="store_true",
                   help="Drop the default det_quality_flag=0 cut and keep "
                        "stars touched by any mask (saturation, blending, "
                        "bright-star masks, …). By default only mask-free "
                        "stars are kept — the clean set for ePSF construction.")
    p.add_argument("--output-dir", default=Config.DEFAULT_OUTPUT_DIR,
                   help="Star catalog root (stars.csv + cutouts/ live here).")
    return p.parse_args()


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
        snr_min=args.snr_min,
        require_unmasked=not args.allow_masked,
    )
    print(result["message"])
    if "Query failed" in str(result.get("message", "")):
        reporter.error(str(result["message"]))

    print(f"\nRUNTIME_SECONDS={time.perf_counter() - t0:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
