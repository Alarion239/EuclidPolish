#!/usr/bin/env python
"""Delete star cutouts that aren't valid in *every* band.

Keeps only the stars whose cutouts open cleanly in all
:data:`Config.BANDS` (VIS, Y_E, J_E, H_E) — a partial multiband set is
useless to the 4-channel pipeline / ePSF construction. Every cutout of any
star missing or corrupted in at least one band is removed from disk (all
bands, all sizes), and (by default) that star's row is dropped from
``stars.csv`` so the catalog stays consistent with disk.

By default it re-validates on disk first (opens every FITS) so the decision
reflects reality, not stale flags. Use ``--dry-run`` to preview.

    python scripts/clean_incomplete_cutouts.py --dry-run
    python scripts/clean_incomplete_cutouts.py
    python scripts/clean_incomplete_cutouts.py --output-dir /path/euclid_stars
    python scripts/clean_incomplete_cutouts.py --skip-validate --keep-catalog-rows
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.catalog.cutout_integrity import (
    purge_incomplete_cutouts, validate_all_cutouts,
)
from euclid_polish.observability.reporter import Reporter


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=Config.DEFAULT_OUTPUT_DIR,
                    help="Star-cutout root holding stars.csv + cutouts/<band>/.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be deleted without touching disk.")
    ap.add_argument("--skip-validate", action="store_true",
                    help="Trust the catalog's existing valid/corrupted flags "
                         "instead of re-opening every FITS first.")
    ap.add_argument("--keep-catalog-rows", action="store_true",
                    help="Delete the cutout files but keep the incomplete "
                         "stars' rows in stars.csv (default: drop them).")
    args = ap.parse_args()

    reporter = Reporter.from_env()
    catalog_path = os.path.join(args.output_dir, Config.CATALOG_FILE)
    if not os.path.exists(catalog_path):
        reporter.error(f"no catalog at {catalog_path}")
        print(f"✗ no catalog at {catalog_path}")
        return 1

    if not args.skip_validate:
        reporter.set_stage("validating cutouts (integrity)")
        v = validate_all_cutouts(args.output_dir, reporter=reporter)
        print(f"  validated {v['checked']} cutouts; {v['unopenable']} unopenable; "
              f"{v['valid_all_bands']} stars valid in all {v['n_bands']} bands")

    reporter.set_stage("purging incomplete cutouts"
                       + (" (dry run)" if args.dry_run else ""))
    s = purge_incomplete_cutouts(
        args.output_dir,
        dry_run=args.dry_run,
        drop_catalog_rows=not args.keep_catalog_rows,
        reporter=reporter,
    )

    verb = "would delete" if args.dry_run else "deleted"
    print(f"{'[dry run] ' if args.dry_run else ''}"
          f"✓ {s['complete_stars']} stars valid in all {s['n_bands']} bands kept; "
          f"{s['incomplete_stars']} incomplete stars → {verb} {s['deleted_files']} "
          f"cutout files"
          + (f"; dropped {s['dropped_rows']} catalog rows" if s['dropped_rows'] else "")
          + (f"; {s['failed_deletes']} deletes failed" if s['failed_deletes'] else ""))
    reporter.set_stage(
        f"done — {s['complete_stars']} complete stars kept, "
        f"{s['deleted_files']} cutouts {verb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
