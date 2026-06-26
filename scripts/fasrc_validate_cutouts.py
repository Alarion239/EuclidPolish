#!/usr/bin/env python
"""Integrity pass over downloaded Euclid star cutouts.

Opens every ``star_<id>_<size>.fits`` under each band's cutout dir and sets
the catalog's per-``(band, size)`` valid/corrupted flag from whether it opens
cleanly. ``download_all_bands.py`` runs this automatically at the end; you can
also run it standalone to re-validate an existing catalog.

    python scripts/fasrc_validate_cutouts.py
    python scripts/fasrc_validate_cutouts.py --output-dir /path/euclid_stars
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.catalog.star_catalog import StarCatalog
from euclid_polish.catalog.cutout_integrity import validate_all_cutouts
from euclid_polish.observability.reporter import Reporter


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=Config.DEFAULT_OUTPUT_DIR,
                    help="Star-cutout root holding stars.csv + cutouts/<band>/.")
    args = ap.parse_args()

    reporter = Reporter.from_env()
    reporter.set_stage("validating cutouts (integrity)")

    cat = StarCatalog(args.output_dir)
    if not cat.exists():
        reporter.error(f"no catalog at {cat.catalog_path}")
        print(f"✗ no catalog at {cat.catalog_path}")
        return 1

    catalog = cat.load()
    summary = validate_all_cutouts(cat, catalog, reporter=reporter)
    print(f"✓ checked {summary['checked']} cutouts; "
          f"{summary['unopenable']} unopenable; "
          f"{summary['valid_all_bands']} stars valid in all "
          f"{summary['n_bands']} bands")
    reporter.set_stage(
        f"done — {summary['valid_all_bands']} stars valid in all "
        f"{summary['n_bands']} bands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
