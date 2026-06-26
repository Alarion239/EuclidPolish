#!/usr/bin/env python
"""Rebuild ``stars.csv`` from the cutouts already on disk (recovery tool).

When a download job is OOM-killed (or otherwise crashes) mid-write it can leave
``stars.csv`` truncated. ``StarCatalog.load`` then reads the corrupt file as an
*empty* catalog, and the next download starts from scratch and overwrites it
with a small fresh batch — **orphaning** the cutout FITS already on disk (the
files survive, the *index* doesn't). That's the "12k downloaded but stars.csv
only shows 5k" symptom.

This re-indexes every ``star_<id>_<size>.fits`` under
``<output-dir>/cutouts/<band>/``, recovering each star's id (filename) + sky
position (FITS WCS ``CRVAL1/2``) + per-(band,size) validity, **merging** with
any surviving rows (existing magnitude / PSF flux are kept; recovered stars get
NaN magnitude — re-queryable later if needed). The write is atomic and keeps a
``stars.csv.bak``.

Run it on FASRC pointing at the star output dir, e.g.::

    python scripts/rebuild_stars_catalog.py --output-dir $DATA_DIR/euclid_stars
    python scripts/rebuild_stars_catalog.py --output-dir ... --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

# Run-from-anywhere: put the repo root on sys.path so ``euclid_polish`` imports
# without an editable install (matches the other scripts/*.py).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.catalog.star_catalog import StarCatalog
from euclid_polish.catalog.cutout_integrity import rebuild_catalog_from_cutouts


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=Config.DEFAULT_OUTPUT_DIR,
                    help="Star catalog root (holds stars.csv + cutouts/).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be recovered; write nothing.")
    args = ap.parse_args()

    cat = StarCatalog(args.output_dir)
    print(f"Rebuilding catalog: {cat.catalog_path}")
    s = rebuild_catalog_from_cutouts(cat, dry_run=args.dry_run)
    print(f"  star ids on disk    : {s['ids_on_disk']}")
    print(f"  already in catalog  : {s['already_in_catalog']}")
    print(f"  recovered (added)   : {s['recovered']}  "
          f"(could not recover ra/dec for {s['missing_radec']})")
    print(f"  catalog: {s['catalog_before']} -> {s['catalog_after']} stars"
          + ("   [DRY RUN — nothing written]" if s["dry_run"] else ""))
    if not s["dry_run"]:
        print(f"  cutouts validated   : {s.get('validated_cutouts')}  "
              f"(valid in all {len(Config.BANDS)} bands: {s.get('valid_all_bands')})")
        print(f"  wrote {cat.catalog_path} (backup at {cat.catalog_path}.bak)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
