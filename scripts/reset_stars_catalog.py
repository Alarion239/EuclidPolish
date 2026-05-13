#!/usr/bin/env python
"""Trim the star catalog to N random entries and wipe their cutouts.

Why this exists: photutils' EPSFBuilder converges with ~200 stars and
adding more mostly wastes archive bandwidth + disk. Keeping 1994 stars
when we only ever use ~200 of them slowed every subsequent download
attempt and inflated the catalog file.

This script:

  1. Picks ``--keep`` random stars from the current ``stars.csv``.
  2. Resets each kept star's ``valid`` / ``corrupted`` / ``download_failed``
     flags to empty (no cutouts on disk yet).
  3. Deletes every file under ``data/euclid_stars/cutouts/`` so the disk
     matches the new catalog state.
  4. Writes the trimmed catalog back to ``stars.csv`` (preserving each
     star's original id, ra, dec, magnitude).

Run with ``--dry-run`` to see what would change.

Usage:
    python scripts/reset_stars_catalog.py --keep 200
    python scripts/reset_stars_catalog.py --keep 200 --dry-run
    python scripts/reset_stars_catalog.py --keep 100 --seed 42
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.euclid.catalog import StarCatalog


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=Config.DEFAULT_OUTPUT_DIR,
                    help="Star catalog root (contains stars.csv + cutouts/)")
    ap.add_argument("--keep", type=int, default=200,
                    help="Number of stars to keep after the random sample")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the random sample (reproducible)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing anything")
    return ap.parse_args()


def _reset_flags(star: dict) -> dict:
    """Return a new dict with every download-state flag wiped."""
    return {k: v for k, v in star.items()
            if k not in ("valid", "corrupted", "download_failed")}


def main() -> int:
    args = parse_args()
    cat = StarCatalog(args.output_dir)
    cutouts_dir = os.path.join(args.output_dir, Config.CUTOUTS_SUBDIR)

    if not cat.exists():
        print(f"✗ no catalog at {cat.catalog_path}")
        return 1

    catalog = cat.load()
    stars = catalog["stars"]
    print(f"loaded {len(stars)} stars from {cat.catalog_path}")

    keep = min(args.keep, len(stars))
    rng = np.random.default_rng(args.seed)
    keep_idx = sorted(rng.choice(len(stars), size=keep, replace=False).tolist())
    kept = [_reset_flags(stars[i]) for i in keep_idx]
    print(f"keeping {len(kept)} random stars (seed={args.seed}); "
          f"dropping {len(stars) - len(kept)}")

    # Count what's on disk for the report.
    n_fits = 0
    for dirpath, _, files in os.walk(cutouts_dir):
        n_fits += sum(1 for f in files if f.lower().endswith(".fits"))
    print(f"will delete {n_fits} cutout FITS files under {cutouts_dir}")

    if args.dry_run:
        print("(dry-run: nothing written)")
        return 0

    # Wipe cutouts/<band>/ entirely.
    if os.path.isdir(cutouts_dir):
        for entry in os.listdir(cutouts_dir):
            full = os.path.join(cutouts_dir, entry)
            if os.path.isdir(full):
                shutil.rmtree(full)
            else:
                os.remove(full)
        print(f"  ✓ wiped {cutouts_dir}/*")

    # Persist the trimmed catalog. ``next_id`` is recomputed from the IDs
    # we kept, which is fine — the ones we dropped are gone for good.
    cat.save({"stars": kept})
    print(f"  ✓ wrote {len(kept)} stars to {cat.catalog_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
