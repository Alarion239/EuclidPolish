#!/usr/bin/env python
"""One-shot migration: flat ``cutouts/star_*.fits`` → ``cutouts/VIS/star_*.fits``.

Earlier versions of the downloader stored every star cutout (all VIS) in
``data/euclid_stars/cutouts/``. The current downloader writes to
``cutouts/<band>/`` so multiple bands can coexist for the same catalog.

This script moves any ``star_XXXX_SSS.fits`` files directly under
``cutouts/`` into ``cutouts/VIS/`` and updates ``stars.json``'s
``valid`` / ``corrupted`` / ``download_failed`` shape from the band-less
``{size: bool}`` mapping to the band-aware ``{band: {size: bool}}``
mapping. The catalog loader auto-handles either shape, so running this
is optional — it just makes the on-disk layout match the new convention.

Usage:
    python scripts/migrate_cutouts_to_per_band.py
    python scripts/migrate_cutouts_to_per_band.py --dry-run
    python scripts/migrate_cutouts_to_per_band.py --output-dir ./data/euclid_stars
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=Config.DEFAULT_OUTPUT_DIR,
                    help="Star catalog root (containing stars.json + cutouts/)")
    ap.add_argument("--band", default="VIS",
                    help="Band to assign to the existing flat cutouts (default VIS)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing anything")
    return ap.parse_args()


def migrate_files(cutout_root: str, target_band: str, dry_run: bool) -> int:
    pattern = os.path.join(cutout_root, "star_[0-9][0-9][0-9][0-9]_*.fits")
    flat_files = sorted(glob.glob(pattern))
    if not flat_files:
        print(f"  (no flat cutouts under {cutout_root})")
        return 0
    target_dir = Config.cutout_dir_for_band(target_band, root=cutout_root)
    if not dry_run:
        os.makedirs(target_dir, exist_ok=True)
    print(f"  moving {len(flat_files)} files → {target_dir}")
    moved = 0
    for src in flat_files:
        dst = os.path.join(target_dir, os.path.basename(src))
        if os.path.exists(dst):
            print(f"    ! skipping (dest exists): {dst}")
            continue
        if dry_run:
            moved += 1
        else:
            shutil.move(src, dst)
            moved += 1
    return moved


def migrate_catalog(catalog_path: str, target_band: str, dry_run: bool) -> bool:
    if not os.path.isfile(catalog_path):
        print(f"  (no stars.json at {catalog_path} — nothing to update)")
        return False
    with open(catalog_path, "r") as fh:
        cat = json.load(fh)

    promoted = 0
    for star in cat.get("stars", []):
        for kind in ("valid", "corrupted", "download_failed"):
            raw = star.get(kind)
            if not isinstance(raw, dict) or not raw:
                continue
            sample = next(iter(raw.values()))
            if isinstance(sample, dict):
                continue  # already nested band → size
            # Band-less {size: bool} → promote to {band: {size: bool}}.
            star[kind] = {target_band: dict(raw)}
            promoted += 1

    print(f"  promoted {promoted} flag-dicts to per-band shape")
    if dry_run or promoted == 0:
        return promoted > 0
    with open(catalog_path, "w") as fh:
        json.dump(cat, fh, indent=2)
    return True


def main() -> int:
    args = parse_args()
    cutout_root = os.path.join(args.output_dir, Config.CUTOUTS_SUBDIR)
    catalog_path = os.path.join(args.output_dir, Config.CATALOG_FILE)

    print(f"output-dir   = {args.output_dir}")
    print(f"cutout root  = {cutout_root}")
    print(f"target band  = {args.band}")
    print(f"dry-run      = {args.dry_run}\n")

    print("[1] Files:")
    n_moved = migrate_files(cutout_root, args.band, args.dry_run)
    print(f"    {'(would move)' if args.dry_run else 'moved'} {n_moved} files\n")

    print("[2] Catalog JSON:")
    migrate_catalog(catalog_path, args.band, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
