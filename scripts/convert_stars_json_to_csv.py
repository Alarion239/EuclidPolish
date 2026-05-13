#!/usr/bin/env python
"""One-shot migrator: ``stars.json`` (legacy) → ``stars.csv`` (current).

The runtime no longer reads JSON catalogs — call this once to bring an
old catalog into the new format. The script tolerates every shape the
old code wrote:

  - ``"valid": True``                                       (very old)
  - ``"valid": {"512": True}``                              (band-less)
  - ``"valid": {"VIS": {"512": True}}``                     (current)

All three roll up into the canonical nested form before being handed
to :class:`StarCatalog.save`, which writes the new CSV.

Usage:
    python scripts/convert_stars_json_to_csv.py
    python scripts/convert_stars_json_to_csv.py --output-dir data/euclid_stars
    python scripts/convert_stars_json_to_csv.py --keep-old   # preserve stars.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.euclid.catalog import StarCatalog

_FLAG_KINDS = ("valid", "corrupted", "download_failed")
_DEFAULT_BAND = "VIS"


def _promote_flag(value):
    """Coerce any of the three historical encodings into the current
    ``{band: {str(size): True}}`` nested-dict form. Returns ``{}`` when
    the input encodes no information."""
    if value is True:
        # Very old: bare bool, treat as VIS / unknown size — drop it
        # because we can't recover the size.
        return {}
    if not isinstance(value, dict) or not value:
        return {}
    sample = next(iter(value.values()))
    if isinstance(sample, dict):
        # Already current — copy True entries through.
        out: dict[str, dict[str, bool]] = {}
        for band, sizes in value.items():
            if not isinstance(sizes, dict):
                continue
            for size, ok in sizes.items():
                if ok:
                    out.setdefault(band, {})[str(size)] = True
        return out
    # Band-less {size: bool} — promote to VIS.
    return {_DEFAULT_BAND: {str(size): True
                            for size, ok in value.items() if ok}}


def _promote_star(star: dict) -> dict:
    out = {
        "id":        int(star["id"]),
        "ra":        float(star["ra"]),
        "dec":       float(star["dec"]),
        "magnitude": float(star["magnitude"]),
    }
    for kind in _FLAG_KINDS:
        nested = _promote_flag(star.get(kind))
        if nested:
            out[kind] = nested
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=Config.DEFAULT_OUTPUT_DIR,
                    help="Catalog root containing stars.json (and where "
                         "stars.csv will be written).")
    ap.add_argument("--keep-old", action="store_true",
                    help="Leave stars.json in place after writing stars.csv "
                         "(default: rename to stars.json.bak).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    json_path = os.path.join(args.output_dir, "stars.json")
    if not os.path.isfile(json_path):
        print(f"✗ no stars.json at {json_path}")
        return 1
    with open(json_path) as fh:
        legacy = json.load(fh)

    raw_stars = legacy.get("stars", [])
    promoted = [_promote_star(s) for s in raw_stars]
    print(f"loaded {len(raw_stars)} stars from {json_path}")

    cat = StarCatalog(args.output_dir)
    cat.save({"stars": promoted})
    print(f"✓ wrote {len(promoted)} stars to {cat.catalog_path}")

    if args.keep_old:
        print(f"  (kept legacy file at {json_path})")
    else:
        bak = json_path + ".bak"
        os.replace(json_path, bak)
        print(f"  (renamed legacy file → {bak})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
