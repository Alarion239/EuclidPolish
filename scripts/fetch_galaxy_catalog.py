#!/usr/bin/env python
"""Build a real-field-galaxy evaluation catalog from the Euclid archive.

Thin CLI over :mod:`euclid_polish.catalog.galaxy_catalog` (the shared build logic
also used by the grouped eval runner). Queries ``catalogue.mer_catalogue`` around
the strong-lens fields for clean, resolved, bigger-end galaxies and writes a
normalized ``id,ra,dec,grade`` CSV (``grade="gal"``). Needs Euclid archive
credentials (``EUCLID_USER``/``EUCLID_PASSWORD`` or a credentials file).

Usage::

    python scripts/fetch_galaxy_catalog.py --n 60 --lens path/to/lenses.csv
    python scripts/fetch_galaxy_catalog.py --n 60 --lens lenses.csv --out gals.csv
    python scripts/fetch_galaxy_catalog.py --n 60 --lens lenses.csv --regenerate
"""
from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.catalog import galaxy_catalog


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, required=True,
                   help="number of galaxies to draw")
    p.add_argument("--lens", required=True,
                   help="normalized lens catalog CSV (id,ra,dec,grade) to sample fields from")
    p.add_argument("--out", default=None,
                   help="output CSV (default: Config.EVAL_CATALOG_DIR/galaxy_catalog/galaxies.csv)")
    p.add_argument("--seed", type=int, default=0, help="random seed for the draw")
    p.add_argument("--regenerate", action="store_true",
                   help="ignore any cached catalog and re-query")
    args = p.parse_args(argv)

    out_csv, n = galaxy_catalog.build(
        args.out, n_galaxies=args.n, lens_catalog_path=args.lens,
        seed=args.seed, regenerate=args.regenerate, log=print)
    print(f"  ✓ wrote {n} galaxies → {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
