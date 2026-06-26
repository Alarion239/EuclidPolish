#!/usr/bin/env python
"""Download the Euclid Q1 strong-lens catalog and normalize it for evaluation.

Thin CLI over :mod:`euclid_polish.eval.lens_catalog` (the shared fetch logic
also used by the WebUI). Pulls the lightweight discovery catalog CSV from
Zenodo record 15025832 and writes a normalized ``id,ra,dec,grade,subset`` CSV
that ``scripts/fasrc_eval_catalog.py`` consumes.

Usage::

    python scripts/fetch_lens_catalog.py                 # all grades
    python scripts/fetch_lens_catalog.py --grade A       # Grade-A only
    python scripts/fetch_lens_catalog.py --out some.csv
    python scripts/fetch_lens_catalog.py --source raw.csv  # skip the download
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.eval import lens_catalog


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None,
                   help="normalized output CSV (default: "
                        "Config.EVAL_CATALOG_DIR/lens_catalog/lenses.csv)")
    p.add_argument("--grade", default=None,
                   help="keep only this grade (A/B/C); default: all grades")
    p.add_argument("--source", default=None,
                   help="use an already-downloaded source CSV instead of "
                        "fetching from Zenodo")
    args = p.parse_args(argv)

    if not args.source:
        print(f"downloading {lens_catalog.SOURCE_URL}")
    out_csv, n = lens_catalog.fetch(args.out, grade=args.grade,
                                    source=args.source)
    print(f"  ✓ wrote {n} rows → {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
