#!/usr/bin/env python
"""Download the Euclid Q1 strong-lens catalog and normalize it for evaluation.

Source: "Euclid Quick Data Release (Q1): The Strong Lensing Discovery Engine"
(Lines et al.), Zenodo record 15025832. We pull the lightweight discovery
catalog CSV (``q1_discovery_engine_lens_catalog.csv``, ~0.4 MB — *not* the
multi-GB ``lens.zip`` image bundle, since the evaluation pipeline re-fetches
cutouts from the archive at each RA/Dec) and write a normalized
``id,ra,dec,grade`` CSV that ``scripts/fasrc_eval_catalog.py`` consumes.

The normalized file lands at ``Config.EVAL_CATALOG_DIR/lens_catalog/lenses.csv``
by default. Columns:

    id     — the catalog's ``id_str``
    ra     — degrees (from ``right_ascension``)
    dec    — degrees (from ``declination``)
    grade  — A / B / C expert grade
    subset — discovery_engine / gz_euclid / q1_serendiptous (provenance)

Usage::

    python scripts/fetch_lens_catalog.py                 # all grades
    python scripts/fetch_lens_catalog.py --grade A       # Grade-A only
    python scripts/fetch_lens_catalog.py --out some.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import urllib.request

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config

ZENODO_RECORD = "15025832"
SOURCE_CSV = "q1_discovery_engine_lens_catalog.csv"
SOURCE_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{SOURCE_CSV}/content"
)


def _default_out() -> str:
    return os.path.join(Config.EVAL_CATALOG_DIR, "lens_catalog", "lenses.csv")


def download_source(dest: str) -> None:
    """Download the raw discovery-engine catalog CSV to ``dest``."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"downloading {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as resp:
        data = resp.read()
    with open(dest, "wb") as f:
        f.write(data)
    print(f"  ✓ {len(data):,} bytes → {dest}")


def normalize(src_csv: str, out_csv: str, grade: str | None = None) -> int:
    """Write a normalized ``id,ra,dec,grade,subset`` CSV; return row count."""
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    want = grade.strip().upper() if grade else None
    n = 0
    with open(src_csv, newline="") as fin, open(out_csv, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["id", "ra", "dec", "grade", "subset"])
        for row in reader:
            row_grade = (row.get("grade") or "").strip()
            if want and row_grade.upper() != want:
                continue
            writer.writerow([
                (row.get("id_str") or "").strip(),
                row.get("right_ascension", ""),
                row.get("declination", ""),
                row_grade,
                (row.get("subset") or "").strip(),
            ])
            n += 1
    print(f"  ✓ wrote {n} rows → {out_csv}")
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
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

    out_csv = args.out or _default_out()
    if args.source:
        src = args.source
    else:
        src = os.path.join(os.path.dirname(out_csv) or ".", SOURCE_CSV)
        download_source(src)

    normalize(src, out_csv, grade=args.grade)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
