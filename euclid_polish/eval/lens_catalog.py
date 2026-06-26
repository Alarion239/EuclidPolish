"""Fetch + normalize the Euclid Q1 strong-lens catalog.

Source: "Euclid Quick Data Release (Q1): The Strong Lensing Discovery Engine"
(Lines et al.), Zenodo record 15025832. We pull only the lightweight discovery
catalog CSV (``q1_discovery_engine_lens_catalog.csv``, ~0.4 MB — *not* the
multi-GB ``lens.zip`` image bundle; the evaluation pipeline re-fetches cutouts
from the archive at each RA/Dec) and write a normalized ``id,ra,dec,grade,subset``
CSV that the evaluation backbone consumes.

This module is the single source of truth for the fetch logic, shared by the
CLI (``scripts/fetch_lens_catalog.py``) and the WebUI fetch endpoint.
"""

from __future__ import annotations

import csv
import os
import urllib.request

from euclid_polish.config import Config

ZENODO_RECORD = "15025832"
SOURCE_CSV = "q1_discovery_engine_lens_catalog.csv"
SOURCE_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{SOURCE_CSV}/content"
)


def default_out_csv() -> str:
    """Default normalized-catalog path under the configured data dir."""
    return os.path.join(Config.EVAL_CATALOG_DIR, "lens_catalog", "lenses.csv")


def download_source(dest: str, *, timeout: int = 120) -> int:
    """Download the raw discovery-engine catalog CSV to ``dest``; return bytes."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with urllib.request.urlopen(SOURCE_URL, timeout=timeout) as resp:
        data = resp.read()
    with open(dest, "wb") as f:
        f.write(data)
    return len(data)


def normalize(src_csv: str, out_csv: str, grade: str | None = None) -> int:
    """Write a normalized ``id,ra,dec,grade,subset`` CSV; return row count.

    ``grade`` (e.g. ``"A"``) keeps only matching rows; ``None`` keeps all.
    """
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
    return n


def fetch(out_csv: str | None = None, *, grade: str | None = None,
          source: str | None = None) -> tuple[str, int]:
    """Download (unless ``source`` is given) + normalize → ``(out_csv, n_rows)``.

    ``source`` lets a caller reuse an already-downloaded raw CSV (and skip the
    network); otherwise the raw CSV is fetched next to ``out_csv``.
    """
    out = out_csv or default_out_csv()
    if source:
        src = source
    else:
        src = os.path.join(os.path.dirname(out) or ".", SOURCE_CSV)
        download_source(src)
    n = normalize(src, out, grade=grade)
    return out, n
