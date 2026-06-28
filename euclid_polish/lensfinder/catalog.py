"""Stamp catalog: schema, IO, and leakage-safe splitting.

Pure (csv + stdlib only) so it imports cleanly in the main env without torch or
TensorFlow. The catalog CSV is the cross-env contract: the TF-side stamp builder
writes it; the torch-side trainer reads it.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Sequence

#: One row per (source, reconstruction). ``recon`` ∈ {lr, sr, hr}; the three
#: rows for one source share ``id_str`` stem, ``field_index`` and ``is_lens``.
CATALOG_COLS = [
    "id_str", "file_loc", "is_lens", "theta_E_arcsec", "recon", "split",
    "field_index", "src_x_pix", "src_y_pix",
]

RECONS = ("lr", "sr", "hr")


def write_catalog(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    """Write catalog rows to CSV; columns = ``CATALOG_COLS`` then any extras."""
    keys: List[str] = list(CATALOG_COLS)
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_catalog(path: str) -> List[Dict[str, str]]:
    """Read a catalog CSV into a list of string-valued dicts (round-trip)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def subset_for_recon(rows: Sequence[Dict[str, Any]], recon: str
                     ) -> List[Dict[str, Any]]:
    """Rows for a single reconstruction (``lr`` / ``sr`` / ``hr``)."""
    return [r for r in rows if r.get("recon") == recon]


def class_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """``{"lens": n_positive, "nonlens": n_negative}`` over ``rows``."""
    pos = sum(1 for r in rows if _as_int(r.get("is_lens")) == 1)
    return {"lens": pos, "nonlens": len(rows) - pos}


def assign_splits(
    rows: Sequence[Dict[str, Any]], *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
    group_key: str = "field_index",
) -> Sequence[Dict[str, Any]]:
    """Assign ``split`` ∈ {train, val, test} to each row, grouped by ``group_key``.

    All rows sharing a group value land in the same split, so a lens and the
    galaxy negatives cut from the same field never straddle splits (prevents
    information leakage). Deterministic for a given ``seed``. Mutates ``rows``
    in place and returns them.
    """
    if val_frac + test_frac >= 1.0:
        raise ValueError(
            f"val_frac + test_frac must be < 1.0 (got {val_frac + test_frac})"
        )
    import random

    groups = sorted({str(r.get(group_key, "")) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = set(groups[:n_test])
    val = set(groups[n_test:n_test + n_val])
    for r in rows:
        g = str(r.get(group_key, ""))
        r["split"] = "test" if g in test else "val" if g in val else "train"
    return rows
