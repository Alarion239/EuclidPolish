"""Join Zoobot prediction CSVs back to the stamp catalog (pure).

The trainer writes one prediction CSV per reconstruction with an ``id_str`` and
the softmax columns (``p_notlens``, ``p_lens``). Here we extract ``P(lens)`` and
merge it with the catalog so the evaluation has ``(score, label, theta_E)`` per
stamp. numpy/csv only — no torch.
"""

from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def read_scores(pred_path: str, *, pos_col: str = "p_lens",
                id_col: str = "id_str") -> Dict[str, float]:
    """Read a prediction CSV → ``{id_str: P(lens)}``.

    Uses ``pos_col`` if present; otherwise falls back to a column whose name
    ends in ``lens`` but is not the ``not``-lens class, else the last column.
    """
    out: Dict[str, float] = {}
    with open(pred_path, newline="") as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
        col: Optional[str] = pos_col if pos_col in cols else None
        if col is None:
            cand = [c for c in cols
                    if c.lower().endswith("lens") and "not" not in c.lower()]
            col = cand[0] if cand else (cols[-1] if cols else None)
        if col is None:
            return out
        for row in rdr:
            try:
                out[row[id_col]] = float(row[col])
            except (TypeError, ValueError):
                continue
    return out


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def join_scores(
    catalog_rows: Sequence[Dict[str, Any]],
    scores: Dict[str, float], *,
    split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Attach ``score = P(lens)`` to catalog rows; optionally restrict to a split.

    Rows without a matching prediction are dropped. Returns dicts with
    ``id_str, is_lens, theta_E_arcsec, recon, score``.
    """
    out: List[Dict[str, Any]] = []
    for r in catalog_rows:
        if split is not None and r.get("split") != split:
            continue
        sid = r.get("id_str")
        if sid not in scores:
            continue
        out.append({
            "id_str": sid,
            "is_lens": int(_f(r.get("is_lens", 0)) or 0),
            "theta_E_arcsec": _f(r.get("theta_E_arcsec")),
            "recon": r.get("recon"),
            "score": scores[sid],
        })
    return out


def scored_arrays(joined: Sequence[Dict[str, Any]]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``join_scores`` output → ``(scores, labels, theta_e)`` numpy arrays."""
    s = np.array([j["score"] for j in joined], dtype=float)
    y = np.array([j["is_lens"] for j in joined], dtype=int)
    te = np.array([j["theta_E_arcsec"] for j in joined], dtype=float)
    return s, y, te
