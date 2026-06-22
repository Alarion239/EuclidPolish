"""Generic evaluation-catalog reader.

An *evaluation catalog* is the input contract for catalog-based model
evaluation (``scripts/fasrc_eval_catalog.py``): a CSV with one row per sky
target the model should be run on. The required columns are::

    id, ra, dec

with an optional ``grade`` column (e.g. ``A``/``B``/``C`` for the Natalie
Lines Euclid Q1 strong-lens catalog). Any extra columns are preserved on the
returned row dicts under ``extra`` so downstream code can carry through
provenance (Einstein radius, notes, …) without this module needing to know
about them.

Column names are matched case-insensitively and a few common aliases are
accepted (``RAJ2000``/``ra_deg`` → ``ra``, etc.) so a catalog can usually be
fed in unmodified. The reader is deliberately format-agnostic: anything that
exposes ``id,ra,dec`` works, which keeps the evaluation backbone reusable for
non-lens catalogs.
"""

from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional

# Accepted source-column aliases → canonical name. Matched case-insensitively
# (the incoming header is lower-cased before lookup).
_ID_ALIASES = ("id_str", "id", "name", "object_id", "object", "iauname",
               "source_id")
_RA_ALIASES = ("ra", "ra_deg", "raj2000", "ra_j2000", "right_ascension", "alpha")
_DEC_ALIASES = ("dec", "dec_deg", "dej2000", "de_j2000", "declination", "delta")
_GRADE_ALIASES = ("grade", "class", "rank", "quality")


class CatalogError(ValueError):
    """Raised when a catalog CSV is missing required columns or has bad rows."""


def _resolve_column(header: List[str], aliases: tuple) -> Optional[str]:
    """Return the first header entry whose lower-case form is in ``aliases``."""
    lower = {h.lower().strip(): h for h in header}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    return None


def read_eval_catalog(
    path: str,
    *,
    grade: Optional[str] = None,
    max_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Read an evaluation catalog CSV into a list of row dicts.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    grade : str, optional
        If given, keep only rows whose ``grade`` column matches (case- and
        whitespace-insensitive). Raises if the catalog has no grade column.
    max_n : int, optional
        Cap the number of rows returned (after grade filtering), in file
        order. ``None`` (or ``<= 0``) means no cap.

    Returns
    -------
    list of dict
        Each dict has keys ``id`` (str), ``ra`` (float), ``dec`` (float),
        ``grade`` (str or None), and ``extra`` (dict of the remaining columns).

    Raises
    ------
    CatalogError
        If required columns are missing, a coordinate fails to parse, or a
        grade filter is requested on a catalog without a grade column.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        id_col = _resolve_column(header, _ID_ALIASES)
        ra_col = _resolve_column(header, _RA_ALIASES)
        dec_col = _resolve_column(header, _DEC_ALIASES)
        grade_col = _resolve_column(header, _GRADE_ALIASES)

        missing = [n for n, c in (("id", id_col), ("ra", ra_col),
                                  ("dec", dec_col)) if c is None]
        if missing:
            raise CatalogError(
                f"{path}: missing required column(s) {missing}; "
                f"found header {header}"
            )
        if grade is not None and grade_col is None:
            raise CatalogError(
                f"{path}: grade filter {grade!r} requested but no grade "
                f"column found in header {header}"
            )

        want_grade = grade.strip().lower() if grade is not None else None
        rows: List[Dict[str, Any]] = []
        for lineno, raw in enumerate(reader, start=2):
            row_grade = (raw.get(grade_col).strip()
                         if grade_col and raw.get(grade_col) is not None
                         else None)
            if want_grade is not None and (
                row_grade is None or row_grade.lower() != want_grade
            ):
                continue
            try:
                ra = float(raw[ra_col])
                dec = float(raw[dec_col])
            except (TypeError, ValueError) as e:
                raise CatalogError(
                    f"{path}:{lineno}: bad ra/dec "
                    f"({raw.get(ra_col)!r}, {raw.get(dec_col)!r}): {e}"
                ) from e
            obj_id = (raw.get(id_col) or "").strip() or f"row{lineno}"
            extra = {k: v for k, v in raw.items()
                     if k not in (id_col, ra_col, dec_col, grade_col)}
            rows.append({
                "id":    obj_id,
                "ra":    ra,
                "dec":   dec,
                "grade": row_grade,
                "extra": extra,
            })
            if max_n and max_n > 0 and len(rows) >= max_n:
                break
        return rows
