"""Build a real-field-galaxy evaluation catalog by querying the Euclid archive.

Mirrors :mod:`euclid_polish.eval.lens_catalog` in shape (one source of truth
shared by a CLI and the WebUI), but the "fetch" is a live ADQL cone query on
``catalogue.mer_catalogue`` around the strong-lens fields rather than a Zenodo
download. It selects clean, resolved, bigger-end galaxies — confidently *not*
gravitational lenses — and writes a normalized ``id,ra,dec,grade`` CSV with
``grade="gal"`` that the grouped eval runner consumes exactly like the A/B/C
lens catalog. The drawn set is cached (stable ids) so each galaxy's cutout is
downloaded at most once across runs.
"""
from __future__ import annotations

import csv
import math
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from astroquery.esa.euclid import Euclid

from euclid_polish.config import Config
from euclid_polish.catalog.client import EuclidCatalog, EuclidAuthError
from euclid_polish.eval.eval_catalog import read_eval_catalog
from euclid_polish.catalog.photometry import uJy_to_ab_mag
from euclid_polish.catalog.validator import angular_separation_arcsec

#: Group label for a real field galaxy (parallels the synthetic ``syn-gal``).
GAL_GRADE = "gal"

# MER catalogue (``catalogue.mer_catalogue``) column names. ``det_quality_flag``
# and the coord/flux columns are already used by EuclidCatalog queries; the
# morphology/classification columns below are the Euclid Q1 names — confirm them
# once against the live schema (see scripts/fetch_galaxy_catalog.py --probe), and
# if a name differs, change it here only (the ADQL is built from these).
_ID_COL = "object_id"
_RA_COL = "right_ascension"
_DEC_COL = "declination"
_SIZE_COL = "segmentation_area"      # px count in the segmentation map
_POINTLIKE_COL = "point_like_flag"   # 1 = point source (star), 0 = extended
_SPURIOUS_COL = "spurious_flag"      # 1 = artifact
_QUALITY_COL = "det_quality_flag"    # 0 = clean (no mask/blend/saturation/border)
_FLUX_COL = "flux_vis_psf"           # µJy — conservative brightness floor

# Galaxy selection window lives in Config.GalaxySelection (shared with
# EuclidCatalog.query_galaxies — single source of truth).
LENS_EXCLUDE_ARCSEC = 10.0           # drop anything this close to a known lens


def default_out_csv() -> str:
    """Default normalized galaxy-catalog path under the configured data dir."""
    return os.path.join(Config.EVAL_CATALOG_DIR, "galaxy_catalog", "galaxies.csv")


def _diam_to_area_px(diam_arcsec: float) -> float:
    """Segmentation area (px) of a circular source of on-sky diameter ``diam``."""
    r_px = (diam_arcsec / 2.0) / Config.VIS_PIXEL_SCALE_ARCSEC
    return math.pi * r_px * r_px


def galaxy_adql(ra: float, dec: float, radius_deg: float,
                limit: int = Config.GalaxySelection.MAX_RESULTS) -> str:
    """ADQL cone query for clean, resolved, bigger-end galaxies at ``(ra, dec)``."""
    area_lo = _diam_to_area_px(Config.GalaxySelection.DIAM_LO_ARCSEC)
    area_hi = _diam_to_area_px(Config.GalaxySelection.DIAM_HI_ARCSEC)
    return f"""
    SELECT TOP {limit}
        {_ID_COL}, {_RA_COL}, {_DEC_COL}, {_SIZE_COL}, {_FLUX_COL}
    FROM catalogue.mer_catalogue
    WHERE CONTAINS(
        POINT('ICRS', {_RA_COL}, {_DEC_COL}),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    ) = 1
      AND {_POINTLIKE_COL} = 0
      AND {_SPURIOUS_COL} = 0
      AND {_QUALITY_COL} = 0
      AND {_SIZE_COL} BETWEEN {area_lo:.1f} AND {area_hi:.1f}
      AND {_FLUX_COL} IS NOT NULL
      AND {_FLUX_COL} > 0
    """


def _unmask_float(value: Any) -> Optional[float]:
    """Float from a possibly-masked/None TAP cell, or None if not finite."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _candidates_from_results(results: Any,
                             mag_floor: float = Config.GalaxySelection.MAG_FLOOR
                             ) -> List[Dict[str, Any]]:
    """Parse a TAP result into candidate galaxy dicts passing the brightness floor.

    Works with an astropy ``Table`` or any iterable of column-indexable rows
    (e.g. dicts in tests). Size/quality cuts are applied server-side in the
    ADQL; here we only drop non-finite or too-faint sources.
    """
    out: List[Dict[str, Any]] = []
    if results is None:
        return out
    for row in results:
        ra = _unmask_float(row[_RA_COL])
        dec = _unmask_float(row[_DEC_COL])
        flux = _unmask_float(row[_FLUX_COL])
        if ra is None or dec is None or flux is None or flux <= 0:
            continue
        mag = uJy_to_ab_mag(flux)
        if mag >= mag_floor:                       # too faint → skip
            continue
        out.append({"id": f"gal_{row[_ID_COL]}", "ra": ra, "dec": dec,
                    "mag_vis": mag})
    return out


def _login() -> bool:
    """True if a Euclid archive session is available (EUCLID_USER/PASSWORD)."""
    try:
        EuclidCatalog()
        return True
    except EuclidAuthError:
        return False


def _run_query(query: str) -> Tuple[Any, str]:
    """Run a synchronous ADQL query; return ``(results_or_None, error)``."""
    try:
        job = Euclid.launch_job(query)
        return (job.get_results() if job is not None else None), ""
    except Exception as e:  # noqa: BLE001 — surfaced to the caller, never raised
        return None, f"{type(e).__name__}: {e}"


def _near_any_lens(ra: float, dec: float, lenses: List[Dict[str, Any]],
                   radius_arcsec: float) -> bool:
    """True if (ra, dec) is within ``radius_arcsec`` of any lens position."""
    for lens in lenses:
        if angular_separation_arcsec(ra, dec, lens["ra"], lens["dec"]) < radius_arcsec:
            return True
    return False


def _read_cached(out_csv: str) -> List[Dict[str, Any]]:
    """Read a previously-written galaxy catalog (stable order), or []."""
    if not os.path.isfile(out_csv):
        return []
    rows: List[Dict[str, Any]] = []
    with open(out_csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "ra": float(r["ra"]),
                         "dec": float(r["dec"]), "grade": r.get("grade", GAL_GRADE)})
    return rows


def _write(out_csv: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "ra", "dec", "grade"])
        for r in rows:
            w.writerow([r["id"], r["ra"], r["dec"], r.get("grade", GAL_GRADE)])


def build(out_csv: Optional[str] = None, *, n_galaxies: int,
          lens_catalog_path: str, seed: int = 0,
          cone_radius_arcmin: float = 3.0, oversample: int = 4,
          regenerate: bool = False,
          log: Optional[Callable[[str], None]] = None) -> Tuple[str, int]:
    """Build (or top up) the galaxy catalog to ``n_galaxies`` rows; return ``(path, n)``.

    Drawn from the same fields as the lenses (cone queries around each lens
    RA/Dec). The CSV is cached: already-drawn galaxies are kept (stable ids →
    their cutouts are reused, never re-downloaded); only the shortfall is queried.
    """
    emit = log or (lambda m: None)
    out = out_csv or default_out_csv()

    cached = [] if regenerate else _read_cached(out)
    if len(cached) >= n_galaxies:
        return out, len(cached)                     # cache already satisfies the request

    if not _login():
        raise RuntimeError(
            "Euclid archive login required to build the galaxy catalog. Set "
            "EUCLID_USER/EUCLID_PASSWORD or a credentials file (same credentials "
            "the lens-cutout downloads use).")

    lenses = read_eval_catalog(lens_catalog_path)
    rng = random.Random(seed)
    fields = list(lenses)
    rng.shuffle(fields)

    pool: List[Dict[str, Any]] = []
    seen_ids = {r["id"] for r in cached}
    target_pool = oversample * n_galaxies
    radius_deg = cone_radius_arcmin / 60.0

    for fld in fields:
        if len(cached) + len(pool) >= target_pool:
            break
        results, err = _run_query(galaxy_adql(fld["ra"], fld["dec"], radius_deg))
        if err:
            emit(f"  galaxy query failed at ({fld['ra']:.4f}, {fld['dec']:.4f}): {err}")
            continue
        for cand in _candidates_from_results(results):
            if cand["id"] in seen_ids:
                continue
            if _near_any_lens(cand["ra"], cand["dec"], lenses, LENS_EXCLUDE_ARCSEC):
                continue
            seen_ids.add(cand["id"])
            pool.append(cand)

    rng.shuffle(pool)
    need = n_galaxies - len(cached)
    drawn = pool[:need]
    rows = cached + [{"id": d["id"], "ra": d["ra"], "dec": d["dec"],
                      "grade": GAL_GRADE} for d in drawn]
    if len(rows) < n_galaxies:
        emit(f"⚠ only {len(rows)} galaxies found (< {n_galaxies} requested); using all")
    _write(out, rows)
    return out, len(rows)
