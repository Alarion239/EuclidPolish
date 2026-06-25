"""Build a real-field-galaxy evaluation catalog by querying the Euclid archive.

Mirrors :mod:`euclid_polish.euclid.lens_catalog` in shape (one source of truth
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
from euclid_polish.euclid.auth import login
from euclid_polish.euclid.eval_catalog import read_eval_catalog
from euclid_polish.euclid.photometry import uJy_to_ab_mag
from euclid_polish.euclid.validator import angular_separation_arcsec

#: Group label for a real field galaxy (parallels the synthetic ``syn-gal``).
GAL_GRADE = "gal"

# MER catalogue (``catalogue.mer_catalogue``) column names. ``det_quality_flag``
# and the coord/flux columns are already used by StarCatalog (catalog.py); the
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

# Selection defaults (tunable).
DIAM_LO_ARCSEC = 2.0                 # bigger-end lower bound
DIAM_HI_ARCSEC = 5.0                 # hard cap (must fit the 53 px LR stamp)
MAG_FLOOR = 23.0                     # keep galaxies brighter than this (VIS PSF mag)
LENS_EXCLUDE_ARCSEC = 10.0           # drop anything this close to a known lens


def default_out_csv() -> str:
    """Default normalized galaxy-catalog path under the configured data dir."""
    return os.path.join(Config.EVAL_CATALOG_DIR, "galaxy_catalog", "galaxies.csv")


def _diam_to_area_px(diam_arcsec: float) -> float:
    """Segmentation area (px) of a circular source of on-sky diameter ``diam``."""
    r_px = (diam_arcsec / 2.0) / Config.VIS_PIXEL_SCALE_ARCSEC
    return math.pi * r_px * r_px


def galaxy_adql(ra: float, dec: float, radius_deg: float) -> str:
    """ADQL cone query for clean, resolved, bigger-end galaxies at ``(ra, dec)``."""
    area_lo = _diam_to_area_px(DIAM_LO_ARCSEC)
    area_hi = _diam_to_area_px(DIAM_HI_ARCSEC)
    return f"""
    SELECT TOP 100000
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
                             mag_floor: float = MAG_FLOOR
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


def _run_query(query: str) -> Tuple[Any, str]:
    """Run a synchronous ADQL query; return ``(results_or_None, error)``."""
    try:
        job = Euclid.launch_job(query)
        return (job.get_results() if job is not None else None), ""
    except Exception as e:  # noqa: BLE001 — surfaced to the caller, never raised
        return None, f"{type(e).__name__}: {e}"
