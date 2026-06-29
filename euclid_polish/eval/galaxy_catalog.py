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
from collections.abc import Callable
from typing import Any

import numpy as np
from astroquery.esa.euclid import Euclid

from euclid_polish.catalog.client import EuclidAuthError, EuclidCatalog
from euclid_polish.catalog.photometry import uJy_to_ab_mag
from euclid_polish.catalog.validator import angular_separation_arcsec
from euclid_polish.config import Config
from euclid_polish.eval.eval_catalog import read_eval_catalog

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
                limit: int = Config.GalaxySelection.MAX_RESULTS,
                *, relax: bool = False) -> str:
    """ADQL cone query for clean, resolved, bigger-end galaxies at ``(ra, dec)``.

    Strict (``relax=False``) requires ``point_like_flag = 0`` and
    ``spurious_flag = 0`` — but that bare equality returned **zero** rows on cones
    with hundreds of sources, because in the live MER catalogue these classifier
    flags are NULL (unset) for ordinary sources and only set (``= 1``) for the
    point-like / spurious ones, so ``= 0`` matched nothing.

    ``relax=True`` (the default profile) is therefore NULL-safe: keep sources NOT
    flagged point-like or spurious (``flag IS NULL OR flag = 0`` ⇒ *extended,
    non-spurious*) while still dropping anything explicitly flagged ``= 1``. This
    is what keeps the set galaxies rather than stars. Both profiles keep the
    ``det_quality_flag = 0`` clean cut and the full resolved-size window — the
    extendedness flag and the segmentation-area floor each independently exclude
    point sources. The diagnostic breakdown counts the strict cuts plus the flag
    value distribution so the encoding stays visible.
    """
    area_lo = _diam_to_area_px(Config.GalaxySelection.DIAM_LO_ARCSEC)
    area_hi = _diam_to_area_px(Config.GalaxySelection.DIAM_HI_ARCSEC)
    preds = [
        f"CONTAINS(POINT('ICRS', {_RA_COL}, {_DEC_COL}), "
        f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1",
    ]
    if relax:                                    # NULL-safe "not point-like / not spurious"
        preds += [f"({_POINTLIKE_COL} IS NULL OR {_POINTLIKE_COL} = 0)",
                  f"({_SPURIOUS_COL} IS NULL OR {_SPURIOUS_COL} = 0)"]
    else:                                        # strict bare equality (zeroed the cone)
        preds += [f"{_POINTLIKE_COL} = 0", f"{_SPURIOUS_COL} = 0"]
    preds += [
        f"{_QUALITY_COL} = 0",
        f"{_SIZE_COL} BETWEEN {area_lo:.1f} AND {area_hi:.1f}",
        f"{_FLUX_COL} IS NOT NULL",
        f"{_FLUX_COL} > 0",
    ]
    where = "\n      AND ".join(preds)
    return f"""
    SELECT TOP {limit}
        {_ID_COL}, {_RA_COL}, {_DEC_COL}, {_SIZE_COL}, {_FLUX_COL}
    FROM catalogue.mer_catalogue
    WHERE {where}
    """


def _unmask_float(value: Any) -> float | None:
    """Float from a possibly-masked/None TAP cell, or None if not finite.

    A masked astropy/numpy cell is treated as missing up front: ``float()`` on a
    masked element warns ("converting a masked element to nan") before yielding
    nan, so we short-circuit on the mask to keep the per-field log clean.
    """
    if value is None or value is np.ma.masked or np.ma.is_masked(value):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _candidates_from_results(results: Any,
                             mag_floor: float = Config.GalaxySelection.MAG_FLOOR
                             ) -> list[dict[str, Any]]:
    """Parse a TAP result into candidate galaxy dicts passing the brightness floor.

    Works with an astropy ``Table`` or any iterable of column-indexable rows
    (e.g. dicts in tests). Size/quality cuts are applied server-side in the
    ADQL; here we only drop non-finite or too-faint sources.
    """
    out: list[dict[str, Any]] = []
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


def _run_query(query: str) -> tuple[Any, str]:
    """Run a synchronous ADQL query; return ``(results_or_None, error)``."""
    try:
        job = Euclid.launch_job(query)
        return (job.get_results() if job is not None else None), ""
    except Exception as e:  # noqa: BLE001 — surfaced to the caller, never raised
        return None, f"{type(e).__name__}: {e}"


def _cone_count(ra: float, dec: float, radius_deg: float,
                where_extra: str = "") -> tuple[int | None, str]:
    """COUNT(*) of ``catalogue.mer_catalogue`` in a cone, with one extra cut.

    Returns ``(count_or_None, error)``. Used by :func:`_diagnose_zero_cone` to
    attribute an empty galaxy query to the specific WHERE cut responsible.
    """
    q = (f"SELECT COUNT(*) FROM catalogue.mer_catalogue "
         f"WHERE CONTAINS(POINT('ICRS', {_RA_COL}, {_DEC_COL}), "
         f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1"
         + (f" AND {where_extra}" if where_extra else ""))
    results, err = _run_query(q)
    if err:
        return None, err
    try:
        return int(results[0][0]), ""
    except (TypeError, IndexError, KeyError, ValueError):
        return None, "unparseable count"


def _diagnose_zero_cone(ra: float, dec: float, radius_deg: float,
                        emit: Callable[[str], None]) -> None:
    """Log a COUNT(*) for the bare cone and each galaxy cut in isolation.

    Run once when a cone unexpectedly yields no galaxies: a cut whose count is 0
    while the bare cone is non-zero is the one eliminating everything (a wrong
    column *value*/unit — a wrong column *name* would error instead). The bare
    cone reading 0 points at coverage/coordinates rather than the cuts.
    """
    area_lo = _diam_to_area_px(Config.GalaxySelection.DIAM_LO_ARCSEC)
    area_hi = _diam_to_area_px(Config.GalaxySelection.DIAM_HI_ARCSEC)
    # Bare cone + each cut alone, plus the flag VALUE distribution (=0 / =1 / IS
    # NULL) so the encoding is unambiguous: if `= 0` reads ≈0 while `= 1` and
    # `IS NULL` carry the rows, the flag marks point sources (=1) and leaves
    # ordinary sources NULL — so extended = (IS NULL OR = 0).
    cuts = [
        ("bare cone (no cuts)", ""),
        (f"{_QUALITY_COL} = 0", f"{_QUALITY_COL} = 0"),
        (f"{_POINTLIKE_COL} = 0", f"{_POINTLIKE_COL} = 0"),
        (f"{_POINTLIKE_COL} = 1", f"{_POINTLIKE_COL} = 1"),
        (f"{_POINTLIKE_COL} IS NULL", f"{_POINTLIKE_COL} IS NULL"),
        (f"{_SPURIOUS_COL} = 0", f"{_SPURIOUS_COL} = 0"),
        (f"{_SPURIOUS_COL} = 1", f"{_SPURIOUS_COL} = 1"),
        (f"{_SPURIOUS_COL} IS NULL", f"{_SPURIOUS_COL} IS NULL"),
        ("extended (IS NULL OR = 0)",
         f"({_POINTLIKE_COL} IS NULL OR {_POINTLIKE_COL} = 0)"),
        (f"{_SIZE_COL} in [{area_lo:.0f},{area_hi:.0f}]",
         f"{_SIZE_COL} BETWEEN {area_lo:.1f} AND {area_hi:.1f}"),
    ]
    emit(f"⟐ cut COUNT(*) breakdown at ({ra:.4f}, {dec:.4f}) — bare cone + each "
         "cut alone + the flag value distribution; a `= 0` reading ≈0 while "
         "`= 1`/`IS NULL` carry the rows means the flag marks point sources:")
    for name, pred in cuts:
        n, err = _cone_count(ra, dec, radius_deg, pred)
        emit(f"    {name:<42} → {('ERROR: ' + err) if err else n}")


def _near_any_lens(ra: float, dec: float, lenses: list[dict[str, Any]],
                   radius_arcsec: float) -> bool:
    """True if (ra, dec) is within ``radius_arcsec`` of any lens position."""
    return any(angular_separation_arcsec(ra, dec, lens["ra"], lens["dec"]) < radius_arcsec for lens in lenses)


def _read_cached(out_csv: str) -> list[dict[str, Any]]:
    """Read a previously-written galaxy catalog (stable order), or []."""
    if not os.path.isfile(out_csv):
        return []
    rows: list[dict[str, Any]] = []
    with open(out_csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "ra": float(r["ra"]),
                         "dec": float(r["dec"]), "grade": r.get("grade", GAL_GRADE)})
    return rows


def _write(out_csv: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "ra", "dec", "grade"])
        for r in rows:
            w.writerow([r["id"], r["ra"], r["dec"], r.get("grade", GAL_GRADE)])


def build(out_csv: str | None = None, *, n_galaxies: int,
          lens_catalog_path: str, seed: int = 0,
          cone_radius_arcmin: float = 3.0, oversample: int = 4,
          regenerate: bool = False, relax: bool = True,
          client: EuclidCatalog | None = None,
          log: Callable[[str], None] | None = None) -> tuple[str, int]:
    """Build (or top up) the galaxy catalog to ``n_galaxies`` rows; return ``(path, n)``.

    Drawn from the same fields as the lenses (cone queries around each lens
    RA/Dec). The CSV is cached: already-drawn galaxies are kept (stable ids →
    their cutouts are reused, never re-downloaded); only the shortfall is queried.

    Authentication: the cone queries run on astroquery's process-global
    ``Euclid`` session. Pass ``client`` to supply an already-authenticated
    :class:`EuclidCatalog` (e.g. the WebUI's ``euclid_session`` login, which has
    no env vars); when given, the env-var ``_login()`` guard is skipped. With no
    ``client`` (the CLI path), credentials fall back to ``EUCLID_USER`` /
    ``EUCLID_PASSWORD``.
    """
    emit = log or (lambda m: None)
    out = out_csv or default_out_csv()

    cached = [] if regenerate else _read_cached(out)
    if len(cached) >= n_galaxies:
        return out, len(cached)                     # cache already satisfies the request

    if client is None and not _login():
        raise RuntimeError(
            "Euclid archive login required to build the galaxy catalog. Log in "
            "on the Evaluation page, or set EUCLID_USER/EUCLID_PASSWORD (same "
            "credentials the lens-cutout downloads use).")

    lenses = read_eval_catalog(lens_catalog_path)
    rng = random.Random(seed)
    fields = list(lenses)
    rng.shuffle(fields)

    pool: list[dict[str, Any]] = []
    seen_ids = {r["id"] for r in cached}
    target_pool = oversample * n_galaxies
    radius_deg = cone_radius_arcmin / 60.0

    # Up-front context so a sparse/empty result is diagnosable from the log
    # alone: how many fields, how big a cone, and the exact ADQL (a wrong MER
    # column name surfaces here as the SQL beside the server's error).
    emit(f"querying {len(fields)} lens field(s) from {lens_catalog_path}; "
         f"cone r={cone_radius_arcmin:g}', target pool {target_pool} "
         f"(oversample {oversample}× of {n_galaxies}); {len(cached)} cached")
    if relax:
        emit("galaxy profile: NULL-safe extendedness — keep sources NOT flagged "
             "point-like/spurious (flag IS NULL OR = 0 ⇒ extended), clean "
             "(det_quality=0), within the resolved size window + mag floor. "
             "(Strict flag = 0 returned 0: the flags are NULL for unflagged "
             "sources, =1 for point-like — so = 0 matched nothing.)")
    if fields:
        f0 = fields[0]
        emit("ADQL (per field):"
             + galaxy_adql(f0["ra"], f0["dec"], radius_deg, relax=relax))

    queried = 0
    diagnosed = False                                # run the 0-cone cascade once
    for fld in fields:
        if len(cached) + len(pool) >= target_pool:
            emit(f"pool target reached ({len(cached) + len(pool)}) — "
                 f"stopping after {queried} field(s)")
            break
        queried += 1
        results, err = _run_query(
            galaxy_adql(fld["ra"], fld["dec"], radius_deg, relax=relax))
        if err:
            emit(f"  field ({fld['ra']:.4f}, {fld['dec']:.4f}): query failed: {err}")
            continue
        raw = len(results) if results is not None else 0
        cands = _candidates_from_results(results)
        kept = 0
        lens_excl = 0                                # dropped: too close to a lens
        for cand in cands:
            if cand["id"] in seen_ids:
                continue
            if _near_any_lens(cand["ra"], cand["dec"], lenses, LENS_EXCLUDE_ARCSEC):
                lens_excl += 1                       # never include the lenses themselves
                continue
            seen_ids.add(cand["id"])
            pool.append(cand)
            kept += 1
        emit(f"  field ({fld['ra']:.4f}, {fld['dec']:.4f}): "
             f"{raw} raw → {len(cands)} passed mag floor → "
             f"+{kept} new ({lens_excl} dropped as lens-coincident) "
             f"(pool {len(pool)})")
        # Show the strict per-cut breakdown once: on the first empty cone, or —
        # under the relaxed investigation profile — on the first queried field
        # even when it yields data, so the offending strict cut is still named.
        if not diagnosed and (raw == 0 or (relax and queried == 1)):
            _diagnose_zero_cone(fld["ra"], fld["dec"], radius_deg, emit)
            diagnosed = True

    rng.shuffle(pool)
    need = n_galaxies - len(cached)
    drawn = pool[:need]
    rows = cached + [{"id": d["id"], "ra": d["ra"], "dec": d["dec"],
                      "grade": GAL_GRADE} for d in drawn]
    if len(rows) < n_galaxies:
        emit(f"⚠ only {len(rows)} galaxies found (< {n_galaxies} requested); using all")
    _write(out, rows)
    emit(f"✓ wrote {len(rows)} galaxies ({len(drawn)} new this run) → {out}")
    return out, len(rows)
