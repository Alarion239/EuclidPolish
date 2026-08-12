"""Deterministic Q1 stellar colour sample for the Gaia-Euclid locus.

Population density comes from the separate Q1 magnitude-bracket counts.  This
sample is stratified by magnitude and exists only to calibrate colours; it
never defines a sky density and never selects random field centres.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from astroquery.esa.euclid import Euclid

from euclid_polish.config import Config
from euclid_polish.photometry import ab_mag_to_uJy, uJy_to_ab_mag

Q1_STELLAR_COLOR_SAMPLE_VERSION = 1
Q1_STELLAR_COLOR_MAG_BIN_WIDTH = 0.5
Q1_STELLAR_COLOR_ROWS_PER_BIN = 500
Q1_STELLAR_COLOR_FIELD_RADIUS_DEG = 0.35
Q1_STELLAR_COLOR_FIELDS = (
    (269.733, 66.018, "EDF-N"),
    (61.241, -48.423, "EDF-S"),
    (52.932, -28.088, "EDF-F"),
)
GAIA_TAP_PROVIDER = "ARI Gaia TAP"
GAIA_TAP_URL = "https://gaia.ari.uni-heidelberg.de/tap"
GAIA_SYNC_MAXREC = 10_000


def q1_stellar_color_catalog_path() -> Path:
    return (
        Path(Config.DATA_DIR)
        / "population_comparison"
        / "q1_stellar_color_sample.csv"
    )


def q1_stellar_color_meta_path() -> Path:
    return q1_stellar_color_catalog_path().with_suffix(".meta.json")


def q1_gaia_color_catalog_path() -> Path:
    return (
        Path(Config.DATA_DIR)
        / "population_comparison"
        / "gaia_population.csv"
    )


def q1_gaia_color_meta_path() -> Path:
    return q1_gaia_color_catalog_path().with_suffix(".meta.json")


def _finite(value: Any) -> float | None:
    if value is None or np.ma.is_masked(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _field(value: Any, name: str) -> Any:
    for candidate in (name, name.lower(), name.upper()):
        try:
            return value[candidate]
        except (KeyError, IndexError, TypeError):
            continue
    return None


def _identifier(value: Any) -> str:
    if value is None or np.ma.is_masked(value):
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _fixed_q1_predicate() -> str:
    clauses = [
        (
            "CONTAINS(POINT('ICRS', mer.right_ascension, mer.declination), "
            f"CIRCLE('ICRS', {ra}, {dec}, "
            f"{Q1_STELLAR_COLOR_FIELD_RADIUS_DEG})) = 1"
        )
        for ra, dec, _name in Q1_STELLAR_COLOR_FIELDS
    ]
    return "(\n        " + "\n        OR ".join(clauses) + "\n      )"


def _euclid_color_query(lower: float, upper: float) -> str:
    faint_flux = float(ab_mag_to_uJy(upper))
    bright_flux = float(ab_mag_to_uJy(lower))
    return f"""
    SELECT TOP {Q1_STELLAR_COLOR_ROWS_PER_BIN}
        mer.object_id, mer.gaia_id, mer.point_like_prob,
        mer.flux_vis_3fwhm_aper, mer.fluxerr_vis_3fwhm_aper,
        mer.flux_y_3fwhm_aper, mer.fluxerr_y_3fwhm_aper,
        mer.flux_j_3fwhm_aper, mer.fluxerr_j_3fwhm_aper,
        mer.flux_h_3fwhm_aper, mer.fluxerr_h_3fwhm_aper,
        cls.phz_star_prob
    FROM catalogue.mer_catalogue AS mer
    LEFT OUTER JOIN catalogue.phz_classification AS cls
      ON mer.object_id = cls.object_id
    WHERE {_fixed_q1_predicate()}
      AND mer.flux_vis_psf > {faint_flux:.16g}
      AND mer.flux_vis_psf <= {bright_flux:.16g}
      AND mer.point_like_prob >= 0.9
      AND mer.point_like_prob <= 1.0
      AND mer.gaia_id IS NOT NULL
    ORDER BY mer.object_id
    """


def _launch_euclid(
    query: str, relogin: Callable[[], bool] | None,
) -> Any:
    job = Euclid.launch_job_async(query)
    if job is None and relogin is not None:
        try:
            refreshed = relogin()
        except Exception:  # noqa: BLE001 - surface archive failure below
            refreshed = False
        if refreshed:
            job = Euclid.launch_job_async(query)
    if job is None:
        raise RuntimeError("The Euclid archive rejected the Q1 stellar-colour query")
    results = job.get_results()
    if results is None:
        raise RuntimeError("The Q1 stellar-colour query returned no table")
    return results


def _euclid_record(raw: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "object_id": _identifier(_field(raw, "object_id")),
        "gaia_id": _identifier(_field(raw, "gaia_id")),
        "type": "star",
        "point_like_prob": _finite(_field(raw, "point_like_prob")),
        "phz_star_prob": _finite(_field(raw, "phz_star_prob")),
    }
    for band in ("vis", "y", "j", "h"):
        flux = _finite(_field(raw, f"flux_{band}_3fwhm_aper"))
        error = _finite(_field(raw, f"fluxerr_{band}_3fwhm_aper"))
        record[f"flux_{band}_aper_uJy"] = flux
        record[f"fluxerr_{band}_aper_uJy"] = error
        magnitude_key = {
            "vis": "mag_vis", "y": "mag_y_e",
            "j": "mag_j_e", "h": "mag_h_e",
        }[band]
        record[magnitude_key] = (
            float(uJy_to_ab_mag(flux))
            if flux is not None and flux > 0.0 else None
        )
    return record


def _gaia_query(ra: float, dec: float) -> str:
    return f"""
    SELECT source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag,
           phot_rp_mean_mag, phot_g_mean_flux, phot_g_mean_flux_error,
           phot_bp_mean_flux, phot_bp_mean_flux_error,
           phot_rp_mean_flux, phot_rp_mean_flux_error,
           bp_rp, teff_gspphot, ag_gspphot
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
      POINT('ICRS', ra, dec),
      CIRCLE('ICRS', {ra}, {dec}, {Q1_STELLAR_COLOR_FIELD_RADIUS_DEG})
    ) = 1
      AND phot_g_mean_mag IS NOT NULL
    """


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return temporary


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return temporary


def query_q1_stellar_color_sample(
    *,
    relogin: Callable[[], bool] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Query one deterministic magnitude-stratified Q1 colour sample."""
    bright = float(Config.STAR_MAG_BRIGHT)
    faint = float(Config.STAR_MAG_FAINT)
    width = Q1_STELLAR_COLOR_MAG_BIN_WIDTH
    bin_count = int(round((faint - bright) / width))
    if not math.isclose(bright + bin_count * width, faint, abs_tol=1e-9):
        raise ValueError("Stellar colour range must divide into 0.5-mag brackets")
    total = bin_count + len(Q1_STELLAR_COLOR_FIELDS)
    euclid_rows: dict[str, dict[str, Any]] = {}
    for index in range(bin_count):
        lower, upper = bright + index * width, bright + (index + 1) * width
        if progress:
            progress(index, total, f"Q1 stellar colours {lower:g} <= VIS < {upper:g}")
        results = _launch_euclid(_euclid_color_query(lower, upper), relogin)
        for raw in results:
            record = _euclid_record(raw)
            if record["object_id"] and record["gaia_id"]:
                euclid_rows[record["object_id"]] = record

    from pyvo.dal import TAPService

    service = TAPService(GAIA_TAP_URL)
    gaia_rows: dict[str, dict[str, Any]] = {}
    field_metadata = []
    for field_index, (ra, dec, name) in enumerate(Q1_STELLAR_COLOR_FIELDS):
        if progress:
            progress(
                bin_count + field_index, total,
                f"Gaia DR3 colours · {name}",
            )
        result = service.run_sync(_gaia_query(ra, dec), maxrec=GAIA_SYNC_MAXREC)
        query_status = str(result.query_status or "").upper()
        if query_status != "OK":
            raise RuntimeError(
                f"{GAIA_TAP_PROVIDER} returned {query_status or 'no'} status "
                f"for {name}; refusing a possibly truncated colour cache"
            )
        field_rows = 0
        for raw in result:
            source_id = _identifier(_field(raw, "source_id"))
            if not source_id:
                continue
            field_rows += 1
            gaia_rows[source_id] = {
                "source_id": source_id,
                "field_index": field_index,
                "ra": _finite(_field(raw, "ra")),
                "dec": _finite(_field(raw, "dec")),
                "g_mag": _finite(_field(raw, "phot_g_mean_mag")),
                "bp_mag": _finite(_field(raw, "phot_bp_mean_mag")),
                "rp_mag": _finite(_field(raw, "phot_rp_mean_mag")),
                "g_flux": _finite(_field(raw, "phot_g_mean_flux")),
                "g_flux_error": _finite(_field(raw, "phot_g_mean_flux_error")),
                "bp_flux": _finite(_field(raw, "phot_bp_mean_flux")),
                "bp_flux_error": _finite(_field(raw, "phot_bp_mean_flux_error")),
                "rp_flux": _finite(_field(raw, "phot_rp_mean_flux")),
                "rp_flux_error": _finite(_field(raw, "phot_rp_mean_flux_error")),
                "bp_rp": _finite(_field(raw, "bp_rp")),
                "temperature_k": _finite(_field(raw, "teff_gspphot")),
                "extinction_g_mag": _finite(_field(raw, "ag_gspphot")),
                "central_selected_star": 0,
            }
        field_metadata.append({
            "name": name, "ra": ra, "dec": dec, "rows": field_rows,
        })

    radius = Q1_STELLAR_COLOR_FIELD_RADIUS_DEG
    area_deg2 = len(Q1_STELLAR_COLOR_FIELDS) * math.pi * radius**2
    common = {
        "version": Q1_STELLAR_COLOR_SAMPLE_VERSION,
        "sampling_kind": "fixed_q1_magnitude_stratified_color_fields",
        "fields": field_metadata,
        "field_count": len(field_metadata),
        "radius_deg": radius,
        "radius_arcmin": 60.0 * radius,
        "area_deg2": area_deg2,
        "area_arcmin2": area_deg2 * 3600.0,
        "random_centres": False,
        "density_role": "none; Q1 0.1-mag brackets normalize the population",
    }
    euclid_meta = {
        **common,
        "catalog_version": Q1_STELLAR_COLOR_SAMPLE_VERSION,
        "rows": len(euclid_rows),
        "magnitude_bracket_width": width,
        "rows_per_bracket_limit": Q1_STELLAR_COLOR_ROWS_PER_BIN,
        "selection": (
            "fixed Q1 calibration fields; POINT_LIKE_PROB >= 0.9; Gaia ID; "
            "stratified by VIS PSF magnitude"
        ),
    }
    gaia_meta = {
        **common,
        "gaia_table": "gaiadr3.gaia_source",
        "tap_provider": GAIA_TAP_PROVIDER,
        "tap_url": GAIA_TAP_URL,
        "query_mode": "sync",
        "sync_maxrec": GAIA_SYNC_MAXREC,
        "rows": len(gaia_rows),
    }
    euclid_columns = [
        "object_id", "gaia_id", "type", "point_like_prob", "phz_star_prob",
        "mag_vis", "mag_y_e", "mag_j_e", "mag_h_e",
        *[
            key
            for band in ("vis", "y", "j", "h")
            for key in (f"flux_{band}_aper_uJy", f"fluxerr_{band}_aper_uJy")
        ],
    ]
    gaia_columns = [
        "source_id", "field_index", "ra", "dec", "g_mag", "bp_mag", "rp_mag",
        "g_flux", "g_flux_error", "bp_flux", "bp_flux_error",
        "rp_flux", "rp_flux_error", "bp_rp", "temperature_k",
        "extinction_g_mag", "central_selected_star",
    ]
    replacements = [
        (_write_csv(
            q1_stellar_color_catalog_path(), euclid_columns,
            list(euclid_rows.values()),
        ), q1_stellar_color_catalog_path()),
        (_write_csv(
            q1_gaia_color_catalog_path(), gaia_columns,
            list(gaia_rows.values()),
        ), q1_gaia_color_catalog_path()),
        (_write_json(q1_stellar_color_meta_path(), euclid_meta),
         q1_stellar_color_meta_path()),
        (_write_json(q1_gaia_color_meta_path(), gaia_meta),
         q1_gaia_color_meta_path()),
    ]
    for temporary, target in replacements:
        os.replace(temporary, target)
    if progress:
        progress(total, total, "Q1 Gaia-Euclid colour sample cached")
    return {"euclid": euclid_meta, "gaia": gaia_meta}


def q1_stellar_color_query_count() -> int:
    bright = float(Config.STAR_MAG_BRIGHT)
    faint = float(Config.STAR_MAG_FAINT)
    brackets = int(round(
        (faint - bright) / Q1_STELLAR_COLOR_MAG_BIN_WIDTH,
    ))
    return brackets + len(Q1_STELLAR_COLOR_FIELDS)


def q1_stellar_color_sample_state() -> dict[str, Any]:
    """Return availability without accepting legacy random-centre metadata."""
    try:
        euclid = json.loads(q1_stellar_color_meta_path().read_text())
        gaia = json.loads(q1_gaia_color_meta_path().read_text())
    except (OSError, json.JSONDecodeError):
        return {"cached": False, "euclid": None, "gaia": None}
    valid = all(
        payload.get("version") == Q1_STELLAR_COLOR_SAMPLE_VERSION
        and payload.get("sampling_kind")
        == "fixed_q1_magnitude_stratified_color_fields"
        and payload.get("random_centres") is False
        for payload in (euclid, gaia)
    )
    valid = bool(
        valid
        and all(
            int(payload.get("field_count") or 0)
            == len(Q1_STELLAR_COLOR_FIELDS)
            and len(payload.get("fields") or [])
            == len(Q1_STELLAR_COLOR_FIELDS)
            for payload in (euclid, gaia)
        )
    )
    return {
        "cached": bool(
            valid
            and q1_stellar_color_catalog_path().is_file()
            and q1_gaia_color_catalog_path().is_file()
        ),
        "euclid": euclid if valid else None,
        "gaia": gaia if valid else None,
    }
