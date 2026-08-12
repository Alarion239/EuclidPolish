"""Aggregate Q1 VIS-2FWHM x Sersic-radius statistics.

The archive returns only sufficient statistics and radius-bin counts.  No
object catalogue or random sky-position sample is materialized by this workflow.
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from astroquery.esa.euclid import Euclid

from euclid_polish.config import Config
from euclid_polish.photometry import ab_mag_to_uJy
from euclid_polish.web.helpers.q1_galaxy_counts import (
    Q1_GALAXY_MAG_BIN_WIDTH,
    Q1_GALAXY_MAG_BRIGHT,
    Q1_GALAXY_MAG_FAINT,
    Q1_GALAXY_PROGRESSIVE_STRIDE,
)
from euclid_polish.web.helpers.q1_star_counts import (
    Q1_DEEP_FIELD_AREA_ARCMIN2,
    Q1_DEEP_FIELD_AREA_DEG2,
)

Q1_GALAXY_RADIUS_VERSION = 1
Q1_GALAXY_RADIUS_SELECTION_VERSION = 1
Q1_GALAXY_RADIUS_MIN_ARCSEC = 0.03
Q1_GALAXY_RADIUS_MAX_ARCSEC = 10.0
Q1_GALAXY_RADIUS_BIN_COUNT = 30

_QUERY_LOCK = threading.Lock()
_Q1_DEEP_FIELD_REGIONS = (
    (269.733, 66.018, 6.0),
    (61.241, -48.423, 6.0),
    (52.932, -28.088, 6.0),
)


def q1_galaxy_radius_statistics_path() -> Path:
    return (
        Path(Config.DATA_DIR)
        / "population_comparison"
        / "q1_galaxy_sersic_radius_statistics.json"
    )


def _finite(value: Any) -> float | None:
    if value is None or np.ma.is_masked(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _result_value(row: Any, name: str) -> float | None:
    for candidate in (name, name.lower(), name.upper()):
        try:
            return _finite(row[candidate])
        except (KeyError, IndexError, TypeError):
            continue
    return None


def _deep_field_predicate() -> str:
    clauses = [
        (
            "CONTAINS(POINT('ICRS', mer.right_ascension, mer.declination), "
            f"CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1"
        )
        for ra, dec, radius in _Q1_DEEP_FIELD_REGIONS
    ]
    return "(\n        " + "\n        OR ".join(clauses) + "\n      )"


def _base_selection() -> str:
    return """
      AND mer.point_like_flag IS NULL
      AND cls.phz_gal_prob IS NOT NULL
      AND cls.phz_gal_prob >= 0.5
      AND cls.phz_gal_prob <= 1.0
      AND morph.sersic_visnir_flags = 0
      AND morph.sersic_sersic_vis_radius IS NOT NULL
      AND morph.sersic_sersic_vis_radius > 0
    """


def _magnitude_moment_query(lower: float, upper: float) -> str:
    faint_flux = float(ab_mag_to_uJy(upper))
    bright_flux = float(ab_mag_to_uJy(lower))
    return f"""
    SELECT
        COUNT(*) AS selected_radii,
        SUM(cls.phz_gal_prob) AS expected_radii,
        SUM(cls.phz_gal_prob * morph.sersic_sersic_vis_radius)
            AS weighted_radius_sum,
        SUM(cls.phz_gal_prob * morph.sersic_sersic_vis_radius
            * morph.sersic_sersic_vis_radius) AS weighted_radius2_sum
    FROM catalogue.mer_catalogue AS mer
    JOIN catalogue.phz_classification AS cls
      ON mer.object_id = cls.object_id
    JOIN catalogue.mer_morphology AS morph
      ON mer.object_id = morph.object_id
    WHERE {_deep_field_predicate()}
      AND mer.flux_vis_2fwhm_aper > {faint_flux:.16g}
      AND mer.flux_vis_2fwhm_aper <= {bright_flux:.16g}
      {_base_selection()}
    """


def _radius_bin_query(lower: float, upper: float) -> str:
    faint_flux = float(ab_mag_to_uJy(Q1_GALAXY_MAG_FAINT))
    bright_flux = float(ab_mag_to_uJy(Q1_GALAXY_MAG_BRIGHT))
    return f"""
    SELECT
        COUNT(*) AS selected_radii,
        SUM(cls.phz_gal_prob) AS expected_radii
    FROM catalogue.mer_catalogue AS mer
    JOIN catalogue.phz_classification AS cls
      ON mer.object_id = cls.object_id
    JOIN catalogue.mer_morphology AS morph
      ON mer.object_id = morph.object_id
    WHERE {_deep_field_predicate()}
      AND mer.flux_vis_2fwhm_aper > {faint_flux:.16g}
      AND mer.flux_vis_2fwhm_aper <= {bright_flux:.16g}
      AND morph.sersic_sersic_vis_radius >= {lower:.16g}
      AND morph.sersic_sersic_vis_radius < {upper:.16g}
      {_base_selection()}
    """


def _launch_with_relogin(
    query: str,
    relogin: Callable[[], bool] | None,
) -> Any:
    job = Euclid.launch_job_async(query)
    if job is None and relogin is not None:
        try:
            refreshed = relogin()
        except Exception:  # noqa: BLE001 - surface the archive failure below
            refreshed = False
        if refreshed:
            job = Euclid.launch_job_async(query)
    if job is None:
        raise RuntimeError("The Euclid archive rejected the Q1 radius query")
    results = job.get_results()
    if results is None or len(results) != 1:
        raise RuntimeError("The Q1 radius query returned no aggregate row")
    return results[0]


def query_q1_galaxy_radius_statistics(
    *,
    bright: float = Q1_GALAXY_MAG_BRIGHT,
    faint: float = Q1_GALAXY_MAG_FAINT,
    bin_width: float = Q1_GALAXY_MAG_BIN_WIDTH,
    progressive_stride: float = Q1_GALAXY_PROGRESSIVE_STRIDE,
    relogin: Callable[[], bool] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Query bracketed radius moments and a marginal radius histogram."""
    lower, upper, width = float(bright), float(faint), float(bin_width)
    stride = float(progressive_stride)
    if not (
        math.isfinite(lower) and math.isfinite(upper) and upper > lower
        and math.isfinite(width) and width > 0.0
        and math.isfinite(stride) and stride >= width
    ):
        raise ValueError("Q1 radius-query limits are invalid")
    bin_count = int(round((upper - lower) / width))
    stride_bins = int(round(stride / width))
    if (
        bin_count <= 0
        or not math.isclose(lower + bin_count * width, upper, abs_tol=1e-9)
        or not math.isclose(stride_bins * width, stride, abs_tol=1e-9)
    ):
        raise ValueError("Q1 radius-query ranges must divide into exact bins")

    magnitude_edges = np.linspace(lower, upper, bin_count + 1)
    radius_edges = np.geomspace(
        Q1_GALAXY_RADIUS_MIN_ARCSEC,
        Q1_GALAXY_RADIUS_MAX_ARCSEC,
        Q1_GALAXY_RADIUS_BIN_COUNT + 1,
    )
    total_queries = bin_count + Q1_GALAXY_RADIUS_BIN_COUNT
    output = q1_galaxy_radius_statistics_path()

    def compatible_cache() -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
        try:
            saved = read_q1_galaxy_radius_statistics()
        except ValueError:
            return {}, {}
        if not (
            math.isclose(float(saved["bright"]), lower, abs_tol=1e-9)
            and math.isclose(float(saved["faint"]), upper, abs_tol=1e-9)
            and math.isclose(float(saved["bin_width"]), width, abs_tol=1e-9)
            and int(saved.get("stride_bins") or 0) == stride_bins
        ):
            return {}, {}
        moments = {
            int(item["bin_index"]): item
            for item in saved["magnitude_bins"]
        }
        radii = {
            int(item["bin_index"]): item
            for item in saved["radius_bins"]
        }
        return moments, radii

    def make_payload(
        moments: dict[int, dict[str, Any]],
        radii: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        ordered_moments = [moments[index] for index in sorted(moments)]
        ordered_radii = [radii[index] for index in sorted(radii)]
        completed = len(ordered_moments) + len(ordered_radii)
        return {
            "version": Q1_GALAXY_RADIUS_VERSION,
            "selection_version": Q1_GALAXY_RADIUS_SELECTION_VERSION,
            "kind": "q1_mer_phz_vis2fwhm_sersic_re_aggregate",
            "survey": "Euclid Q1 deep fields",
            "fields": ["EDF-N", "EDF-S", "EDF-F"],
            "footprint_area_deg2": Q1_DEEP_FIELD_AREA_DEG2,
            "footprint_area_arcmin2": Q1_DEEP_FIELD_AREA_ARCMIN2,
            "bright": lower,
            "faint": upper,
            "bin_width": width,
            "progressive_stride": stride,
            "stride_bins": stride_bins,
            "magnitude_edges": magnitude_edges.tolist(),
            "radius_edges_arcsec": radius_edges.tolist(),
            "magnitude_bins": ordered_moments,
            "radius_bins": ordered_radii,
            "completed_queries": completed,
            "total_queries": total_queries,
            "complete": completed == total_queries,
            "selection": (
                "Q1 deep fields; VIS 2FWHM magnitude bracket; "
                "POINT_LIKE_FLAG IS NULL; PHZ_GAL_PROB >= 0.5; "
                "positive MER VIS Sersic R_e; SERSIC_VISNIR_FLAGS = 0"
            ),
            "acquisition": (
                "aggregate magnitude brackets and aggregate radius bins; "
                "no object rows and no random sky-position sampling"
            ),
        }

    def checkpoint(payload: dict[str, Any]) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, output)

    with _QUERY_LOCK:
        moments, radii = compatible_cache()
        completed = len(moments) + len(radii)
        if progress and completed:
            progress(completed, total_queries, f"resume {completed}/{total_queries} radius aggregates")
        for phase in range(min(stride_bins, bin_count)):
            for index in range(phase, bin_count, stride_bins):
                if index in moments:
                    continue
                mag_lo = float(magnitude_edges[index])
                mag_hi = float(magnitude_edges[index + 1])
                if progress:
                    progress(
                        len(moments) + len(radii), total_queries,
                        f"Sersic R_e moments · {mag_lo:g} <= VIS 2FWHM < {mag_hi:g}",
                    )
                row = _launch_with_relogin(
                    _magnitude_moment_query(mag_lo, mag_hi), relogin,
                )
                selected = _result_value(row, "selected_radii")
                expected = _result_value(row, "expected_radii")
                radius_sum = _result_value(row, "weighted_radius_sum")
                radius2_sum = _result_value(row, "weighted_radius2_sum")
                if selected is None or selected < 0.0:
                    raise RuntimeError("Q1 radius query returned an invalid row count")
                expected = 0.0 if expected is None else expected
                radius_sum = 0.0 if radius_sum is None else radius_sum
                radius2_sum = 0.0 if radius2_sum is None else radius2_sum
                if min(expected, radius_sum, radius2_sum) < 0.0:
                    raise RuntimeError("Q1 radius query returned invalid weighted moments")
                moments[index] = {
                    "bin_index": index,
                    "phase": phase,
                    "mag_lo": mag_lo,
                    "mag_hi": mag_hi,
                    "selected_radii": int(selected),
                    "expected_radii": expected,
                    "weighted_radius_sum_arcsec": radius_sum,
                    "weighted_radius2_sum_arcsec2": radius2_sum,
                }
                checkpoint(make_payload(moments, radii))
        for index, (radius_lo, radius_hi) in enumerate(
            zip(radius_edges[:-1], radius_edges[1:], strict=True)
        ):
            if index in radii:
                continue
            if progress:
                progress(
                    len(moments) + len(radii), total_queries,
                    f"VIS Sersic R_e bin {index + 1}/{len(radius_edges) - 1}",
                )
            row = _launch_with_relogin(
                _radius_bin_query(float(radius_lo), float(radius_hi)), relogin,
            )
            selected = _result_value(row, "selected_radii")
            expected = _result_value(row, "expected_radii")
            if selected is None or selected < 0.0:
                raise RuntimeError("Q1 radius-bin query returned an invalid row count")
            expected = 0.0 if expected is None else expected
            if expected < 0.0:
                raise RuntimeError("Q1 radius-bin query returned an invalid weight")
            dex_width = math.log10(float(radius_hi) / float(radius_lo))
            radii[index] = {
                "bin_index": index,
                "radius_lo_arcsec": float(radius_lo),
                "radius_hi_arcsec": float(radius_hi),
                "selected_radii": int(selected),
                "expected_radii": expected,
                "density_arcmin2_dex": (
                    expected / Q1_DEEP_FIELD_AREA_ARCMIN2 / dex_width
                ),
            }
            checkpoint(make_payload(moments, radii))
        payload = make_payload(moments, radii)
        if progress:
            progress(total_queries, total_queries, "Q1 Sersic-radius aggregates cached")
        return payload


def read_q1_galaxy_radius_statistics() -> dict[str, Any]:
    """Read the radius aggregates only when their contract is current."""
    try:
        payload = json.loads(q1_galaxy_radius_statistics_path().read_text())
        magnitude_edges = np.asarray(payload["magnitude_edges"], dtype=np.float64)
        radius_edges = np.asarray(payload["radius_edges_arcsec"], dtype=np.float64)
        magnitude_bins = payload["magnitude_bins"]
        radius_bins = payload["radius_bins"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Query Q1 aggregate Sersic-radius statistics first") from exc
    if (
        payload.get("version") != Q1_GALAXY_RADIUS_VERSION
        or payload.get("selection_version") != Q1_GALAXY_RADIUS_SELECTION_VERSION
        or payload.get("kind") != "q1_mer_phz_vis2fwhm_sersic_re_aggregate"
        or magnitude_edges.size < 2
        or radius_edges.size < 2
        or not np.all(np.isfinite(magnitude_edges))
        or not np.all(np.isfinite(radius_edges) & (radius_edges > 0.0))
        or not np.all(np.diff(magnitude_edges) > 0.0)
        or not np.all(np.diff(radius_edges) > 0.0)
    ):
        raise ValueError("Q1 aggregate Sersic-radius cache is stale or malformed")
    completed = len(magnitude_bins) + len(radius_bins)
    if completed != int(payload.get("completed_queries", -1)):
        raise ValueError("Q1 aggregate Sersic-radius cache is malformed")
    if len({int(item["bin_index"]) for item in magnitude_bins}) != len(magnitude_bins):
        raise ValueError("Q1 magnitude-radius brackets are duplicated")
    if len({int(item["bin_index"]) for item in radius_bins}) != len(radius_bins):
        raise ValueError("Q1 radius brackets are duplicated")
    return payload
