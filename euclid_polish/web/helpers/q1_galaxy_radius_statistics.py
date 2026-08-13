"""Aggregate Q1 VIS-2FWHM x Sersic-radius statistics.

The archive returns only a bounded, two-dimensional histogram.  No object
catalogue or random sky-position sample is materialized by this workflow.
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Callable, Iterable
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

Q1_GALAXY_RADIUS_VERSION = 2
Q1_GALAXY_RADIUS_SELECTION_VERSION = 2
Q1_GALAXY_RADIUS_MIN_ARCSEC = 0.03
Q1_GALAXY_RADIUS_MAX_ARCSEC = 10.0
Q1_GALAXY_RADIUS_BIN_COUNT = 30
# Keep every grouped TAP response below the common 2,000-row archive limit.
Q1_GALAXY_RADIUS_MAG_BINS_PER_QUERY = 35
Q1_GALAXY_RADIUS_TOTAL_QUERIES = math.ceil(
    round(
        (Q1_GALAXY_MAG_FAINT - Q1_GALAXY_MAG_BRIGHT)
        / Q1_GALAXY_MAG_BIN_WIDTH
    )
    / Q1_GALAXY_RADIUS_MAG_BINS_PER_QUERY
)

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
    return f"""
      AND mer.point_like_flag IS NULL
      AND cls.phz_gal_prob IS NOT NULL
      AND cls.phz_gal_prob >= 0.5
      AND cls.phz_gal_prob <= 1.0
      AND morph.sersic_visnir_flags = 0
      AND morph.sersic_sersic_vis_radius >= {Q1_GALAXY_RADIUS_MIN_ARCSEC:.16g}
      AND morph.sersic_sersic_vis_radius < {Q1_GALAXY_RADIUS_MAX_ARCSEC:.16g}
    """


def _joint_histogram_query(
    *,
    bright: float,
    chunk_bright: float,
    chunk_faint: float,
    bin_width: float,
    radius_bin_count: int,
) -> str:
    faint_flux = float(ab_mag_to_uJy(chunk_faint))
    bright_flux = float(ab_mag_to_uJy(chunk_bright))
    radius_log_min = math.log10(Q1_GALAXY_RADIUS_MIN_ARCSEC)
    radius_log_width = (
        math.log10(Q1_GALAXY_RADIUS_MAX_ARCSEC) - radius_log_min
    ) / radius_bin_count
    magnitude_bin = (
        f"FLOOR(({float(Config.AB_ZP_UJY):.16g} - "
        "2.5 * LOG10(mer.flux_vis_2fwhm_aper) - "
        f"{bright:.16g}) / {bin_width:.16g})"
    )
    radius_bin = (
        "FLOOR((LOG10(morph.sersic_sersic_vis_radius) - "
        f"({radius_log_min:.16g})) / {radius_log_width:.16g})"
    )
    return f"""
    SELECT
        {magnitude_bin} AS magnitude_bin,
        {radius_bin} AS radius_bin,
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
      {_base_selection()}
    GROUP BY magnitude_bin, radius_bin
    """


def _launch_with_relogin(
    query: str,
    relogin: Callable[[], bool] | None,
) -> Iterable[Any]:
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
    if results is None:
        raise RuntimeError("The Q1 radius query returned no grouped result")
    return results


def query_q1_galaxy_radius_statistics(
    *,
    bright: float = Q1_GALAXY_MAG_BRIGHT,
    faint: float = Q1_GALAXY_MAG_FAINT,
    bin_width: float = Q1_GALAXY_MAG_BIN_WIDTH,
    progressive_stride: float = Q1_GALAXY_PROGRESSIVE_STRIDE,
    relogin: Callable[[], bool] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Query bounded joint magnitude-radius bins in compact TAP chunks."""
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
    chunk_starts = list(range(
        0, bin_count, Q1_GALAXY_RADIUS_MAG_BINS_PER_QUERY,
    ))
    total_queries = len(chunk_starts)
    output = q1_galaxy_radius_statistics_path()

    def make_payload(
        joint: dict[tuple[int, int], dict[str, Any]], completed: int,
    ) -> dict[str, Any]:
        magnitude_selected = np.zeros(bin_count, dtype=np.int64)
        magnitude_expected = np.zeros(bin_count, dtype=np.float64)
        radius_selected = np.zeros(Q1_GALAXY_RADIUS_BIN_COUNT, dtype=np.int64)
        radius_expected = np.zeros(Q1_GALAXY_RADIUS_BIN_COUNT, dtype=np.float64)
        for (mag_index, radius_index), item in joint.items():
            magnitude_selected[mag_index] += int(item["selected_radii"])
            magnitude_expected[mag_index] += float(item["expected_radii"])
            radius_selected[radius_index] += int(item["selected_radii"])
            radius_expected[radius_index] += float(item["expected_radii"])
        magnitude_bins = [
            {
                "bin_index": index,
                "mag_lo": float(magnitude_edges[index]),
                "mag_hi": float(magnitude_edges[index + 1]),
                "selected_radii": int(magnitude_selected[index]),
                "expected_radii": float(magnitude_expected[index]),
            }
            for index in range(bin_count)
        ]
        radius_bins = []
        for index, (radius_lo, radius_hi) in enumerate(zip(
            radius_edges[:-1], radius_edges[1:], strict=True,
        )):
            dex_width = math.log10(float(radius_hi) / float(radius_lo))
            radius_bins.append({
                "bin_index": index,
                "radius_lo_arcsec": float(radius_lo),
                "radius_hi_arcsec": float(radius_hi),
                "selected_radii": int(radius_selected[index]),
                "expected_radii": float(radius_expected[index]),
                "density_arcmin2_dex": float(
                    radius_expected[index]
                    / Q1_DEEP_FIELD_AREA_ARCMIN2
                    / dex_width
                ),
            })
        return {
            "version": Q1_GALAXY_RADIUS_VERSION,
            "selection_version": Q1_GALAXY_RADIUS_SELECTION_VERSION,
            "kind": "q1_mer_phz_vis2fwhm_sersic_re_joint_aggregate",
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
            "joint_bins": [joint[key] for key in sorted(joint)],
            "magnitude_bins": magnitude_bins,
            "radius_bins": radius_bins,
            "completed_queries": completed,
            "total_queries": total_queries,
            "complete": completed == total_queries,
            "selection": (
                "Q1 deep fields; VIS 2FWHM magnitude; POINT_LIKE_FLAG IS "
                "NULL; PHZ_GAL_PROB >= 0.5; 0.03 <= MER VIS Sersic R_e "
                "< 10 arcsec; SERSIC_VISNIR_FLAGS = 0"
            ),
            "acquisition": (
                "grouped aggregate magnitude x log-radius bins; no object "
                "rows and no random sky-position sampling"
            ),
        }

    def checkpoint(payload: dict[str, Any]) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, output)

    with _QUERY_LOCK:
        joint: dict[tuple[int, int], dict[str, Any]] = {}
        for query_index, start in enumerate(chunk_starts):
            stop = min(start + Q1_GALAXY_RADIUS_MAG_BINS_PER_QUERY, bin_count)
            chunk_bright = float(magnitude_edges[start])
            chunk_faint = float(magnitude_edges[stop])
            if progress:
                progress(
                    query_index, total_queries,
                    f"Sersic R_e histogram · {chunk_bright:g} <= VIS "
                    f"2FWHM < {chunk_faint:g}",
                )
            rows = _launch_with_relogin(
                _joint_histogram_query(
                    bright=lower,
                    chunk_bright=chunk_bright,
                    chunk_faint=chunk_faint,
                    bin_width=width,
                    radius_bin_count=Q1_GALAXY_RADIUS_BIN_COUNT,
                ),
                relogin,
            )
            for row in rows:
                mag_value = _result_value(row, "magnitude_bin")
                radius_value = _result_value(row, "radius_bin")
                selected = _result_value(row, "selected_radii")
                expected = _result_value(row, "expected_radii")
                if mag_value is None or radius_value is None:
                    raise RuntimeError("Q1 joint histogram returned an invalid bin")
                mag_index = int(round(mag_value))
                radius_index = int(round(radius_value))
                if (
                    not math.isclose(mag_value, mag_index, abs_tol=1e-9)
                    or not math.isclose(radius_value, radius_index, abs_tol=1e-9)
                    or not start <= mag_index < stop
                    or not 0 <= radius_index < Q1_GALAXY_RADIUS_BIN_COUNT
                    or selected is None or selected < 0.0
                    or (expected is not None and expected < 0.0)
                ):
                    raise RuntimeError("Q1 joint histogram returned an invalid row")
                key = (mag_index, radius_index)
                if key in joint:
                    raise RuntimeError("Q1 joint histogram duplicated a bin")
                joint[key] = {
                    "magnitude_bin": mag_index,
                    "radius_bin": radius_index,
                    "selected_radii": int(round(selected)),
                    "expected_radii": 0.0 if expected is None else expected,
                }
            checkpoint(make_payload(joint, query_index + 1))
        payload = make_payload(joint, total_queries)
        if progress:
            progress(total_queries, total_queries, "Q1 joint Sersic-radius histogram cached")
        return payload


def read_q1_galaxy_radius_statistics() -> dict[str, Any]:
    """Read the bounded joint histogram only when its contract is current."""
    try:
        payload = json.loads(q1_galaxy_radius_statistics_path().read_text())
        magnitude_edges = np.asarray(payload["magnitude_edges"], dtype=np.float64)
        radius_edges = np.asarray(payload["radius_edges_arcsec"], dtype=np.float64)
        joint_bins = payload["joint_bins"]
        magnitude_bins = payload["magnitude_bins"]
        radius_bins = payload["radius_bins"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Query Q1 aggregate Sersic-radius statistics first") from exc
    if (
        payload.get("version") != Q1_GALAXY_RADIUS_VERSION
        or payload.get("selection_version") != Q1_GALAXY_RADIUS_SELECTION_VERSION
        or payload.get("kind")
        != "q1_mer_phz_vis2fwhm_sersic_re_joint_aggregate"
        or magnitude_edges.size < 2
        or radius_edges.size < 2
        or len(magnitude_bins) != magnitude_edges.size - 1
        or len(radius_bins) != radius_edges.size - 1
        or not np.all(np.isfinite(magnitude_edges))
        or not np.all(np.isfinite(radius_edges) & (radius_edges > 0.0))
        or not np.all(np.diff(magnitude_edges) > 0.0)
        or not np.all(np.diff(radius_edges) > 0.0)
        or int(payload.get("completed_queries", -1))
        != int(payload.get("total_queries", -2))
        or not payload.get("complete")
    ):
        raise ValueError("Q1 aggregate Sersic-radius cache is stale or malformed")
    keys: set[tuple[int, int]] = set()
    for item in joint_bins:
        try:
            key = (int(item["magnitude_bin"]), int(item["radius_bin"]))
            selected = int(item["selected_radii"])
            expected = float(item["expected_radii"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Q1 joint histogram is malformed") from exc
        if (
            key in keys
            or not 0 <= key[0] < magnitude_edges.size - 1
            or not 0 <= key[1] < radius_edges.size - 1
            or selected < 0
            or not math.isfinite(expected)
            or expected < 0.0
        ):
            raise ValueError("Q1 joint histogram is malformed")
        keys.add(key)
    return payload
