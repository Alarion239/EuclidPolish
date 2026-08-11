"""Q1-wide PHZ-selected galaxy counts in the four VIS apertures."""

from __future__ import annotations

import hashlib
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
from euclid_polish.population.magnitude_law import (
    StraightMagnitudeLaw,
    fit_straight_region,
)
from euclid_polish.web.helpers.q1_star_counts import (
    Q1_DEEP_FIELD_AREA_ARCMIN2,
    Q1_DEEP_FIELD_AREA_DEG2,
)

Q1_GALAXY_COUNT_VERSION = 3
Q1_GALAXY_FIT_VERSION = 2
Q1_GALAXY_SELECTION_VERSION = 2
Q1_GALAXY_MAG_BRIGHT = 14.0
Q1_GALAXY_MAG_FAINT = 28.0
Q1_GALAXY_LAW_FAINT = 29.0
Q1_GALAXY_MAG_BIN_WIDTH = 0.1
Q1_GALAXY_PROGRESSIVE_STRIDE = 0.5

_QUERY_LOCK = threading.Lock()

_Q1_DEEP_FIELD_REGIONS = (
    (269.733, 66.018, 6.0),  # EDF-N
    (61.241, -48.423, 6.0),  # EDF-S
    (52.932, -28.088, 6.0),  # EDF-F
)

Q1_VIS_APERTURES = {
    "f1": ("flux_vis_1fwhm_aper", "VIS · 1 FWHM", "1-FWHM diameter aperture"),
    "f2": ("flux_vis_2fwhm_aper", "VIS · 2 FWHM", "2-FWHM diameter aperture"),
    "f3": ("flux_vis_3fwhm_aper", "VIS · 3 FWHM", "3-FWHM diameter aperture"),
    "f4": ("flux_vis_4fwhm_aper", "VIS · 4 FWHM", "4-FWHM diameter aperture"),
}


def q1_galaxy_counts_path() -> Path:
    return (
        Path(Config.DATA_DIR)
        / "population_comparison"
        / "q1_galaxy_aperture_counts.json"
    )


def q1_galaxy_fit_path() -> Path:
    return (
        Path(Config.DATA_DIR)
        / "population_comparison"
        / "q1_galaxy_aperture_fit.json"
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


def _aperture_bin_query(column: str, lower: float, upper: float) -> str:
    faint_flux = float(ab_mag_to_uJy(upper))
    bright_flux = float(ab_mag_to_uJy(lower))
    return f"""
    SELECT
        COUNT(*) AS selected_galaxies,
        SUM(cls.phz_gal_prob) AS expected_galaxies,
        SUM(cls.phz_gal_prob * (1.0 - cls.phz_gal_prob))
            AS classification_variance
    FROM catalogue.mer_catalogue AS mer
    JOIN catalogue.phz_classification AS cls
      ON mer.object_id = cls.object_id
    WHERE {_deep_field_predicate()}
      AND mer.{column} > {faint_flux:.16g}
      AND mer.{column} <= {bright_flux:.16g}
      AND mer.point_like_flag IS NULL
      AND cls.phz_gal_prob IS NOT NULL
      AND cls.phz_gal_prob >= 0.5
      AND cls.phz_gal_prob <= 1.0
    """


def _launch_with_relogin(
    query: str,
    relogin: Callable[[], bool] | None,
) -> Any:
    job = Euclid.launch_job_async(query)
    if job is None and relogin is not None:
        try:
            refreshed = relogin()
        except Exception:  # noqa: BLE001 - preserve the archive failure below
            refreshed = False
        if refreshed:
            job = Euclid.launch_job_async(query)
    if job is None:
        raise RuntimeError("The Euclid archive rejected the Q1 galaxy-count query")
    results = job.get_results()
    if results is None or len(results) != 1:
        raise RuntimeError("The Q1 galaxy-count query returned no aggregate row")
    return results[0]


def query_q1_galaxy_aperture_counts(
    *,
    bright: float = Q1_GALAXY_MAG_BRIGHT,
    faint: float = Q1_GALAXY_MAG_FAINT,
    bin_width: float = Q1_GALAXY_MAG_BIN_WIDTH,
    progressive_stride: float = Q1_GALAXY_PROGRESSIVE_STRIDE,
    relogin: Callable[[], bool] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Progressively query and checkpoint MER+PHZ counts for each aperture."""
    lower = float(bright)
    upper = float(faint)
    width = float(bin_width)
    if not (math.isfinite(lower) and math.isfinite(upper) and upper > lower):
        raise ValueError("Q1 galaxy magnitude limits must be finite and ordered")
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("Q1 galaxy magnitude-bin width must be positive")
    stride = float(progressive_stride)
    if not math.isfinite(stride) or stride < width:
        raise ValueError(
            "Q1 galaxy progressive stride must be at least one bin wide"
        )
    bin_count = int(round((upper - lower) / width))
    if bin_count <= 0 or not math.isclose(
        lower + bin_count * width,
        upper,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Q1 galaxy magnitude range must be divisible by bin width")
    stride_bins = int(round(stride / width))
    if not math.isclose(
        stride_bins * width, stride, rel_tol=0.0, abs_tol=1e-9,
    ):
        raise ValueError(
            "Q1 galaxy progressive stride must be divisible by bin width"
        )

    edges = [lower + index * width for index in range(bin_count + 1)]
    total_queries = bin_count * len(Q1_VIS_APERTURES)
    output = q1_galaxy_counts_path()

    def compatible_cache() -> dict[tuple[str, int], dict[str, Any]]:
        try:
            saved = read_q1_galaxy_aperture_counts()
        except ValueError:
            return {}
        if not (
            math.isclose(float(saved["bright"]), lower, abs_tol=1e-9)
            and math.isclose(float(saved["bin_width"]), width, abs_tol=1e-9)
            and int(saved.get("stride_bins") or 0) == stride_bins
            and int(saved.get("selection_version") or 0)
            == Q1_GALAXY_SELECTION_VERSION
        ):
            return {}
        cached: dict[tuple[str, int], dict[str, Any]] = {}
        for key, aperture in saved["apertures"].items():
            for item in aperture["bins"]:
                mag_lo = float(item["mag_lo"])
                mag_hi = float(item["mag_hi"])
                index = int(round((mag_lo - lower) / width))
                if (
                    0 <= index < bin_count
                    and math.isclose(
                        edges[index], mag_lo, rel_tol=0.0, abs_tol=1e-9,
                    )
                    and math.isclose(
                        edges[index + 1], mag_hi,
                        rel_tol=0.0, abs_tol=1e-9,
                    )
                ):
                    cached[key, index] = {**item, "bin_index": index}
        return cached

    def make_payload(
        cached: dict[tuple[str, int], dict[str, Any]],
    ) -> dict[str, Any]:
        apertures: dict[str, dict[str, Any]] = {}
        for key, (column, label, estimator) in Q1_VIS_APERTURES.items():
            bins = [
                cached[key, index]
                for index in range(bin_count)
                if (key, index) in cached
            ]
            apertures[key] = {
                "label": label,
                "flux_field": f"MER {column.upper()}",
                "estimator": estimator,
                "bins": bins,
                "queried_bins": len(bins),
                "selected_galaxies": int(sum(
                    item["selected_galaxies"] for item in bins
                )),
                "expected_galaxies": float(sum(
                    item["expected_galaxies"] for item in bins
                )),
            }
        completed = len(cached)
        phase_complete = [
            all(
                (key, index) in cached
                for index in range(phase, bin_count, stride_bins)
                for key in Q1_VIS_APERTURES
            )
            for phase in range(min(stride_bins, bin_count))
        ]
        return {
            "version": Q1_GALAXY_COUNT_VERSION,
            "selection_version": Q1_GALAXY_SELECTION_VERSION,
            "survey": "Euclid Q1 deep fields",
            "fields": ["EDF-N", "EDF-S", "EDF-F"],
            "footprint_area_deg2": Q1_DEEP_FIELD_AREA_DEG2,
            "footprint_area_arcmin2": Q1_DEEP_FIELD_AREA_ARCMIN2,
            "magnitude_system": "AB",
            "bright": lower,
            "faint": upper,
            "bin_width": width,
            "progressive_stride": stride,
            "stride_bins": stride_bins,
            "phase_count": len(phase_complete),
            "phases_completed": sum(phase_complete),
            "edges": edges,
            "selection": (
                "three Q1 deep-field regions; POINT_LIKE_FLAG IS NULL; "
                "PHZ_GAL_PROB >= 0.5; positive flux in the aperture being "
                "binned; no additional MER quality cut"
            ),
            "count_definition": (
                "sum of PHZ_GAL_PROB among non-point-like PHZ galaxies"
            ),
            "apertures": apertures,
            "query_count": completed,
            "completed_queries": completed,
            "total_queries": total_queries,
            "complete": completed == total_queries,
            "fit_ready": any(
                aperture["queried_bins"] >= 4
                for aperture in apertures.values()
            ),
        }

    def checkpoint(payload: dict[str, Any]) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, output)

    with _QUERY_LOCK:
        cached = compatible_cache()
        payload = make_payload(cached)
        if progress and cached:
            progress(
                len(cached), total_queries,
                f"resume {len(cached)}/{total_queries} cached MER + PHZ counts",
            )
        for phase in range(min(stride_bins, bin_count)):
            for index in range(phase, bin_count, stride_bins):
                mag_lo, mag_hi = edges[index], edges[index + 1]
                for key, (column, label, _estimator) in Q1_VIS_APERTURES.items():
                    if (key, index) in cached:
                        continue
                    if progress:
                        progress(
                            len(cached), total_queries,
                            f"phase {phase + 1}/{min(stride_bins, bin_count)} · "
                            f"{label} · {mag_lo:g} <= VIS < {mag_hi:g}",
                        )
                    row = _launch_with_relogin(
                        _aperture_bin_query(column, mag_lo, mag_hi), relogin,
                    )
                    selected = _result_value(row, "selected_galaxies")
                    expected = _result_value(row, "expected_galaxies")
                    variance = _result_value(row, "classification_variance")
                    if selected is None or selected < 0.0:
                        raise RuntimeError(
                            "Q1 galaxy query returned an invalid row count"
                        )
                    expected = 0.0 if expected is None else expected
                    variance = 0.0 if variance is None else variance
                    if expected < 0.0 or variance < 0.0:
                        raise RuntimeError(
                            "Q1 galaxy query returned invalid PHZ probabilities"
                        )
                    count = int(selected)
                    cached[key, index] = {
                        "bin_index": index,
                        "phase": phase,
                        "mag_lo": mag_lo,
                        "mag_hi": mag_hi,
                        "selected_galaxies": count,
                        "expected_galaxies": expected,
                        "classification_variance": variance,
                        "density_arcmin2_mag": (
                            expected / Q1_DEEP_FIELD_AREA_ARCMIN2 / width
                        ),
                        "classification_sigma_arcmin2_mag": (
                            math.sqrt(variance)
                            / Q1_DEEP_FIELD_AREA_ARCMIN2 / width
                        ),
                        "poisson_sigma_arcmin2_mag": (
                            math.sqrt(count)
                            / Q1_DEEP_FIELD_AREA_ARCMIN2 / width
                        ),
                    }
                    payload = make_payload(cached)
                    checkpoint(payload)
                    if progress:
                        progress(
                            len(cached), total_queries,
                            f"cached {len(cached)}/{total_queries} MER + PHZ counts",
                        )
        if progress:
            progress(
                total_queries, total_queries,
                "Q1 MER + PHZ aperture counts cached",
            )
        return payload


def read_q1_galaxy_aperture_counts() -> dict[str, Any]:
    """Read and validate the cached Q1 bright-galaxy aperture counts."""
    try:
        payload = json.loads(q1_galaxy_counts_path().read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Query Q1 galaxy aperture counts first") from exc
    try:
        edges = np.asarray(payload["edges"], dtype=np.float64)
        apertures = payload["apertures"]
        area = float(payload["footprint_area_arcmin2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Q1 galaxy aperture-count cache is malformed") from exc
    if (
        payload.get("version") != Q1_GALAXY_COUNT_VERSION
        or payload.get("selection_version") != Q1_GALAXY_SELECTION_VERSION
        or not np.all(np.isfinite(edges))
        or edges.size < 2
        or area <= 0.0
        or set(apertures) != set(Q1_VIS_APERTURES)
    ):
        raise ValueError("Q1 galaxy aperture-count cache is stale or malformed")
    expected_bins = edges.size - 1
    completed = 0
    for aperture in apertures.values():
        bins = aperture.get("bins", [])
        indices = [int(item.get("bin_index", -1)) for item in bins]
        if (
            len(indices) != len(set(indices))
            or any(index < 0 or index >= expected_bins for index in indices)
        ):
            raise ValueError("Q1 galaxy aperture-count cache is malformed")
        completed += len(indices)
    if completed != int(payload.get("completed_queries", -1)):
        raise ValueError("Q1 galaxy aperture-count cache is malformed")
    return payload


def fit_q1_galaxy_aperture_counts() -> dict[str, Any]:
    """Fit the straight Q1 VIS 2FWHM differential-count law.

    This deliberately fits only the observable supported by the aggregate
    cache. It does not infer redshift, size, mass, or a generator response.
    """
    source_path = q1_galaxy_counts_path()
    payload = read_q1_galaxy_aperture_counts()
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    aperture = payload["apertures"]["f2"]
    bins = aperture["bins"]
    x = np.asarray([
        0.5 * (float(item["mag_lo"]) + float(item["mag_hi"]))
        for item in bins
    ], dtype=np.float64)
    density = np.asarray([
        float(item["density_arcmin2_mag"]) for item in bins
    ], dtype=np.float64)
    area = float(payload["footprint_area_arcmin2"])
    width = float(payload["bin_width"])
    sigma = np.asarray([
        math.sqrt(
            max(float(item["selected_galaxies"]), 0.0)
            + max(float(item["classification_variance"]), 0.0)
        ) / area / width
        for item in bins
    ], dtype=np.float64)
    region = fit_straight_region(
        x, density, sigma,
        minimum_span_mag=4.0,
        minimum_r_squared=0.998,
    )
    law = StraightMagnitudeLaw(
        slope=region.slope,
        intercept=region.intercept,
        mag_bright=Q1_GALAXY_MAG_BRIGHT,
        mag_faint=Q1_GALAXY_LAW_FAINT,
        fit_bright=float(x[region.start]),
        fit_faint=float(x[region.stop - 1]),
        covariance=tuple(
            tuple(float(value) for value in row)
            for row in region.covariance
        ),
        r_squared=region.r_squared,
        rms_log10_density=region.rms,
        source="Euclid Q1 MER FLUX_VIS_2FWHM_APER + PHZ_GAL_PROB",
    )
    grid = np.linspace(Q1_GALAXY_MAG_BRIGHT, Q1_GALAXY_LAW_FAINT, 301)
    curves = {
        "f2": {
            "label": aperture["label"],
            "estimator": aperture["estimator"],
            "law": law.to_payload(),
            "x": grid.tolist(),
            "density": law.density(grid).tolist(),
            "input_bins": int(region.stop - region.start),
            "fit_bin_start": int(region.start),
            "fit_bin_stop": int(region.stop),
            "extrapolated_faint_interval": [
                float(payload["faint"]), Q1_GALAXY_LAW_FAINT,
            ],
        },
    }
    fitted = {
        "version": Q1_GALAXY_FIT_VERSION,
        "kind": "q1_mer_phz_aperture_density_fit",
        "source_counts_sha256": source_sha256,
        "bin_width": float(payload["bin_width"]),
        "footprint_area_deg2": float(payload["footprint_area_deg2"]),
        "method": (
            "widest consecutive positive-count window spanning at least 4 mag "
            "with inverse-variance weighted R^2 >= 0.998; one straight line "
            "in log10 differential density, evaluated over VIS 14-29"
        ),
        "scope": (
            "apparent-brightness aperture curves only; no cone catalogue, "
            "redshift, radius, stellar-mass, or generator-response fit"
        ),
        "apertures": curves,
    }
    output = q1_galaxy_fit_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(fitted, indent=2, sort_keys=True))
    os.replace(temporary, output)
    return fitted


def read_q1_galaxy_aperture_fit() -> dict[str, Any]:
    """Read the aperture fit only when it matches the current count cache."""
    try:
        payload = json.loads(q1_galaxy_fit_path().read_text())
        source_sha256 = hashlib.sha256(
            q1_galaxy_counts_path().read_bytes()
        ).hexdigest()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Fit the cached Q1 aperture counts first") from exc
    if (
        payload.get("version") != Q1_GALAXY_FIT_VERSION
        or payload.get("source_counts_sha256") != source_sha256
        or not isinstance(payload.get("apertures"), dict)
        or not payload["apertures"]
    ):
        raise ValueError("Q1 aperture-count fit is stale or malformed")
    for curve in payload["apertures"].values():
        x = np.asarray(curve.get("x", []), dtype=np.float64)
        density = np.asarray(curve.get("density", []), dtype=np.float64)
        if (
            x.size < 2
            or x.shape != density.shape
            or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(density) & (density > 0.0))
        ):
            raise ValueError("Q1 aperture-count fit is stale or malformed")
        StraightMagnitudeLaw.from_payload(curve.get("law") or {})
    return payload
