"""Q1-wide probability-weighted point-source and PHZ stellar counts."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from astroquery.esa.euclid import Euclid

from euclid_polish.config import Config
from euclid_polish.photometry import ab_mag_to_uJy

Q1_DEEP_FIELD_AREA_DEG2 = 63.1
Q1_DEEP_FIELD_AREA_ARCMIN2 = Q1_DEEP_FIELD_AREA_DEG2 * 3600.0
Q1_STAR_COUNT_VERSION = 3

# These broad regions select the three released deep fields while excluding
# the separate LDN 1641 commissioning field. The normalization is the released
# Q1 deep-field footprint area, not the area of these selection circles.
_Q1_DEEP_FIELD_REGIONS = (
    (269.733, 66.018, 6.0),  # EDF-N
    (61.241, -48.423, 6.0),  # EDF-S
    (52.932, -28.088, 6.0),  # EDF-F
)


def q1_star_counts_path() -> Path:
    return (
        Path(Config.DATA_DIR)
        / "population_comparison"
        / "q1_phz_star_counts.json"
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
    candidates = (name, name.lower(), name.upper())
    for candidate in candidates:
        try:
            return _finite(row[candidate])
        except (KeyError, IndexError, TypeError):
            continue
    return None


def _ab_flux_ujy(magnitude: float) -> float:
    return float(ab_mag_to_uJy(float(magnitude)))


def _deep_field_predicate() -> str:
    clauses = [
        (
            "CONTAINS(POINT('ICRS', mer.right_ascension, mer.declination), "
            f"CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1"
        )
        for ra, dec, radius in _Q1_DEEP_FIELD_REGIONS
    ]
    return "(\n        " + "\n        OR ".join(clauses) + "\n      )"


def _magnitude_predicate(lower: float, upper: float) -> str:
    # AB magnitude increases as flux decreases. Use the VIS PSF flux because
    # this is a point-source population and the archive stores it in microJy.
    faint_flux = _ab_flux_ujy(upper)
    bright_flux = _ab_flux_ujy(lower)
    return f"""{_deep_field_predicate()}
      AND mer.flux_vis_psf > {faint_flux:.16g}
      AND mer.flux_vis_psf <= {bright_flux:.16g}
      AND mer.point_like_prob >= 0.9
      AND mer.point_like_prob <= 1.0"""


def _point_source_bin_query(lower: float, upper: float) -> str:
    return f"""
    SELECT
        COUNT(*) AS selected_point_sources,
        SUM(mer.point_like_prob) AS expected_point_sources,
        SUM(mer.point_like_prob * (1.0 - mer.point_like_prob))
            AS point_source_variance
    FROM catalogue.mer_catalogue AS mer
    WHERE {_magnitude_predicate(lower, upper)}
    """


def _phz_bin_query(lower: float, upper: float) -> str:
    return f"""
    SELECT
        COUNT(*) AS classified_rows,
        SUM(cls.phz_star_prob) AS expected_stars,
        SUM(cls.phz_star_prob * (1.0 - cls.phz_star_prob))
            AS classification_variance
    FROM catalogue.mer_catalogue AS mer
    JOIN catalogue.phz_classification AS cls
      ON mer.object_id = cls.object_id
    WHERE {_magnitude_predicate(lower, upper)}
      AND cls.phz_star_prob IS NOT NULL
      AND cls.phz_star_prob >= 0.0
      AND cls.phz_star_prob <= 1.0
    """


def _launch_with_relogin(
    query: str,
    relogin: Callable[[], bool] | None,
    label: str,
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
        raise RuntimeError(f"The Euclid archive rejected the Q1 {label} query")
    results = job.get_results()
    if results is None or len(results) != 1:
        raise RuntimeError(f"The Q1 {label} query returned no aggregate row")
    return results[0]


def query_q1_phz_star_counts(
    *,
    bright: float | None = None,
    faint: float | None = None,
    bin_width: float = 0.1,
    relogin: Callable[[], bool] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Query PHZ-weighted VIS star counts over the released Q1 footprint.

    Each archive request returns only three aggregate scalars. The live cache
    is replaced only after every requested magnitude bin succeeds.
    """
    lower = float(Config.STAR_MAG_BRIGHT if bright is None else bright)
    upper = float(Config.STAR_MAG_FAINT if faint is None else faint)
    width = float(bin_width)
    if not (math.isfinite(lower) and math.isfinite(upper) and upper > lower):
        raise ValueError("Q1 stellar magnitude limits must be finite and ordered")
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("Q1 stellar magnitude-bin width must be positive")
    bin_count = int(round((upper - lower) / width))
    if bin_count <= 0 or not math.isclose(
        lower + bin_count * width, upper, rel_tol=0.0, abs_tol=1e-9,
    ):
        raise ValueError("Q1 stellar magnitude range must be divisible by bin width")

    edges = [lower + index * width for index in range(bin_count + 1)]
    bins: list[dict[str, Any]] = []
    for index, (mag_lo, mag_hi) in enumerate(
        zip(edges[:-1], edges[1:], strict=True),
    ):
        if progress:
            progress(
                2 * index,
                2 * bin_count,
                f"Q1 point sources {mag_lo:g} <= VIS < {mag_hi:g}",
            )
        point_row = _launch_with_relogin(
            _point_source_bin_query(mag_lo, mag_hi),
            relogin,
            "point-source count",
        )
        selected_point_sources = _result_value(
            point_row, "selected_point_sources"
        )
        expected_point_sources = _result_value(
            point_row, "expected_point_sources"
        )
        point_source_variance = _result_value(
            point_row, "point_source_variance"
        )
        if selected_point_sources is None or selected_point_sources < 0.0:
            raise RuntimeError("Q1 point-source query returned an invalid row count")
        expected_point_sources = (
            0.0 if expected_point_sources is None else expected_point_sources
        )
        point_source_variance = (
            0.0 if point_source_variance is None else point_source_variance
        )
        if expected_point_sources < 0.0 or point_source_variance < 0.0:
            raise RuntimeError("Q1 point-source query returned invalid probabilities")

        if progress:
            progress(
                2 * index + 1,
                2 * bin_count,
                f"Q1 PHZ stars {mag_lo:g} <= VIS < {mag_hi:g}",
            )
        phz_row = _launch_with_relogin(
            _phz_bin_query(mag_lo, mag_hi), relogin, "PHZ count"
        )
        classified_rows = _result_value(phz_row, "classified_rows")
        expected_stars = _result_value(phz_row, "expected_stars")
        variance = _result_value(phz_row, "classification_variance")
        if classified_rows is None or classified_rows < 0.0:
            raise RuntimeError("Q1 PHZ count query returned an invalid row count")
        # SQL SUM is NULL for an empty bin; its expected count and variance are 0.
        expected_stars = 0.0 if expected_stars is None else expected_stars
        variance = 0.0 if variance is None else variance
        if expected_stars < 0.0 or variance < 0.0:
            raise RuntimeError("Q1 PHZ count query returned invalid probabilities")
        bins.append({
            "mag_lo": mag_lo,
            "mag_hi": mag_hi,
            "selected_point_sources": int(selected_point_sources),
            "expected_point_sources": expected_point_sources,
            "point_source_variance": point_source_variance,
            "point_source_density_arcmin2_mag": (
                expected_point_sources / Q1_DEEP_FIELD_AREA_ARCMIN2 / width
            ),
            "classified_rows": int(classified_rows),
            "expected_stars": expected_stars,
            "classification_variance": variance,
            "density_arcmin2_mag": (
                expected_stars / Q1_DEEP_FIELD_AREA_ARCMIN2 / width
            ),
            "classification_sigma_arcmin2_mag": (
                math.sqrt(variance) / Q1_DEEP_FIELD_AREA_ARCMIN2 / width
            ),
        })

    payload = {
        "version": Q1_STAR_COUNT_VERSION,
        "survey": "Euclid Q1 deep fields",
        "fields": ["EDF-N", "EDF-S", "EDF-F"],
        "footprint_area_deg2": Q1_DEEP_FIELD_AREA_DEG2,
        "footprint_area_arcmin2": Q1_DEEP_FIELD_AREA_ARCMIN2,
        "magnitude_system": "AB",
        "magnitude_field": "MER FLUX_VIS_PSF",
        "classification_field": "PHZ_STAR_PROB",
        "selection": (
            "three Q1 deep-field regions; positive VIS PSF flux; "
            "finite POINT_LIKE_PROB >= 0.9 and <= 1; PHZ quantities use only "
            "finite 0 <= PHZ_STAR_PROB <= 1; no additional MER quality cut"
        ),
        "edges": edges,
        "bins": bins,
        "selected_point_sources": int(sum(
            item["selected_point_sources"] for item in bins
        )),
        "expected_point_sources": float(sum(
            item["expected_point_sources"] for item in bins
        )),
        "point_source_variance": float(sum(
            item["point_source_variance"] for item in bins
        )),
        "expected_stars": float(sum(item["expected_stars"] for item in bins)),
        "classification_variance": float(sum(
            item["classification_variance"] for item in bins
        )),
    }
    output = q1_star_counts_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, output)
    if progress:
        progress(2 * bin_count, 2 * bin_count, "Q1 stellar counts cached")
    return payload


def read_q1_phz_star_counts(
    *, bright: float | None = None, faint: float | None = None,
) -> dict[str, Any]:
    """Read and validate the Q1 count artifact required by stellar fitting."""
    try:
        payload = json.loads(q1_star_counts_path().read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Query Q1 PHZ stellar number counts before fitting") from exc
    lower = float(Config.STAR_MAG_BRIGHT if bright is None else bright)
    upper = float(Config.STAR_MAG_FAINT if faint is None else faint)
    try:
        edges = np.asarray(payload["edges"], dtype=np.float64)
        bins = list(payload["bins"])
        area = float(payload["footprint_area_arcmin2"])
        counts = np.asarray(
            [item["expected_stars"] for item in bins], dtype=np.float64,
        )
        point_source_counts = np.asarray(
            [item["expected_point_sources"] for item in bins],
            dtype=np.float64,
        )
        selected_point_sources = np.asarray(
            [item["selected_point_sources"] for item in bins],
            dtype=np.float64,
        )
        variances = np.asarray(
            [item["classification_variance"] for item in bins], dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Q1 PHZ stellar count cache is malformed") from exc
    if (
        payload.get("version") != Q1_STAR_COUNT_VERSION
        or edges.size != counts.size + 1
        or not np.all(np.isfinite(edges))
        or not np.all(np.diff(edges) > 0.0)
        or not np.allclose(np.diff(edges), 0.1, rtol=0.0, atol=1e-9)
        or not math.isclose(edges[0], lower, abs_tol=1e-9)
        or not math.isclose(edges[-1], upper, abs_tol=1e-9)
        or not np.all(np.isfinite(counts)) or np.any(counts < 0.0)
        or not np.all(np.isfinite(point_source_counts))
        or np.any(point_source_counts < 0.0)
        or not np.all(np.isfinite(selected_point_sources))
        or np.any(selected_point_sources < 0.0)
        or not np.all(np.isfinite(variances)) or np.any(variances < 0.0)
        or not math.isclose(area, Q1_DEEP_FIELD_AREA_ARCMIN2, abs_tol=1e-6)
    ):
        raise ValueError("Q1 PHZ stellar count cache is incompatible with this fit")
    return payload
