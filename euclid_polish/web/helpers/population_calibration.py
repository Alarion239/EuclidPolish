"""Versioned calibration artifacts for galaxy density and stellar priors."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from euclid_polish.config import Config
from euclid_polish.sky.generation.cosmos_tng_prior import (
    brightness_transfer_payload,
)


def calibration_dir() -> Path:
    return Path(Config.DATA_DIR) / "population_comparison" / "calibrations"


def active_transfer_path() -> Path:
    return calibration_dir() / "photometric_transfer_active.json"


def density_calibration_path() -> Path:
    return calibration_dir() / "tng_density_calibration.json"


def active_density_path() -> Path:
    return calibration_dir() / "tng_density_active.json"


def star_candidate_path() -> Path:
    return calibration_dir() / "star_population_candidate.json"


def active_star_path() -> Path:
    return calibration_dir() / "star_population_active.json"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def photometric_candidate() -> dict[str, Any] | None:
    return brightness_transfer_payload(Config.COSMOS_EUCLID_FIT_PATH)


def active_transfer() -> dict[str, Any] | None:
    return _read(active_transfer_path())


def activate_photometric_transfer(
    *, allow_quality_warnings: bool = False,
) -> dict[str, Any]:
    candidate = photometric_candidate()
    if candidate is None:
        raise ValueError("No fixed-normalization photometric fit is available")
    quality = candidate.get("fit_quality") or {}
    if not quality.get("valid", False) and not allow_quality_warnings:
        warnings = "; ".join(quality.get("warnings") or [])
        raise ValueError(
            "Fixed-normalization fit failed its quality gate"
            + (f": {warnings}" if warnings else "")
        )
    payload = {
        **candidate,
        "active": True,
        "validated": bool(quality.get("valid", False)),
        "activated_with_quality_warnings": bool(
            not quality.get("valid", False)
        ),
    }
    _write(active_transfer_path(), payload)
    return payload


def transfer_state() -> dict[str, Any]:
    candidate = photometric_candidate()
    active = active_transfer()
    return {
        "candidate": candidate,
        "active": active,
        "is_active": bool(
            candidate and active
            and candidate.get("fingerprint") == active.get("fingerprint")
        ),
    }


def source_transfer_fingerprints(rows: list[dict[str, Any]]) -> list[str]:
    fingerprints: set[str] = set()
    for row in rows:
        source = str(row.get("brightness_transfer") or "")
        if not source:
            continue
        parts = source.split(":", 2)
        if len(parts) >= 2 and len(parts[1]) == 64:
            fingerprints.add(parts[1])
        else:
            fingerprints.add("legacy:" + hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest())
    return sorted(fingerprints)


def current_transfer_compatibility(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source = source_transfer_fingerprints(rows)
    state = transfer_state()
    active = state.get("active") or {}
    expected = active.get("fingerprint")
    compatible = bool(expected and source == [expected])
    return {
        "compatible": compatible,
        "source_fingerprints": source,
        "active_fingerprint": expected,
        "reason": (
            "matched active fixed-normalization transfer"
            if compatible else
            "not actionable—brightness transfer changed"
        ),
    }


def _isotonic(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted PAVA, returned at the original sorted locations."""
    blocks: list[list[float]] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        blocks.append([float(index), float(index), float(value), float(weight)])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            total = left[3] + right[3]
            mean = (left[2] * left[3] + right[2] * right[3]) / total
            blocks.append([left[0], right[1], mean, total])
    result = np.empty(len(values), dtype=np.float64)
    for start, end, mean, _weight in blocks:
        result[int(start):int(end) + 1] = mean
    return result


def _invert_response(
    densities: np.ndarray, response: np.ndarray, target: float,
) -> float | None:
    if target < response[0] or target > response[-1]:
        return None
    if float(response[-1] - response[0]) <= 1e-6:
        return None
    return float(np.interp(target, response, densities))


def fit_density_response(
    densities: list[float],
    field_detections: list[list[float]],
    euclid_field_detections: list[float],
    *,
    transfer_fingerprint: str,
    active_transfer_fingerprint: str | None,
    field_area_arcmin2: float,
    euclid_cone_detection_densities: list[float] | None = None,
    bootstraps: int = 1000,
    seed: int = 71031,
) -> dict[str, Any]:
    """Fit and invert a paired monotone density-response sweep."""
    x = np.asarray(densities, dtype=np.float64)
    matrix = np.asarray(field_detections, dtype=np.float64)
    real = np.asarray(euclid_field_detections, dtype=np.float64)
    cone_densities = np.asarray(
        euclid_cone_detection_densities or [], dtype=np.float64,
    )
    cone_densities = cone_densities[np.isfinite(cone_densities)]
    if matrix.ndim != 2 or matrix.shape[0] != len(x):
        raise ValueError("density sweep must contain one field row per density")
    if matrix.shape[1] < 2 or real.size < 2 or field_area_arcmin2 <= 0:
        raise ValueError("density sweep needs at least two synthetic and real fields")
    order = np.argsort(x)
    x = x[order]
    matrix = matrix[order]
    means = np.mean(matrix, axis=1) / field_area_arcmin2
    weights = np.full(len(x), matrix.shape[1], dtype=np.float64)
    response = _isotonic(means, weights)
    target = float(np.mean(real) / field_area_arcmin2)
    estimate = _invert_response(x, response, target)
    fingerprint_match = bool(
        active_transfer_fingerprint
        and transfer_fingerprint == active_transfer_fingerprint
    )
    warnings: list[str] = []
    if not fingerprint_match:
        warnings.append("sweep brightness transfer is not the active transfer")
    if estimate is None:
        warnings.append("Euclid target is not bracketed by the sweep response")
    if float(response[-1] - response[0]) < 1.0:
        warnings.append("synthetic detection response is effectively flat")

    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(int(bootstraps)):
        synthetic_index = rng.integers(0, matrix.shape[1], matrix.shape[1])
        real_index = rng.integers(0, real.size, real.size)
        boot_means = np.mean(matrix[:, synthetic_index], axis=1) / field_area_arcmin2
        boot_response = _isotonic(boot_means, weights)
        boot_target = float(np.mean(real[real_index]) / field_area_arcmin2)
        if cone_densities.size >= 2 and float(np.mean(cone_densities)) > 0:
            cone_index = rng.integers(
                0, cone_densities.size, cone_densities.size,
            )
            boot_target *= float(
                np.mean(cone_densities[cone_index])
                / np.mean(cone_densities)
            )
        value = _invert_response(x, boot_response, boot_target)
        if value is not None and math.isfinite(value):
            samples.append(value)
    valid = not warnings and len(samples) >= max(20, bootstraps // 4)
    interval = None
    if samples:
        interval = {
            "median": float(np.median(samples)),
            "p16": float(np.percentile(samples, 16)),
            "p84": float(np.percentile(samples, 84)),
        }
    return {
        "version": 1,
        "method": "paired nested-thinning fields plus weighted isotonic response",
        "valid": valid,
        "warnings": warnings,
        "transfer_fingerprint": transfer_fingerprint,
        "active_transfer_fingerprint": active_transfer_fingerprint,
        "response_points": [
            {
                "density_arcmin2": float(density),
                "detected_density_arcmin2": float(detected),
                "isotonic_density_arcmin2": float(fitted),
            }
            for density, detected, fitted in zip(x, means, response, strict=True)
        ],
        "euclid_detected_density_arcmin2": target,
        "recommended_density_arcmin2": estimate,
        "interval_arcmin2": interval,
        "synthetic_fields_per_point": int(matrix.shape[1]),
        "euclid_fields": int(real.size),
        "euclid_cones": int(cone_densities.size),
    }


def density_state() -> dict[str, Any]:
    candidate = _read(density_calibration_path())
    transfer = photometric_candidate() or {}
    if candidate and candidate.get("transfer_fingerprint") != transfer.get(
        "fingerprint"
    ):
        candidate = dict(candidate)
        candidate["valid"] = False
        candidate["warnings"] = list(candidate.get("warnings") or []) + [
            "brightness-transfer candidate changed after the sweep"
        ]
    active = _read(active_density_path())
    return {
        "candidate": candidate,
        "active": active,
        "is_active": bool(
            candidate and active
            and candidate.get("calibration_fingerprint")
            == active.get("calibration_fingerprint")
        ),
    }


def activate_density_candidate() -> dict[str, Any]:
    """Activate a valid, transfer-matched sweep and update the job config."""
    candidate = _read(density_calibration_path())
    if not candidate or not candidate.get("valid"):
        raise ValueError("No valid matched density calibration is available")
    transfer = active_transfer() or {}
    if candidate.get("transfer_fingerprint") != transfer.get("fingerprint"):
        raise ValueError(
            "Density calibration used a different brightness transfer"
        )
    recommendation = candidate.get("recommended_density_arcmin2")
    if recommendation is None or not math.isfinite(float(recommendation)):
        raise ValueError("Density calibration has no finite recommendation")
    from euclid_polish.web import job_config

    payload = {
        **candidate,
        "active": True,
        "activated_density_arcmin2": float(recommendation),
    }
    job_config.update({"galaxy_density_arcmin2": float(recommendation)})
    _write(active_density_path(), payload)
    return payload


def galaxy_recommendation_state() -> dict[str, Any]:
    """Return every fitted generator parameter as one reviewable proposal."""
    transfer = photometric_candidate()
    density = density_state().get("candidate")
    coefficients = (transfer or {}).get("coefficients") or {}
    quality = (transfer or {}).get("fit_quality") or {}
    warnings = list(quality.get("warnings") or [])
    if density:
        warnings.extend(density.get("warnings") or [])
    recommendation_available = bool(
        transfer
        and density
        and density.get("valid")
        and density.get("recommended_density_arcmin2") is not None
        and density.get("transfer_fingerprint") == transfer.get("fingerprint")
    )
    return {
        "recommendation_available": recommendation_available,
        "validated": bool(recommendation_available and quality.get("valid")),
        "warnings": list(dict.fromkeys(str(item) for item in warnings)),
        "transfer_fingerprint": (transfer or {}).get("fingerprint"),
        "density_calibration_fingerprint": (
            (density or {}).get("calibration_fingerprint")
        ),
        "generator_parameters": {
            "galaxy_density_arcmin2": (
                (density or {}).get("recommended_density_arcmin2")
            ),
            "cosmos_vis_offset_mag": coefficients.get("offset_mag"),
            "cosmos_vis_magnitude_slope": coefficients.get("magnitude_slope"),
            "cosmos_vis_scatter_mag": coefficients.get("scatter_mag"),
        },
        "observation_model_diagnostics": (
            (transfer or {}).get("observation_model") or {}
        ),
        "density_interval_arcmin2": (density or {}).get("interval_arcmin2"),
    }


def activate_galaxy_recommendation() -> dict[str, Any]:
    """Freeze and apply the complete fitted generator parameter proposal."""
    state = galaxy_recommendation_state()
    if not state["recommendation_available"]:
        raise ValueError(
            "Run a transfer-matched density sweep before activating parameters"
        )
    transfer = activate_photometric_transfer(allow_quality_warnings=True)
    density = activate_density_candidate()
    return {
        **state,
        "active": True,
        "brightness_transfer": transfer,
        "density_calibration": density,
    }


def star_state() -> dict[str, Any]:
    candidate = _read(star_candidate_path())
    active = _read(active_star_path())
    return {
        "candidate": candidate,
        "active": active,
        "is_active": bool(
            candidate and active
            and candidate.get("fingerprint") == active.get("fingerprint")
        ),
    }


def active_star() -> dict[str, Any] | None:
    return _read(active_star_path())


def activate_star_candidate() -> dict[str, Any]:
    candidate = _read(star_candidate_path())
    if not candidate or not candidate.get("valid"):
        raise ValueError("No valid fitted stellar population is available")
    payload = {**candidate, "active": True}
    _write(active_star_path(), payload)
    return payload


def write_star_candidate(payload: dict[str, Any]) -> None:
    _write(star_candidate_path(), payload)
