"""Focused tests for data-derived TNG density calibration."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.web.helpers.tng_prior import (
    catalog_prior_estimate,
    visible_prior_estimate,
)


def _rows(kind: str, magnitudes: list[float]):
    return [{"type": kind, "mag_vis": magnitude} for magnitude in magnitudes]


def test_catalog_prior_recovers_area_normalized_count_scale():
    synthetic = _rows(
        "galaxy",
        [20.1] * 40 + [20.6] * 60 + [21.1] * 80 + [21.6] * 100,
    )
    euclid = _rows(
        "unknown",
        [20.1] * 20 + [20.6] * 30 + [21.1] * 40 + [21.6] * 50,
    )

    result = catalog_prior_estimate(
        synthetic,
        euclid,
        synthetic_area_arcmin2=10.0,
        euclid_area_arcmin2=2.5,
        current_prior=60.0,
    )

    assert result is not None
    assert result["fitted_prior_arcmin2"] == pytest.approx(120.0)
    assert result["single_scalar_adequate"]
    assert result["reduced_poisson_deviance"] == pytest.approx(0.0)


def test_catalog_prior_rejects_magnitude_dependent_scale():
    synthetic = _rows(
        "galaxy",
        [20.1] * 100 + [20.6] * 100 + [21.1] * 100 + [21.6] * 100,
    )
    euclid = _rows(
        "unknown",
        [20.1] * 10 + [20.6] * 20 + [21.1] * 40 + [21.6] * 80,
    )

    result = catalog_prior_estimate(
        synthetic,
        euclid,
        synthetic_area_arcmin2=10.0,
        euclid_area_arcmin2=2.5,
        current_prior=60.0,
    )

    assert result is not None
    assert not result["single_scalar_adequate"]
    assert result["log10_prior_slope_per_mag"] > 0.3


def test_visible_prior_uses_common_net_detection_density():
    source_detection = {
        "settings": {"threshold_sigma": 4.0},
        "synthetic": {
            "positive": [5, 5, 5],
            "negative": [1, 1, 1],
            "matched_stars": [1, 1, 1],
            "matched_galaxies": [2, 2, 2],
            "truth_galaxies": [6, 6, 6],
        },
        "real": {
            "positive": [8, 8, 8],
            "negative": [1, 1, 1],
        },
    }
    euclid = [{"type": "star"}]

    result = visible_prior_estimate(
        source_detection,
        euclid,
        euclid_area_arcmin2=10.0,
        field_area_arcmin2=1.0,
        current_prior=60.0,
    )

    assert result is not None
    assert result["synthetic_detected_density_arcmin2"] == pytest.approx(3.0)
    assert result["real_detected_density_arcmin2"] == pytest.approx(6.9)
    assert result["fitted_prior_arcmin2"] == pytest.approx(138.0)
    assert result["matched_truth_fraction"] == pytest.approx(1 / 3)
    assert np.isfinite(result["interval_arcmin2"]["median"])
