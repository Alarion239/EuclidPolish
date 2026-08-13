"""Straight log-density magnitude-law fitting and sampling contracts."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.population.magnitude_law import (
    StraightMagnitudeLaw,
    fit_shared_slope,
    fit_straight_region,
)


def _law(**overrides) -> StraightMagnitudeLaw:
    values = {
        "slope": 0.4,
        "intercept": -8.0,
        "mag_bright": 14.0,
        "mag_faint": 29.0,
        "fit_bright": 19.0,
        "fit_faint": 25.0,
        "covariance": ((1.0e-4, 0.0), (0.0, 1.0e-3)),
        "r_squared": 0.999,
        "rms_log10_density": 0.01,
        "source": "fixture",
    }
    values.update(overrides)
    return StraightMagnitudeLaw(**values)


def test_integral_and_inverse_cdf_respect_full_domain_boundaries():
    law = _law()
    beta = law.slope * np.log(10.0)
    expected = (
        10.0 ** law.intercept
        * (np.exp(beta * law.mag_faint) - np.exp(beta * law.mag_bright))
        / beta
    )
    assert law.integrated_density() == pytest.approx(expected)
    draws = np.asarray([law.sample(np.random.default_rng(seed)) for seed in range(5000)])
    assert np.all((draws >= 14.0) & (draws < 29.0))
    expected_cdf_25 = (
        np.exp(beta * (25.0 - 14.0)) - 1.0
    ) / (np.exp(beta * (29.0 - 14.0)) - 1.0)
    assert np.mean(draws < 25.0) == pytest.approx(expected_cdf_25, abs=0.015)


def test_density_cap_preserves_bright_law_and_moves_only_faint_limit():
    fitted = _law()
    generated = fitted.truncated_to_density(100.0)

    assert generated.slope == fitted.slope
    assert generated.intercept == fitted.intercept
    assert generated.mag_bright == fitted.mag_bright
    assert generated.mag_faint < fitted.mag_faint
    assert generated.integrated_density() == pytest.approx(100.0)
    probe = np.linspace(fitted.mag_bright, generated.mag_faint, 20)
    assert generated.density(probe) == pytest.approx(fitted.density(probe))


def test_density_cap_leaves_already_sparse_law_unchanged():
    fitted = _law(intercept=-20.0)
    assert fitted.truncated_to_density(100.0) is fitted


def test_straight_region_selects_widest_passing_consecutive_window():
    magnitude = np.arange(14.05, 24.05, 0.1)
    density = 10.0 ** (0.35 * magnitude - 7.0)
    density[:5] *= np.asarray([100.0, 0.01, 80.0, 0.02, 50.0])
    density[-5:] *= np.asarray([0.02, 50.0, 0.01, 80.0, 0.02])
    sigma = 0.03 * density
    fit = fit_straight_region(
        magnitude, density, sigma,
        minimum_span_mag=4.0, minimum_r_squared=0.998,
    )
    assert fit.start == 5
    assert fit.stop == magnitude.size - 5
    assert fit.slope == pytest.approx(0.35)


def test_shared_slope_keeps_separate_survey_normalisations():
    x_gaia = np.linspace(13.0, 20.0, 50)
    x_q1 = np.linspace(18.0, 23.0, 40)
    slope = 0.18
    gaia = 10.0 ** (slope * x_gaia - 3.0)
    q1 = 10.0 ** (slope * x_q1 - 3.7)
    fitted_slope, intercepts, covariance, r_squared, rms = fit_shared_slope([
        (x_gaia, gaia, 0.05 * gaia),
        (x_q1, q1, 0.05 * q1),
    ])
    assert fitted_slope == pytest.approx(slope)
    assert intercepts == pytest.approx([-3.0, -3.7])
    assert covariance.shape == (3, 3)
    assert r_squared == pytest.approx(1.0)
    assert rms < 1.0e-10


def test_old_empirical_or_inconsistent_payloads_fail_closed():
    with pytest.raises(ValueError, match="not a straight"):
        StraightMagnitudeLaw.from_payload({"edges": [12, 25], "cdf": [0, 1]})
    payload = _law().to_payload()
    payload["surface_density_arcmin2"] *= 2.0
    with pytest.raises(ValueError, match="inconsistent"):
        StraightMagnitudeLaw.from_payload(payload)
