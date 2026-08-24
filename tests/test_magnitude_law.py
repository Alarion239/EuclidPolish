"""Straight log-density magnitude-law fitting and sampling contracts."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.population.euclid_galaxy_prior import (
    BRIGHT_BRIDGE_JOIN_MAGNITUDES,
    fit_continuous_generation_magnitude_law,
)
from euclid_polish.population.magnitude_law import (
    ContinuousBrightBridgeFaintCappedMagnitudeLaw,
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


def _continuous_bridge_law(
    **overrides,
) -> ContinuousBrightBridgeFaintCappedMagnitudeLaw:
    values = {
        "straight_law": _law(intercept=-8.0),
        "bright_slopes": (1.2, 0.25, 0.5),
        "bright_join_magnitudes": BRIGHT_BRIDGE_JOIN_MAGNITUDES,
        "density_cap_arcmin2_mag": 100.0,
    }
    values.update(overrides)
    return ContinuousBrightBridgeFaintCappedMagnitudeLaw(**values)


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


def test_continuous_bright_bridge_joins_each_line_then_main_and_faint_cap():
    law = _continuous_bridge_law()
    straight = law.straight_law

    assert law.mag_bright == 14.0
    assert law.mag_faint == 29.0
    assert law.slope == 0.4
    assert law.intercept == -8.0
    assert law.source == "fixture"
    assert law.bright_intercepts == pytest.approx((-20.92, -5.34, -10.09))
    assert law.break_magnitude == pytest.approx(25.0)
    assert law.density_cap_arcmin2_mag == 100.0

    epsilon = 1.0e-10
    for join in law.bright_join_magnitudes:
        assert law.density(join - epsilon) == pytest.approx(
            law.density(join + epsilon), rel=1.0e-9,
        )
    assert law.density([
        14.0, 16.4, 18.0, 19.0, 20.0, 20.9, 24.0, 25.0, 28.0,
    ]) == pytest.approx([
        10.0 ** (1.2 * 14.0 - 20.92),
        10.0 ** (0.25 * 16.4 - 5.34),
        10.0 ** (0.25 * 18.0 - 5.34),
        10.0 ** (0.5 * 19.0 - 10.09),
        10.0 ** (0.5 * 20.0 - 10.09),
        straight.density(20.9),
        straight.density(24.0),
        100.0,
        100.0,
    ])


def test_continuous_bright_bridge_integrates_and_samples_every_component():
    law = _continuous_bridge_law()
    masses = law._component_masses()
    assert len(masses) == 5
    assert masses[-1] == pytest.approx(400.0)
    expected_line_masses = [
        law._line_integral(slope, intercept, bright, faint)
        for bright, faint, slope, intercept in law._line_components()
    ]
    assert masses[:-1] == pytest.approx(expected_line_masses)
    assert law.integrated_density() == pytest.approx(sum(masses))
    total = sum(masses)

    class FixedRng:
        def __init__(self, target: float):
            self.value = target / total

        def random(self) -> float:
            return self.value

    before = 0.0
    for mass, (bright, faint, _slope, _intercept) in zip(
        masses[:-1], law._line_components(), strict=True,
    ):
        draw = law.sample(FixedRng(before + mass / 2.0))
        assert bright < draw < faint
        before += mass
    faint_draw = law.sample(FixedRng(before + masses[-1] / 2.0))
    assert faint_draw == pytest.approx(27.0)

    draws = np.asarray([
        law.sample(np.random.default_rng(seed)) for seed in range(6000)
    ])
    assert np.all((draws >= law.mag_bright) & (draws < law.mag_faint))
    assert np.mean(draws >= law.break_magnitude) == pytest.approx(
        masses[-1] / total, abs=0.015,
    )


def test_continuous_bright_bridge_round_trips_and_rejects_bad_payloads():
    law = _continuous_bridge_law()
    payload = law.to_payload()
    assert ContinuousBrightBridgeFaintCappedMagnitudeLaw.from_payload(
        payload
    ) == law
    assert "empirical_edges" not in payload
    assert "empirical_density_arcmin2_mag" not in payload

    for field in (
        "break_magnitude", "surface_density_arcmin2",
    ):
        payload = law.to_payload()
        payload[field] += 1.0
        with pytest.raises(ValueError, match="inconsistent"):
            ContinuousBrightBridgeFaintCappedMagnitudeLaw.from_payload(payload)

    payload = law.to_payload()
    payload["bright_intercepts"][0] += 1.0
    with pytest.raises(ValueError, match="inconsistent"):
        ContinuousBrightBridgeFaintCappedMagnitudeLaw.from_payload(payload)

    payload = law.to_payload()
    payload["source"] = "different source"
    with pytest.raises(ValueError, match="inconsistent"):
        ContinuousBrightBridgeFaintCappedMagnitudeLaw.from_payload(payload)

    payload = law.to_payload()
    payload.pop("bright_slopes")
    with pytest.raises(ValueError, match="malformed"):
        ContinuousBrightBridgeFaintCappedMagnitudeLaw.from_payload(payload)

    with pytest.raises(ValueError, match="not a continuous bright-bridge"):
        ContinuousBrightBridgeFaintCappedMagnitudeLaw.from_payload({})
    with pytest.raises(ValueError, match="slopes are invalid"):
        _continuous_bridge_law(bright_slopes=(1.0, 0.0, 0.5))
    with pytest.raises(ValueError, match="slopes are invalid"):
        _continuous_bridge_law(bright_slopes=(1.0, 0.5))
    with pytest.raises(ValueError, match="joins are invalid"):
        _continuous_bridge_law(bright_join_magnitudes=(16.4, 16.0, 20.9))
    with pytest.raises(ValueError, match="joins must lie before"):
        _continuous_bridge_law(bright_join_magnitudes=(14.0, 19.0, 20.9))
    with pytest.raises(ValueError, match="joins must lie before"):
        _continuous_bridge_law(bright_join_magnitudes=(16.4, 19.0, 25.0))
    with pytest.raises(ValueError, match="inside the domain"):
        _continuous_bridge_law(density_cap_arcmin2_mag=1.0e8)


def test_continuous_bright_bridge_fit_recovers_three_slopes_and_diagnostics():
    area = 100.0
    expected_slopes = (1.62, 0.275, 0.503)
    source_law = _continuous_bridge_law(bright_slopes=expected_slopes)
    edges = np.arange(14.0, 21.01, 0.1)
    bins = []
    for index, (bright, faint) in enumerate(zip(
        edges[:-1], edges[1:], strict=True,
    )):
        mass = 0.0
        for start, stop, slope, intercept in source_law._line_components():
            overlap_bright = max(float(bright), start)
            overlap_faint = min(float(faint), stop)
            if overlap_faint > overlap_bright:
                mass += source_law._line_integral(
                    slope, intercept, overlap_bright, overlap_faint,
                )
        expected_galaxies = area * mass
        if index < 3:
            expected_galaxies = 0.0
        bins.append({
            "mag_lo": float(bright),
            "mag_hi": float(faint),
            "expected_galaxies": expected_galaxies,
        })

    fitted, diagnostics = fit_continuous_generation_magnitude_law(
        source_law.straight_law,
        bins,
        footprint_area_arcmin2=area,
    )

    assert fitted.bright_join_magnitudes == BRIGHT_BRIDGE_JOIN_MAGNITUDES
    assert fitted.bright_slopes == pytest.approx(expected_slopes, abs=0.01)
    assert diagnostics["bright_fit_bin_count"] == 69
    assert diagnostics["bright_fit_zero_bin_count"] == 3
    assert diagnostics["bright_fit_parameter_count"] == 3
    assert diagnostics["bright_bridge_join_magnitudes"] == list(
        BRIGHT_BRIDGE_JOIN_MAGNITUDES
    )
    assert diagnostics["bright_bridge_slopes"] == pytest.approx(
        fitted.bright_slopes
    )
    assert float(diagnostics["bright_fit_poisson_deviance"]) >= 0.0


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
