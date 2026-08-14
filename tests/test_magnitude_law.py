"""Straight log-density magnitude-law fitting and sampling contracts."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.population.magnitude_law import (
    EmpiricalBrightFaintCappedMagnitudeLaw,
    FaintCappedMagnitudeLaw,
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


def _empirical_law(**overrides) -> EmpiricalBrightFaintCappedMagnitudeLaw:
    values = {
        "straight_law": _law(intercept=-8.0),
        "empirical_edges": (14.0, 16.0, 19.0),
        "empirical_density_arcmin2_mag": (0.25, 1.5),
        "density_cap_arcmin2_mag": 100.0,
    }
    values.update(overrides)
    return EmpiricalBrightFaintCappedMagnitudeLaw(**values)


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


def test_faint_capped_law_breaks_then_stays_flat_and_samples_full_domain():
    fitted = _law(intercept=-8.0)
    generated = FaintCappedMagnitudeLaw(fitted, 100.0)

    assert generated.break_magnitude == pytest.approx(25.0)
    assert generated.density([24.0, 25.0, 27.0]) == pytest.approx([
        fitted.density(24.0), 100.0, 100.0,
    ])
    # Integral to the knee plus the rectangular faint tail.
    beta = fitted.slope * np.log(10.0)
    straight = (
        10.0 ** fitted.intercept
        * (np.exp(beta * 25.0) - np.exp(beta * 14.0))
        / beta
    )
    assert generated.integrated_density() == pytest.approx(straight + 400.0)

    draws = np.asarray([
        generated.sample(np.random.default_rng(seed)) for seed in range(6000)
    ])
    assert np.all((draws >= 14.0) & (draws < 29.0))
    expected_tail_fraction = 400.0 / generated.integrated_density()
    assert np.mean(draws >= 25.0) == pytest.approx(
        expected_tail_fraction, abs=0.015,
    )

    restored = FaintCappedMagnitudeLaw.from_payload(generated.to_payload())
    assert restored == generated


def test_empirical_bright_law_uses_bins_then_straight_then_flat():
    law = _empirical_law()
    straight = law.straight_law

    assert law.mag_bright == 14.0
    assert law.mag_faint == 29.0
    assert law.fit_bright == 19.0
    assert law.fit_faint == 25.0
    assert law.empirical_faint == 19.0
    assert law.break_magnitude == pytest.approx(25.0)
    assert law.density_cap_arcmin2_mag == 100.0
    assert law.source == "fixture"
    assert law.density([
        14.0, 15.999, 16.0, 18.999, 19.0, 24.0, 25.0, 28.0,
    ]) == pytest.approx([
        0.25, 0.25, 1.5, 1.5,
        straight.density(19.0), straight.density(24.0), 100.0, 100.0,
    ])
    assert law.density(17.0) == pytest.approx(1.5)

    beta = straight.slope * np.log(10.0)
    straight_middle = (
        10.0 ** straight.intercept
        * (np.exp(beta * 25.0) - np.exp(beta * 19.0))
        / beta
    )
    empirical = 2.0 * 0.25 + 3.0 * 1.5
    assert law.integrated_density() == pytest.approx(
        empirical + straight_middle + 400.0
    )


def test_empirical_bright_law_inverse_cdf_samples_all_three_components():
    law = _empirical_law()
    empirical, straight, faint = law._component_masses()
    total = float(np.sum(empirical) + straight + faint)

    class FixedRng:
        def __init__(self, target: float):
            self.value = target / total

        def random(self) -> float:
            return self.value

    # Halfway through the first empirical bin.
    assert law.sample(FixedRng(0.25)) == pytest.approx(15.0)
    # Halfway through the analytical straight component.
    middle_draw = law.sample(FixedRng(float(np.sum(empirical)) + straight / 2.0))
    assert law.empirical_faint < middle_draw < law.break_magnitude
    # Halfway through the constant faint component.
    faint_draw = law.sample(FixedRng(
        float(np.sum(empirical)) + straight + faint / 2.0
    ))
    assert faint_draw == pytest.approx(27.0)

    draws = np.asarray([
        law.sample(np.random.default_rng(seed)) for seed in range(6000)
    ])
    assert np.all((draws >= law.mag_bright) & (draws < law.mag_faint))
    assert np.mean(draws >= law.break_magnitude) == pytest.approx(
        faint / total, abs=0.015,
    )


def test_empirical_bright_law_round_trips_and_rejects_bad_payloads():
    law = _empirical_law()
    assert EmpiricalBrightFaintCappedMagnitudeLaw.from_payload(
        law.to_payload()
    ) == law

    payload = law.to_payload()
    payload["surface_density_arcmin2"] *= 2.0
    with pytest.raises(ValueError, match="inconsistent"):
        EmpiricalBrightFaintCappedMagnitudeLaw.from_payload(payload)

    payload = law.to_payload()
    payload["source"] = "different source"
    with pytest.raises(ValueError, match="inconsistent"):
        EmpiricalBrightFaintCappedMagnitudeLaw.from_payload(payload)

    with pytest.raises(ValueError, match="bins are invalid"):
        _empirical_law(empirical_edges=(14.0, 16.0, 15.0))
    with pytest.raises(ValueError, match="bins are invalid"):
        _empirical_law(empirical_density_arcmin2_mag=(0.25, -1.0))
    with pytest.raises(ValueError, match="bright limit"):
        _empirical_law(empirical_edges=(14.1, 16.0, 19.0))
    with pytest.raises(ValueError, match="inside the output domain"):
        _empirical_law(empirical_edges=(14.0, 20.0, 25.0))


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
