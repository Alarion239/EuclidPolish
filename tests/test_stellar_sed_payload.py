"""The Euclid-only stellar colour model: payload, deconvolution, bin pooling.

Stars adopt the colours of resampled Q1 point sources in their 0.5-mag bin,
with noise removed by joint three-colour extreme deconvolution; bins with too
little star weight use their nearest populated bin; the payload fails closed.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from euclid_polish.population.magnitude_law import StraightMagnitudeLaw
from euclid_polish.sky.generation.stellar_sed import (
    EmpiricalStellarPrior,
    StarColorModel,
    bin_moments,
    build_star_color_model,
    color_noise_covariance,
    deconvolved_draw,
)
from tests._star_fixtures import fixture_star_color_model


def _law() -> dict:
    return StraightMagnitudeLaw(
        slope=0.15, intercept=-3.5, mag_bright=12.0, mag_faint=25.0,
        fit_bright=18.0, fit_faint=21.0,
        covariance=((1.0e-4, 0.0), (0.0, 1.0e-3)), r_squared=1.0,
        rms_log10_density=0.0, source="fixture",
    ).to_payload()


def _colors(prior: EmpiricalStellarPrior, magnitude: float, n: int = 2000) -> np.ndarray:
    rng = np.random.default_rng(5)
    seds = [prior.sample(rng, magnitude) for _ in range(n)]
    return np.asarray([
        [s.magnitudes["VIS"] - s.magnitudes["Y_E"],
         s.magnitudes["Y_E"] - s.magnitudes["J_E"],
         s.magnitudes["J_E"] - s.magnitudes["H_E"]] for s in seds
    ])


def test_sampled_stars_follow_the_q1_locus_at_the_requested_vis():
    prior = EmpiricalStellarPrior.from_payload({
        "population": {"magnitude_distribution": _law()},
        "color_model": fixture_star_color_model(),
    })
    colors = _colors(prior, 20.3)
    # The fixture's locus: Y−J = 0.05 + 0.1 (VIS−Y), J−H = 0.02 (VIS−Y).
    np.testing.assert_allclose(
        colors[:, 1], 0.05 + 0.1 * colors[:, 0], atol=0.12,
    )
    assert 0.3 < np.median(colors[:, 0]) < 1.5
    seds = prior.sample(np.random.default_rng(0), 20.3).magnitudes
    assert seds["VIS"] == pytest.approx(20.3)


def test_deconvolution_keeps_precise_rows_and_shrinks_noisy_ones_along_the_locus():
    rng = np.random.default_rng(1)
    vis_y = rng.uniform(0.3, 1.5, 4000)
    true = np.column_stack((vis_y, 0.05 + 0.1 * vis_y, 0.02 * vis_y))
    sigma = np.full((4000, 4), 0.01)
    noise = color_noise_covariance(sigma)
    location, intrinsic = bin_moments(true, noise, np.ones(4000))
    # Intrinsic scatter is mostly along VIS−Y; Y−J follows it.
    assert intrinsic[0, 0] > 20 * intrinsic[1, 1]
    precise = deconvolved_draw(
        location, intrinsic, true[0], color_noise_covariance(np.full(4, 1e-4)), rng,
    )
    np.testing.assert_allclose(precise, true[0], atol=5e-3)
    noisy_row = location + np.asarray([0.0, 0.5, 0.0])     # far off-locus in Y−J
    pulled = [
        deconvolved_draw(location, intrinsic, noisy_row,
                         color_noise_covariance(np.full(4, 0.3)), rng)[1]
        for _ in range(500)
    ]
    assert abs(np.mean(pulled) - location[1]) < 0.1


def test_noise_covariance_anticorrelates_adjacent_colours():
    cov = color_noise_covariance(np.asarray([0.1, 0.2, 0.3, 0.4]))
    np.testing.assert_allclose(cov, [
        [0.05, -0.04, 0.0], [-0.04, 0.13, -0.09], [0.0, -0.09, 0.25],
    ])


def test_empty_bins_borrow_the_nearest_populated_bin():
    payload = fixture_star_color_model()
    model = StarColorModel.from_payload(payload)
    edges = model.magnitude_edges
    populated = [i for i, w in enumerate(payload["bin_weight"]) if w >= payload["min_bin_weight"]]
    assert model.serving_bin(13.0) == min(populated)
    assert model.serving_bin(24.9) == max(populated)
    inside = 0.5 * (edges[populated[0]] + edges[populated[0] + 1])
    assert model.serving_bin(float(inside)) == populated[0]


def test_payload_fails_closed_when_tampered_or_gaia_shaped():
    payload = fixture_star_color_model()
    tampered = copy.deepcopy(payload)
    tampered["rows"]["color_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        StarColorModel.from_payload(tampered)
    with pytest.raises(ValueError, match="Euclid point-source colour model"):
        EmpiricalStellarPrior.from_payload({
            "population": {"magnitude_distribution": _law()},
            "color_model": {"kind": "gaia_euclid_latent_locus_v1"},
        })


def test_builder_rejects_inconsistent_rows():
    with pytest.raises(ValueError, match="inconsistent shapes"):
        build_star_color_model(
            np.ones(3), np.ones((3, 3)), np.ones((3, 3)), np.ones(3),
            mag_bright=12.0, mag_faint=25.0,
        )
    with pytest.raises(ValueError, match="minimum star weight"):
        build_star_color_model(
            np.full(3, 20.0), np.zeros((3, 3)), np.full((3, 4), 0.01), np.ones(3),
            mag_bright=12.0, mag_faint=25.0,
        )
