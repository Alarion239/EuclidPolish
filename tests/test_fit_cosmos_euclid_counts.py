from __future__ import annotations

import csv
import math
from dataclasses import asdict, replace

import numpy as np
import pytest

from euclid_polish.population import joint_galaxy
from euclid_polish.sky.generation import cosmos_tng_prior


def test_euclid_catalog_uses_fractional_membership_and_size_proxy(tmp_path):
    path = tmp_path / "euclid.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "mag_vis", "spurious_prob", "point_like_prob",
            "semimajor_axis", "ellipticity", "flux_vis_aper_uJy",
            "fluxerr_vis_aper_uJy",
        ))
        writer.writeheader()
        writer.writerows([
            {"mag_vis": 20.2, "spurious_prob": 0.0,
             "point_like_prob": 0.25, "semimajor_axis": 3.0,
             "ellipticity": 0.0, "flux_vis_aper_uJy": 10.0,
             "fluxerr_vis_aper_uJy": 1.0},
            {"mag_vis": 20.3, "spurious_prob": 0.0,
             "point_like_prob": 0.80, "semimajor_axis": 4.0,
             "ellipticity": 0.75, "flux_vis_aper_uJy": 5.0,
             "fluxerr_vis_aper_uJy": 1.0},
            {"mag_vis": 20.4, "spurious_prob": 0.9,
             "point_like_prob": 0.0, "semimajor_axis": 3.0,
             "ellipticity": 0.0, "flux_vis_aper_uJy": 10.0,
             "fluxerr_vis_aper_uJy": 1.0},
            {"mag_vis": 21.2, "spurious_prob": 0.0,
             "point_like_prob": "", "semimajor_axis": 3.0,
             "ellipticity": 0.0, "flux_vis_aper_uJy": 10.0,
             "fluxerr_vis_aper_uJy": 1.0},
        ])

    catalog = joint_galaxy.read_euclid_population(path)

    np.testing.assert_allclose(catalog["weight"], [0.75, 0.20])
    np.testing.assert_allclose(catalog["radius_arcsec"], [0.30, 0.20])
    np.testing.assert_allclose(
        catalog["magnitude_error"],
        (2.5 / math.log(10.0)) * np.asarray([0.1, 0.2]),
    )
    np.testing.assert_allclose(catalog["flux_error_uJy"], [1.0, 1.0])
    assert catalog["missing_probability_rows"] == 1
    assert catalog["classification_weighting"] == (
        "galaxy_weight=1-POINT_LIKE_PROB"
    )
    assert "not a fitted half-light radius" in catalog["size_estimator"]


def test_size_relation_recovers_luminosity_and_redshift_trends():
    rng = np.random.default_rng(17)
    redshift = rng.uniform(0.10, 2.5, 5000)
    magnitude = rng.uniform(19.0, 26.5, 5000)
    absolute_like = magnitude - joint_galaxy.Planck15.distmod(redshift).value
    magnitude_coordinate = absolute_like + 20.0
    redshift_coordinate = np.log10(1.0 + redshift)
    expected_log_kpc = (
        0.62 - 0.14 * magnitude_coordinate
        - 0.90 * redshift_coordinate
        + 0.006 * magnitude_coordinate**2
        + 0.08 * magnitude_coordinate * redshift_coordinate
    )
    scatter = 0.18 * np.exp(-0.015 * magnitude_coordinate)
    log_kpc = expected_log_kpc + rng.normal(0.0, scatter, len(redshift))
    kpc_per_arcsec = (
        joint_galaxy.Planck15.kpc_proper_per_arcmin(redshift).value / 60.0
    )
    radius_arcsec = np.power(10.0, log_kpc) / kpc_per_arcsec

    fitted = joint_galaxy.fit_size_evolution(
        magnitude, redshift, radius_arcsec,
    )

    assert abs(fitted.log10_r0_kpc - 0.62) < 0.02
    assert abs(fitted.magnitude_slope + 0.14) < 0.01
    assert abs(fitted.log1pz_slope + 0.90) < 0.08
    assert abs(fitted.magnitude_curvature - 0.006) < 0.003
    assert abs(fitted.magnitude_redshift_interaction - 0.08) < 0.04
    assert abs(fitted.scatter_dex - 0.18) < 0.01
    assert abs(fitted.scatter_magnitude_slope + 0.015) < 0.01


def test_euclid_response_is_surface_brightness_dependent():
    latent_density = np.ones((1, 2), dtype=np.float64)
    latent_magnitude = np.asarray([25.0])
    latent_log_radius = np.log10(np.asarray([0.10, 0.80]))
    parameters = np.asarray([
        0.0, 0.0, 0.0, math.log(0.10), 0.0, math.log(0.02),
        25.2, math.log(0.40), math.log(0.50),
    ])

    _prediction, completeness = joint_galaxy.predict_euclid_histogram(
        latent_density, latent_magnitude, latent_log_radius, parameters,
        measurement_flux_error_uJy=0.07,
    )

    # At the same total magnitude, the larger source has lower mean surface
    # brightness and therefore lower fitted detection probability.
    assert np.all(completeness[0] > completeness[1])


def test_signed_poisson_residual_squares_to_cash_deviance():
    observed = np.asarray([0.0, 2.0, 8.0])
    predicted = np.asarray([1.0, 3.0, 5.0])
    residual = joint_galaxy.signed_poisson_residual(observed, predicted)
    logarithmic = np.zeros_like(observed)
    positive = observed > 0.0
    logarithmic[positive] = observed[positive] * np.log(
        observed[positive] / predicted[positive]
    )
    expected = 2.0 * (predicted - observed + logarithmic)

    np.testing.assert_allclose(residual * residual, expected)


def test_extra_poisson_fractional_scatter_is_recovered():
    rng = np.random.default_rng(91)
    mean = np.linspace(2.0, 120.0, 4000)
    true_tau = 0.22
    shape = 1.0 / true_tau**2
    observed = rng.negative_binomial(shape, shape / (shape + mean))

    fitted = joint_galaxy._fit_fractional_overdispersion(observed, mean)

    assert fitted == pytest.approx(true_tau, abs=0.02)


def test_censored_fit_ignores_exact_subresolution_radius():
    radius_edges = np.unique(np.append(
        joint_galaxy.EUCLID_LOG_RE_EDGES, math.log10(0.10),
    ))
    latent = {
        "density": np.ones((1, 2, 2), dtype=np.float64),
        "magnitude": np.asarray([23.0, 25.0]),
        "log_radius": np.log10(np.asarray([0.15, 0.30])),
    }
    shared = {
        "magnitude": np.asarray([23.1, 25.1]),
        "weight": np.asarray([1.0, 0.5]),
        "cone_index": np.asarray([0, 0]),
    }

    fitted: list[joint_galaxy.EuclidResponseFit] = []
    for radii in ([0.080, 0.098], [0.098, 0.080]):
        result, _observed, _predicted = joint_galaxy.fit_euclid_response(
            latent,
            {**shared, "radius_arcsec": np.asarray(radii)},
            area_arcmin2=10.0,
            unresolved_policy="censor",
            unresolved_radius_arcsec=0.10,
            log_radius_edges=radius_edges,
            measurement_flux_error_uJy=0.07,
        )
        fitted.append(result)

    assert fitted[0] == fitted[1]


def test_tng_draw_target_excludes_euclid_measurement_error():
    latent = {
        "density": np.ones((1, 2, 2), dtype=np.float64),
        "z": np.asarray([0.5]),
        "z_edges": np.asarray([0.4, 0.6]),
        "magnitude": np.asarray([23.0, 25.0]),
        "log_radius": np.log10(np.asarray([0.15, 0.30])),
        "log_radius_edges": np.log10(np.asarray([0.10, 0.20, 0.40])),
    }
    fitted_response = joint_galaxy.EuclidResponseFit(
        population_scale=0.7,
        vis_minus_f814w_mag=-0.2,
        magnitude_slope=0.8,
        scatter_mag=0.12,
        measurement_flux_error_uJy=0.02,
        size_scale=1.1,
        size_floor_arcsec=0.1,
        completeness_m50=25.0,
        completeness_width_mag=0.4,
        surface_brightness_penalty=0.5,
        bright_transfer_magnitude_max=24.0,
        bright_poisson_deviance=1.0,
        bright_dof=1,
        poisson_deviance=1.0,
        dof=1,
        standard_errors=(0.0,) * 9,
    )
    altered_measurement = replace(
        fitted_response, measurement_flux_error_uJy=2.0,
    )

    first = joint_galaxy.tng_draw_population_cube(latent, fitted_response)
    second = joint_galaxy.tng_draw_population_cube(
        latent, altered_measurement,
    )

    np.testing.assert_allclose(first["density"], second["density"])


def test_activated_joint_population_samples_z_vis_and_radius(monkeypatch):
    luminosity = joint_galaxy.SchechterEvolutionFit(
        log_phi_star=-3.0, m_star_0=-21.0, alpha=-1.2,
        m_star_log1pz_slope=0.0, log_phi_log1pz_slope=0.0,
        alpha_log1pz_slope=0.0, m_star_log1pz_quadratic=0.0,
        log_phi_log1pz_quadratic=0.0,
        cosmic_variance_fractional_scatter=0.1,
        poisson_deviance=1.0, negative_binomial_deviance=1.0,
        dof=1, standard_errors=(0.0,) * 8,
    )
    size = joint_galaxy.SizeEvolutionFit(
        log10_r0_kpc=0.5, magnitude_slope=-0.1, log1pz_slope=-0.5,
        magnitude_curvature=0.0, magnitude_redshift_interaction=0.0,
        scatter_dex=0.2, scatter_magnitude_slope=0.0,
        residual_rms_dex=0.2, fitted_rows=100, clipped_rows=0,
        standard_errors=(0.0,) * 7,
    )
    response = joint_galaxy.EuclidResponseFit(
        population_scale=1.0, vis_minus_f814w_mag=0.0,
        magnitude_slope=1.0, scatter_mag=0.2,
        measurement_flux_error_uJy=0.07, size_scale=1.0,
        size_floor_arcsec=0.1, completeness_m50=25.0,
        completeness_width_mag=0.4, surface_brightness_penalty=0.2,
        bright_transfer_magnitude_max=24.0, bright_poisson_deviance=1.0,
        bright_dof=1, poisson_deviance=1.0, dof=1,
        standard_errors=(0.0,) * 9,
    )
    density = np.arange(1.0, 9.0).reshape(2, 2, 2)
    monkeypatch.setattr(
        cosmos_tng_prior, "latent_population_cube", lambda *_args: {},
    )
    monkeypatch.setattr(
        cosmos_tng_prior, "tng_draw_population_cube", lambda *_args: {
            "density": density,
            "z_edges": np.asarray([0.1, 0.5, 1.0]),
            "vis_magnitude_edges": np.asarray([20.0, 24.0, 28.0]),
            "log_radius_edges": np.log10(np.asarray([0.05, 0.2, 0.8])),
        },
    )
    prior = cosmos_tng_prior.JointGalaxyPopulationPrior({
        "version": 1, "kind": "joint_analytical_tng_draw",
        "valid": True, "active": True, "fingerprint": "c" * 64,
        "model": {
            "luminosity_function": asdict(luminosity),
            "size_relation": asdict(size),
            "euclid_response": asdict(response),
        },
        "generation": {"surface_density_arcmin2": float(np.sum(density))},
    })

    draws = [prior.sample(np.random.default_rng(seed)) for seed in range(200)]

    assert prior.surface_density_arcmin2 == pytest.approx(36.0)
    assert all(0.1 <= draw.z < 1.0 for draw in draws)
    assert all(20.0 <= draw.target_vis_mag < 28.0 for draw in draws)
    assert all(0.05 <= draw.re_arcsec < 0.8 for draw in draws)
    assert all(draw.activity_class == "unconditioned" for draw in draws)
