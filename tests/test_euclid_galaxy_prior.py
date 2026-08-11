from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.photometry import ab_mag_to_uJy
from euclid_polish.population.euclid_galaxy_prior import (
    ConditionalRadiusLaw,
    fit_conditional_radius_law,
    joint_density_grid,
)
from euclid_polish.population.magnitude_law import StraightMagnitudeLaw
from euclid_polish.sky.generation.cosmos_tng_prior import (
    JointGalaxyPopulationPrior,
)
from euclid_polish.web.helpers.population_calibration import (
    activate_joint_galaxy_candidate,
    active_joint_galaxy_path,
    fit_euclid_joint_galaxy_candidate,
    joint_galaxy_candidate_path,
)


def magnitude_law() -> StraightMagnitudeLaw:
    return StraightMagnitudeLaw(
        slope=0.2, intercept=-3.0,
        mag_bright=14.0, mag_faint=29.0,
        fit_bright=18.0, fit_faint=27.0,
        covariance=((1e-4, 0.0), (0.0, 1e-3)),
        r_squared=0.999, rms_log10_density=0.01,
        source="fixture",
    )


def radius_law() -> ConditionalRadiusLaw:
    return ConditionalRadiusLaw(
        version=1, pivot_mag=23.0, intercept_log10_arcsec=-0.4,
        slope_log10_arcsec_per_mag=-0.08, scatter_dex=0.18,
        log_radius_min=np.log10(0.03), log_radius_max=np.log10(10.0),
        fitted_rows=1000, clipped_rows=4, weighted_rows=800.0,
        residual_rms_dex=0.18, r_squared=0.3,
        covariance=((1e-4, 0.0), (0.0, 1e-5)), selection="fixture",
    )


def active_payload() -> dict:
    mag = magnitude_law()
    return {
        "version": 4, "kind": "euclid_vis2fwhm_sersic_re_joint",
        "valid": True, "active": True, "fingerprint": "b" * 64,
        "magnitude_law": mag.to_payload(),
        "radius_law": radius_law().to_payload(),
        "magnitude_plot": {
            "law": {"x": [14.0, 29.0], "density": [0.1, 100.0]},
        },
        "plots": {
            "radius": {
                "x": [-1.0, 0.0], "density": [1.0, 1.0],
                "observed_density": [1.0, 1.0],
            },
            "conditional_radius": {
                "magnitude": [14.0, 29.0],
                "model_mean_log10_arcsec": [0.32, -0.88],
            },
        },
        "generation": {"surface_density_arcmin2": mag.integrated_density()},
    }


def test_radius_fit_recovers_straight_conditional_relation():
    rng = np.random.default_rng(7)
    magnitude = rng.uniform(18.0, 27.0, 4000)
    expected_intercept, expected_slope = -0.35, -0.075
    log_radius = (
        expected_intercept + expected_slope * (magnitude - 23.0)
        + rng.normal(0.0, 0.16, magnitude.size)
    )
    law = fit_conditional_radius_law(
        magnitude, 10.0**log_radius, np.ones(magnitude.size),
    )

    assert law.intercept_log10_arcsec == pytest.approx(expected_intercept, abs=0.01)
    assert law.slope_log10_arcsec_per_mag == pytest.approx(expected_slope, abs=0.005)
    assert law.scatter_dex == pytest.approx(0.16, abs=0.01)


def test_joint_grid_integrates_to_straight_brightness_density():
    mag = magnitude_law()
    grid = joint_density_grid(mag, radius_law())

    assert np.sum(grid["density"]) == pytest.approx(
        mag.integrated_density(), rel=2e-3,
    )


def test_prior_draws_radius_first_then_brightness_conditioned_on_radius():
    prior = JointGalaxyPopulationPrior(active_payload())
    rng = np.random.default_rng(14)
    geometry = prior.sample_geometry(rng)
    magnitude, flux = prior.sample_brightness(
        rng, radius_arcsec=geometry.re_arcsec,
    )

    assert np.isnan(geometry.target_vis_mag)
    assert 0.03 <= geometry.re_arcsec < 10.0
    assert 14.0 <= magnitude < 29.0
    assert flux > 0.0
    assert prior.morphology_mode == "balanced_random_tng_atlas"


def test_old_cosmos_joint_artifacts_fail_closed():
    payload = active_payload()
    payload.update({"version": 3, "kind": "joint_analytical_tng_draw"})

    with pytest.raises(ValueError, match="unsupported version"):
        JointGalaxyPopulationPrior(payload)


def test_euclid_candidate_activates_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    payload = active_payload()
    payload["active"] = False
    path = joint_galaxy_candidate_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    updates = []
    monkeypatch.setattr(
        "euclid_polish.web.job_config.update", lambda patch: updates.append(patch),
    )

    active = activate_joint_galaxy_candidate()

    assert active["active"] is True
    assert active_joint_galaxy_path().is_file()
    assert updates == [{
        "galaxy_density_arcmin2": pytest.approx(
            magnitude_law().integrated_density(),
        ),
    }]


def test_candidate_fit_uses_only_euclid_brightness_and_sersic_radius(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    catalog = tmp_path / "euclid.csv"
    meta = tmp_path / "euclid.json"
    fieldnames = [
        "flux_vis_2fwhm_aper_uJy", "morph_sersic_vis_radius_arcsec",
        "morph_sersic_visnir_flags", "phz_gal_prob", "spurious_prob",
    ]
    rng = np.random.default_rng(9)
    with catalog.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for magnitude in np.linspace(18.0, 27.0, 500):
            writer.writerow({
                "flux_vis_2fwhm_aper_uJy": float(ab_mag_to_uJy(magnitude)),
                "morph_sersic_vis_radius_arcsec": 10.0 ** (
                    -0.4 - 0.06 * (magnitude - 23.0)
                    + rng.normal(0.0, 0.12)
                ),
                "morph_sersic_visnir_flags": 0,
                "phz_gal_prob": 0.8,
                "spurious_prob": 0.0,
            })
    meta.write_text(json.dumps({
        "catalog_version": 7, "rows": 500, "area_arcmin2": 100.0,
    }))
    monkeypatch.setattr(
        "euclid_polish.web.helpers.population_comparison.euclid_catalog_path",
        lambda: catalog,
    )
    monkeypatch.setattr(
        "euclid_polish.web.helpers.population_comparison.euclid_catalog_meta_path",
        lambda: meta,
    )
    monkeypatch.setattr(
        "euclid_polish.web.helpers.q1_galaxy_counts."
        "read_q1_galaxy_aperture_fit",
        lambda: {"apertures": {"f2": {
            "law": magnitude_law().to_payload(),
            "x": [14.0, 21.5, 29.0],
            "density": [0.1, 3.0, 100.0],
        }}},
    )
    monkeypatch.setattr(
        "euclid_polish.web.helpers.q1_galaxy_counts."
        "read_q1_galaxy_aperture_counts",
        lambda: {
            "faint": 28.0,
            "apertures": {"f2": {"bins": [
                {"mag_lo": 14.0, "mag_hi": 15.0,
                 "density_arcmin2_mag": 0.1},
                {"mag_lo": 27.0, "mag_hi": 28.0,
                 "density_arcmin2_mag": 50.0},
            ]}},
        },
    )

    payload = fit_euclid_joint_galaxy_candidate()

    assert payload["version"] == 4
    assert payload["provenance"]["cosmos_used"] is False
    assert payload["radius_law"]["slope_log10_arcsec_per_mag"] == pytest.approx(
        -0.06, abs=0.01,
    )
    assert payload["plots"]["radius"]["observed_density"]
    assert payload["plots"]["conditional_radius"]["model_mean_log10_arcsec"]
    assert payload["magnitude_plot"]["extrapolated_interval"] == [28.0, 29.0]
    assert "model" not in payload
    assert "redshift" not in json.dumps(payload).lower()
