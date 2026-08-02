from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.sky.generation.sky_simulator import (
    SkySimulator,
    SkySimulatorConfig,
)
from euclid_polish.sky.generation.stellar_sed import EmpiricalStellarPrior
from euclid_polish.web.helpers.population_calibration import (
    activate_galaxy_recommendation,
    active_transfer_path,
    density_calibration_path,
    fit_density_response,
    fit_local_catalog_density,
    galaxy_recommendation_state,
)
from euclid_polish.web.helpers.star_population import (
    _weighted_summary,
    fit_star_population,
)


def test_probability_weighted_summary_and_empirical_magnitude_cdf():
    values = np.asarray([20.0, 22.0])
    weights = np.asarray([0.25, 0.75])
    summary = _weighted_summary(
        values, weights, area_arcmin2=2.0,
        classification_variance=float(np.sum(weights * (1.0 - weights))),
    )
    assert summary["expected_count"] == pytest.approx(1.0)
    assert summary["density_arcmin2"] == pytest.approx(0.5)
    assert summary["mean"] == pytest.approx(21.5)
    assert summary["effective_n"] == pytest.approx(1.6)
    assert summary["classification_sigma_count"] == pytest.approx(0.612372)

    prior = EmpiricalStellarPrior.from_payload({
        "gaia": {"bp_rp_quantiles": [0.5, 1.0],
                 "temperature_quantiles_k": [4000.0, 6000.0]},
        "euclid_mapping": {
            "g_to_band_offset_coefficients": {
                key: [0.0, 0.0, 0.0]
                for key in ("mag_vis", "mag_y_e", "mag_j_e", "mag_h_e")
            },
            "residual_covariance": np.eye(4).tolist(),
        },
        "population": {"magnitude_distribution": {
            "edges": [20.0, 21.0, 22.0], "cdf": [0.0, 0.25, 1.0],
        }},
    })
    draws = np.asarray([
        prior.sample_magnitude(
            np.random.default_rng(seed), slope=0.2,
            m_bright=20.0, m_faint=22.0,
        )
        for seed in range(2000)
    ])
    assert np.mean(draws < 21.0) == pytest.approx(0.25, abs=0.04)


def test_density_response_is_reproducible_and_rejects_wrong_transfer():
    densities = [240.0, 280.0, 320.0, 360.0, 400.0]
    fields = [[density / 10 + offset for offset in (-1, 0, 1, 0)]
              for density in densities]
    real = [31.0, 32.0, 33.0, 32.0]
    first = fit_density_response(
        densities, fields, real, transfer_fingerprint="same",
        active_transfer_fingerprint="same", field_area_arcmin2=1.0,
        euclid_cone_detection_densities=[30.0, 32.0, 34.0],
        bootstraps=100, seed=5,
    )
    second = fit_density_response(
        densities, fields, real, transfer_fingerprint="same",
        active_transfer_fingerprint="same", field_area_arcmin2=1.0,
        euclid_cone_detection_densities=[30.0, 32.0, 34.0],
        bootstraps=100, seed=5,
    )
    assert first["valid"]
    assert first["recommended_density_arcmin2"] == pytest.approx(320.0)
    assert first["interval_arcmin2"] == second["interval_arcmin2"]
    assert first["euclid_cones"] == 3

    mismatch = fit_density_response(
        densities, fields, real, transfer_fingerprint="old",
        active_transfer_fingerprint="new", field_area_arcmin2=1.0,
        bootstraps=100,
    )
    assert not mismatch["valid"]
    assert "different" in " ".join(mismatch["warnings"]) or "not the active" in " ".join(mismatch["warnings"])


def test_local_catalog_fit_recovers_raw_density_without_rendering(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    root = tmp_path / "population_comparison"
    prior_path = root / "cosmos2025" / "prior.npz"
    fit_path = root / "cosmos2025" / "fit.json"
    prior_path.parent.mkdir(parents=True)
    monkeypatch.setattr(Config, "COSMOS_TNG_PRIOR_PATH", str(prior_path))
    monkeypatch.setattr(Config, "COSMOS_EUCLID_FIT_PATH", str(fit_path))
    size = 2_000
    np.savez_compressed(
        prior_path,
        catalog_id=np.arange(size),
        mag_hst_f814w=np.full(size, 24.0),
        z_phot=np.full(size, 1.0),
        logmass_lephare=np.full(size, 10.0),
        re_combined_arcsec=np.full(size, 0.4),
        generator_ready=np.ones(size, dtype=bool),
    )
    import euclid_polish.sky.generation.redshift_model as redshift_module
    import euclid_polish.sky.generation.tng_galaxy as galaxy_module
    import euclid_polish.sky.generation.tng_radius_manifest as radius_module

    monkeypatch.setattr(
        radius_module, "validate_manifest",
        lambda _path: {"valid": True, "manifest_fingerprint": "radius-v1"},
    )
    monkeypatch.setattr(
        galaxy_module, "list_tng_galaxies",
        lambda _path: [("atlas/1", "1"), ("atlas/2", "2")],
    )
    monkeypatch.setattr(
        redshift_module, "load_tng_properties",
        lambda: {
            "1": {"mass_stars": 10.0 ** 9.8},
            "2": {"mass_stars": 10.0 ** 10.2},
        },
    )
    fit_path.write_text(json.dumps({
        "inputs": {
            "euclid_cone_count": 4,
            "euclid_area_arcmin2": 4.0,
            "euclid_cones": [{"star_id": str(index)} for index in range(4)],
        },
        "fit": {
            "vis_minus_f814w_mag": 0.0,
            "magnitude_slope": 1.0,
            "scatter_mag": 0.0,
            "completeness_m50": 24.0,
            "completeness_width_mag": 1.0,
            "poisson_deviance": 4.0,
            "dof": 4,
        },
    }))
    (root / "euclid_population_meta.json").write_text(json.dumps({
        "cone_count": 4,
        "area_arcmin2": 4.0,
        "cones": [{"star_id": str(index)} for index in range(4)],
    }))
    rows = []
    for cone_index in range(4):
            rows.extend({
                "type": "unknown",
                "spurious_prob": "0.0",
                "point_like_prob": "0.0",
                "mag_vis": "24.0",
            "cone_index": str(cone_index),
        } for _ in range(10))
    _write_csv(root / "euclid_population.csv", rows)

    first = fit_local_catalog_density(draws=10_000, bootstraps=100, seed=8)
    second = fit_local_catalog_density(draws=10_000, bootstraps=100, seed=8)

    assert first["valid"]
    assert first["retained_detection_fraction"] == pytest.approx(0.5)
    assert first["euclid_detected_density_arcmin2"] == pytest.approx(10.0)
    assert first["recommended_density_arcmin2"] == pytest.approx(20.0)
    assert first["cosmos_generator_rows"] == size
    assert first["morphology_model"]["eligible_cosmos_rows"] == size
    assert first["magnitude_fit_quality"]["valid"]
    assert first["calibration_fingerprint"] == second["calibration_fingerprint"]
    assert first["interval_arcmin2"] == second["interval_arcmin2"]
    assert density_calibration_path().exists()


def test_nested_thinning_keeps_nuisance_population_identical(monkeypatch):
    import euclid_polish.sky.generation.sky_simulator as module

    monkeypatch.setattr(module, "list_tng_galaxies", lambda _path: [("x", "1")])
    monkeypatch.setattr(module, "load_tng_properties", lambda _path: {})

    def build(density: float) -> dict:
        simulator = SkySimulator(object(), SkySimulatorConfig(
            image_size=10, pixel_scale=6.0,
            galaxy_density_arcmin2=density,
            galaxy_thinning_max_density_arcmin2=8.0,
            star_density_arcmin2=4.0, lens_density_arcmin2=0.0,
        ))
        simulator._add_tng_galaxy = lambda _canvas, rng: {
            "proposal": int(rng.integers(0, 2**31)),
        }
        return simulator.simulate_field(np.random.default_rng(91))[1]

    lower = build(4.0)
    upper = build(8.0)
    low_ids = {item["proposal"] for item in lower["galaxies"]}
    high_ids = {item["proposal"] for item in upper["galaxies"]}
    assert low_ids <= high_ids
    assert lower["stars"] == upper["stars"]
    assert lower["lenses"] == upper["lenses"]


def test_complete_generator_recommendation_can_activate_with_fit_warnings(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    fit_path = tmp_path / "fit.json"
    monkeypatch.setattr(Config, "COSMOS_EUCLID_FIT_PATH", str(fit_path))
    fit_path.write_text(json.dumps({
        "inputs": {"euclid_cone_count": 6},
        "fit": {
            "vis_minus_f814w_mag": 0.4,
            "magnitude_slope": 0.7,
            "scatter_mag": 1.0,
            "completeness_m50": 25.1,
            "completeness_width_mag": 0.5,
            "poisson_deviance": 60.0,
            "dof": 10,
        },
    }))
    from euclid_polish.sky.generation.cosmos_tng_prior import (
        brightness_transfer_payload,
    )
    transfer = brightness_transfer_payload(fit_path)
    assert transfer is not None
    density_calibration_path().parent.mkdir(parents=True, exist_ok=True)
    density_calibration_path().write_text(json.dumps({
        "valid": True,
        "warnings": [],
        "transfer_fingerprint": transfer["fingerprint"],
        "calibration_fingerprint": "sweep",
        "recommended_density_arcmin2": 315.0,
        "interval_arcmin2": {"median": 315, "p16": 295, "p84": 338},
    }))
    updates = []
    monkeypatch.setattr(
        "euclid_polish.web.job_config.update", lambda patch: updates.append(patch),
    )

    state = galaxy_recommendation_state()
    assert state["recommendation_available"]
    assert not state["validated"]
    assert state["generator_parameters"] == {
        "galaxy_density_arcmin2": 315.0,
        "cosmos_vis_offset_mag": 0.4,
        "cosmos_vis_magnitude_slope": 0.7,
        "cosmos_vis_scatter_mag": 1.0,
    }

    activated = activate_galaxy_recommendation()
    assert activated["active"]
    assert activated["brightness_transfer"][
        "activated_with_quality_warnings"
    ]
    assert json.loads(active_transfer_path().read_text())["fingerprint"] == transfer[
        "fingerprint"
    ]
    assert updates == [{"galaxy_density_arcmin2": 315.0}]


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_star_fit_excludes_centres_and_samples_correlated_euclid_colors(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    root = tmp_path / "population_comparison"
    gaia = []
    euclid = []
    for index in range(24):
        cone = index % 2
        source_id = str(1000 + index)
        color = 0.5 + 0.05 * index
        g_mag = 16.0 + 0.18 * index
        gaia.append({
            "source_id": source_id, "cone_index": cone, "ra": 1,
            "dec": 2, "g_mag": g_mag, "bp_mag": g_mag + color / 2,
            "rp_mag": g_mag - color / 2, "bp_rp": color,
            "temperature_k": 8000 - 150 * index, "extinction_g_mag": 0.1,
            "central_selected_star": 1 if index < 2 else 0,
        })
        euclid.append({
            "object_id": index, "gaia_id": source_id, "type": "star",
            "point_like_prob": "1.0",
            "mag_vis": g_mag + 0.2 * color,
            "mag_y_e": g_mag - 0.4 * color,
            "mag_j_e": g_mag - 0.55 * color,
            "mag_h_e": g_mag - 0.60 * color,
        })
    _write_csv(root / "gaia_population.csv", gaia)
    _write_csv(root / "euclid_population.csv", euclid)
    (root / "gaia_population.meta.json").write_text(json.dumps({
        "cone_count": 2, "radius_arcmin": 2.0,
        "area_arcmin2": 8 * np.pi, "euclid_cone_selection_seed": 7,
    }))

    fit = fit_star_population()
    assert fit["valid"]
    assert fit["cone_provenance"]["central_sources_excluded"] == 2
    assert fit["euclid_mapping"]["matched_stars"] == 24
    assert fit["diagnostics"]["star_density_per_cone"]["observed"] == pytest.approx([
        11 / (4 * np.pi), 11 / (4 * np.pi),
    ])
    magnitude = fit["diagnostics"]["parameters"]["mag_vis"]
    assert magnitude["density_unit"] == "point sources / arcmin² / mag"
    assert magnitude["observed_count"] == 22
    assert magnitude["weighted_count"] == pytest.approx(22.0)
    assert sum(magnitude["euclid_weighted"]) == pytest.approx(0.0)
    assert sum(value for value in magnitude["observed"] if value is not None) * 0.5 \
        == pytest.approx(22 / (8 * np.pi))
    assert fit["population"]["magnitude_distribution"][
        "euclid_faint_expected_count"
    ] == pytest.approx(0.0)
    assert any(
        value > 0.0
        for centre, value in zip(magnitude["x"], magnitude["observed"], strict=True)
        if centre > magnitude["observed_limit_mag"]
    ) is False

    prior = EmpiricalStellarPrior.from_payload(fit)
    sed = prior.sample(np.random.default_rng(8), 21.5)
    assert sed.magnitudes["VIS"] == pytest.approx(21.5)
    assert set(sed.magnitudes) == {"VIS", "Y_E", "J_E", "H_E"}
