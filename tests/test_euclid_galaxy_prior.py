from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ndtr

from euclid_polish.config import Config
from euclid_polish.population.euclid_galaxy_prior import (
    GALAXY_FAINT_DENSITY_CAP_ARCMIN2_MAG,
    JOINT_EUCLID_GALAXY_VERSION,
    ConditionalRadiusLaw,
    fit_broken_conditional_radius_law_from_binned_counts,
    fit_conditional_radius_law,
    fit_conditional_radius_law_from_aggregate_moments,
    fit_conditional_radius_law_from_binned_counts,
    generation_magnitude_law,
    joint_density_grid,
)
from euclid_polish.population.magnitude_law import StraightMagnitudeLaw
from euclid_polish.sky.generation.cosmos_tng_prior import (
    JointGalaxyPopulationPrior,
)
from euclid_polish.sky.generation.sky_simulator import (
    SkySimulator,
    SkySimulatorConfig,
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
    fitted_mag = magnitude_law()
    mag = generation_magnitude_law(fitted_mag)
    plot_x = [14.0, mag.break_magnitude, 29.0]
    return {
        "version": JOINT_EUCLID_GALAXY_VERSION,
        "kind": "euclid_vis2fwhm_sersic_re_joint",
        "valid": True, "active": True, "fingerprint": "b" * 64,
        "fitted_magnitude_law": fitted_mag.to_payload(),
        "magnitude_law": mag.to_payload(),
        "radius_law": radius_law().to_payload(),
        "magnitude_plot": {
            "law": {"x": [14.0, 29.0], "density": [0.1, 100.0]},
            "generation_law": {
                "x": plot_x,
                "density": mag.density(plot_x).tolist(),
            },
        },
        "plots": {
            "radius": {
                "x": [-1.0, 0.0], "density": [1.0, 1.0],
                "q1_weighted_density": [1.0, 1.0],
                "observed_density": [1.0, 1.0],
            },
            "conditional_radius": {
                "magnitude": [14.0, 29.0],
                "model_mean_log10_arcsec": [0.32, -0.88],
            },
        },
        "generation": {
            "surface_density_arcmin2": mag.integrated_density(),
            "differential_density_cap_arcmin2_mag": (
                GALAXY_FAINT_DENSITY_CAP_ARCMIN2_MAG
            ),
            "break_magnitude": mag.break_magnitude,
            "fitted_surface_density_arcmin2": fitted_mag.integrated_density(),
            "vis_magnitude_min": mag.mag_bright,
            "vis_magnitude_max": mag.mag_faint,
            "fitted_vis_magnitude_max": fitted_mag.mag_faint,
            "faint_end_policy": "cap_differential_counts_after_break",
        },
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


def test_aggregate_radius_moments_recover_straight_relation():
    magnitude = np.linspace(18.0, 27.0, 40)
    expected_intercept, expected_slope, expected_scatter = -0.35, -0.075, 0.16
    mean_log10 = expected_intercept + expected_slope * (magnitude - 23.0)
    sigma_ln = expected_scatter * np.log(10.0)
    mean_ln = mean_log10 * np.log(10.0)
    expected = np.full(magnitude.shape, 80.0)
    first = expected * np.exp(mean_ln + 0.5 * sigma_ln**2)
    second = expected * np.exp(2.0 * mean_ln + 2.0 * sigma_ln**2)

    law = fit_conditional_radius_law_from_aggregate_moments(
        magnitude, np.full(magnitude.shape, 100), expected, first, second,
    )

    assert law.intercept_log10_arcsec == pytest.approx(expected_intercept)
    assert law.slope_log10_arcsec_per_mag == pytest.approx(expected_slope)
    assert law.scatter_dex == pytest.approx(expected_scatter)


def test_binned_radius_counts_recover_bounded_conditional_relation():
    magnitude_edges = np.linspace(18.0, 27.0, 91)
    radius_edges = np.linspace(np.log10(0.03), np.log10(10.0), 51)
    magnitude = 0.5 * (magnitude_edges[:-1] + magnitude_edges[1:])
    expected_intercept, expected_slope, expected_scatter = -0.35, -0.075, 0.16
    mean = expected_intercept + expected_slope * (magnitude - 23.0)
    upper = (radius_edges[None, 1:] - mean[:, None]) / expected_scatter
    lower = (radius_edges[None, :-1] - mean[:, None]) / expected_scatter
    probability = ndtr(upper) - ndtr(lower)
    probability /= np.sum(probability, axis=1, keepdims=True)
    expected = 1000.0 * probability
    selected = np.rint(1250.0 * probability)

    law = fit_conditional_radius_law_from_binned_counts(
        magnitude_edges,
        radius_edges,
        selected,
        expected,
        fit_bright=18.0,
        fit_faint=27.0,
    )

    assert law.intercept_log10_arcsec == pytest.approx(expected_intercept, abs=2e-3)
    assert law.slope_log10_arcsec_per_mag == pytest.approx(expected_slope, abs=2e-3)
    assert law.scatter_dex == pytest.approx(expected_scatter, abs=2e-3)
    assert "bounded aggregate" in law.selection


def test_broken_radius_counts_recover_plateau_jump_slope_and_tail():
    magnitude_edges = np.linspace(14.0, 28.0, 141)
    radius_edges = np.linspace(np.log10(0.03), np.log10(10.0), 41)
    magnitude = 0.5 * (magnitude_edges[:-1] + magnitude_edges[1:])
    bright, intercept, slope, scatter, tail = -0.8, -0.35, -0.08, 0.16, 0.12
    core_mean = np.where(
        magnitude < 18.0,
        bright,
        intercept + slope * (magnitude - 23.0),
    )
    upper = (radius_edges[None, 1:] - core_mean[:, None]) / scatter
    lower = (radius_edges[None, :-1] - core_mean[:, None]) / scatter
    core = ndtr(upper) - ndtr(lower)
    core /= np.sum(core, axis=1, keepdims=True)
    uniform = np.diff(radius_edges) / (radius_edges[-1] - radius_edges[0])
    probability = (1.0 - tail) * core + tail * uniform[None, :]
    expected = 1000.0 * probability
    selected = np.rint(1250.0 * probability)

    law = fit_broken_conditional_radius_law_from_binned_counts(
        magnitude_edges, radius_edges, selected, expected,
    )

    assert law.version == 2
    assert law.bright_intercept_log10_arcsec == pytest.approx(bright, abs=0.02)
    assert law.break_magnitude == pytest.approx(18.0, abs=0.11)
    assert law.intercept_log10_arcsec == pytest.approx(intercept, abs=0.02)
    assert law.slope_log10_arcsec_per_mag == pytest.approx(slope, abs=0.01)
    assert law.scatter_dex == pytest.approx(scatter, abs=0.02)
    assert law.tail_fraction == pytest.approx(tail, abs=0.02)
    assert law.tail_distribution == "uniform_log_radius"


def test_prior_draws_radius_first_then_brightness_conditioned_on_radius():
    prior = JointGalaxyPopulationPrior(active_payload())
    rng = np.random.default_rng(14)
    geometry = prior.sample_geometry(rng)
    magnitude, flux = prior.sample_brightness(
        rng, radius_arcsec=geometry.re_arcsec,
    )

    assert np.isnan(geometry.target_vis_mag)
    assert 0.03 <= geometry.re_arcsec < 10.0
    assert 14.0 <= magnitude < prior.magnitude_law.mag_faint
    assert flux > 0.0
    assert prior.morphology_mode == "balanced_random_tng_atlas"
    assert prior.surface_density_arcmin2 == pytest.approx(
        generation_magnitude_law(magnitude_law()).integrated_density()
    )


def test_simulator_rejects_density_above_activated_magnitude_law():
    prior = JointGalaxyPopulationPrior(active_payload())
    with pytest.raises(ValueError, match="magnitude-law population limit"):
        SkySimulator(
            prior,
            SkySimulatorConfig(
                galaxy_density_arcmin2=prior.surface_density_arcmin2 + 0.1,
                star_density_arcmin2=0.0,
                lens_density_arcmin2=0.0,
            ),
        )


def test_simulator_accepts_density_equal_to_activated_magnitude_law(
    monkeypatch,
):
    import euclid_polish.sky.generation.sky_simulator as module

    monkeypatch.setattr(
        module, "validate_manifest", lambda *args, **kwargs: {"valid": True},
    )
    monkeypatch.setattr(
        module, "load_manifest",
        lambda *args, **kwargs: {"manifest_fingerprint": "r" * 64},
    )
    monkeypatch.setattr(module, "radius_lookup", lambda payload: {})
    monkeypatch.setattr(module, "list_tng_galaxies", lambda path: [("o", "1")])
    monkeypatch.setattr(module, "load_tng_properties", lambda path: {})
    prior = JointGalaxyPopulationPrior(active_payload())

    SkySimulator(
        prior,
        SkySimulatorConfig(
            galaxy_density_arcmin2=prior.surface_density_arcmin2,
            star_density_arcmin2=0.0,
            lens_density_arcmin2=0.0,
        ),
    )


def test_staged_generator_selects_and_renders_donor_before_brightness(
    monkeypatch,
):
    import euclid_polish.sky.generation.sky_simulator as module

    prior = JointGalaxyPopulationPrior(active_payload())
    events = []
    original_brightness = prior.sample_brightness

    def sample_brightness(rng, *, radius_arcsec=None):
        events.append("brightness")
        return original_brightness(rng, radius_arcsec=radius_arcsec)

    prior.sample_brightness = sample_brightness
    simulator = object.__new__(SkySimulator)
    simulator.population_prior = prior
    simulator.config = SimpleNamespace(pixel_scale=0.05)
    simulator._radius_lookup = {("42", 1): 5.0}
    simulator._radius_manifest_fingerprint = "r" * 64
    simulator._tng_max_output_side = 65

    def pick_donor(_rng):
        events.append("donor")
        return [("atlas", "42")], {}

    def render_donor(*_args, **_kwargs):
        events.append("render")
        return np.ones((3, 3, 4), dtype=np.float32), {}

    class StopAfterBrightness(Exception):
        pass

    def stop_at_psf(_rng):
        events.append("psf")
        raise StopAfterBrightness

    simulator._pick_random_field_galaxy = pick_donor
    simulator._draw_aperture_psf = stop_at_psf
    monkeypatch.setattr(module, "sample_tng_stamp", render_donor)

    with pytest.raises(StopAfterBrightness):
        simulator._add_tng_galaxy(
            np.zeros((8, 8, 4), dtype=np.float32),
            np.random.default_rng(19),
        )

    assert events == ["donor", "render", "brightness", "psf"]


def test_old_cosmos_joint_artifacts_fail_closed():
    payload = active_payload()
    payload.update({"version": 3, "kind": "joint_analytical_tng_draw"})

    with pytest.raises(ValueError, match="unsupported version"):
        JointGalaxyPopulationPrior(payload)


def test_hard_truncation_version_six_artifacts_fail_closed():
    payload = active_payload()
    payload["version"] = 6

    with pytest.raises(ValueError, match="unsupported version"):
        JointGalaxyPopulationPrior(payload)


def test_previous_version_seven_active_prior_remains_loadable():
    payload = active_payload()
    payload["version"] = 7

    prior = JointGalaxyPopulationPrior(payload)

    assert "_v7_" in prior.population_label


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
            generation_magnitude_law(magnitude_law()).integrated_density(),
        ),
    }]


def test_candidate_fit_uses_only_aggregate_euclid_brightness_and_sersic_radius(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path))
    from euclid_polish.web.helpers.q1_galaxy_counts import (
        q1_galaxy_counts_path,
        q1_galaxy_fit_path,
    )
    from euclid_polish.web.helpers.q1_galaxy_radius_statistics import (
        q1_galaxy_radius_statistics_path,
    )

    for path in (
        q1_galaxy_counts_path(), q1_galaxy_fit_path(),
        q1_galaxy_radius_statistics_path(),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    magnitude_edges = np.linspace(14.0, 28.0, 141)
    magnitude = 0.5 * (magnitude_edges[:-1] + magnitude_edges[1:])
    radius_edges = np.geomspace(0.03, 10.0, 31)
    log_radius_edges = np.log10(radius_edges)
    mean_log10 = np.where(
        magnitude < 18.0,
        -0.8,
        -0.4 - 0.06 * (magnitude - 23.0),
    )
    upper = (log_radius_edges[None, 1:] - mean_log10[:, None]) / 0.12
    lower = (log_radius_edges[None, :-1] - mean_log10[:, None]) / 0.12
    probability = ndtr(upper) - ndtr(lower)
    probability /= np.sum(probability, axis=1, keepdims=True)
    uniform_probability = np.diff(log_radius_edges) / (
        log_radius_edges[-1] - log_radius_edges[0]
    )
    probability = 0.9 * probability + 0.1 * uniform_probability[None, :]
    expected_grid = 80.0 * probability
    selected_grid = np.rint(100.0 * probability)
    magnitude_bins = [
        {
            "mag_lo": float(value - 0.05), "mag_hi": float(value + 0.05),
            "selected_radii": int(np.sum(selected_grid[index])),
            "expected_radii": float(np.sum(expected_grid[index])),
        }
        for index, value in enumerate(magnitude)
    ]
    joint_bins = [
        {
            "magnitude_bin": mag_index,
            "radius_bin": radius_index,
            "selected_radii": int(selected_grid[mag_index, radius_index]),
            "expected_radii": float(expected_grid[mag_index, radius_index]),
        }
        for mag_index in range(expected_grid.shape[0])
        for radius_index in range(expected_grid.shape[1])
        if expected_grid[mag_index, radius_index] > 0.0
    ]
    radius_bins = [
        {
            "density_arcmin2_dex": float(index + 1),
            "expected_radii": float(np.sum(expected_grid[:, index])),
        }
        for index in range(30)
    ]
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
            "complete": True,
            "faint": 28.0,
            "apertures": {"f2": {"bins": [
                {"mag_lo": 14.0, "mag_hi": 15.0,
                 "density_arcmin2_mag": 0.1},
                {"mag_lo": 27.0, "mag_hi": 28.0,
                 "density_arcmin2_mag": 50.0},
            ]}},
        },
    )
    monkeypatch.setattr(
        "euclid_polish.web.helpers.q1_galaxy_radius_statistics."
        "read_q1_galaxy_radius_statistics",
        lambda: {
            "complete": True,
            "footprint_area_arcmin2": 100.0,
            "magnitude_edges": magnitude_edges.tolist(),
            "magnitude_bins": magnitude_bins,
            "joint_bins": joint_bins,
            "radius_bins": radius_bins,
            "radius_edges_arcsec": radius_edges.tolist(),
        },
    )

    payload = fit_euclid_joint_galaxy_candidate()

    assert payload["version"] == JOINT_EUCLID_GALAXY_VERSION
    assert payload["provenance"]["cosmos_used"] is False
    assert payload["provenance"]["random_cones_used"] is False
    assert payload["provenance"]["object_catalog_used"] is False
    assert payload["radius_law"]["slope_log10_arcsec_per_mag"] == pytest.approx(
        -0.06, abs=0.01,
    )
    assert payload["radius_law"]["break_magnitude"] == pytest.approx(
        18.0, abs=0.11,
    )
    assert payload["radius_law"]["tail_fraction"] == pytest.approx(
        0.1, abs=0.02,
    )
    assert payload["plots"]["radius"]["observed_density"]
    assert payload["plots"]["radius"]["q1_weighted_density"]
    assert "nominal continuous-space" in (
        payload["plots"]["radius"]["model_semantics"]
    )
    assert payload["plots"]["conditional_radius"]["model_mean_log10_arcsec"]
    assert payload["magnitude_plot"]["extrapolated_interval"] == [28.0, 29.0]
    assert payload["generation"]["differential_density_cap_arcmin2_mag"] == (
        GALAXY_FAINT_DENSITY_CAP_ARCMIN2_MAG
    )
    assert payload["generation"]["vis_magnitude_max"] == 29.0
    assert 14.0 < payload["generation"]["break_magnitude"] < 29.0
    assert payload["magnitude_plot"]["generation_interval"] == [14.0, 29.0]
    generated_density = payload["magnitude_plot"]["generation_law"]["density"]
    assert generated_density[-1] == pytest.approx(
        GALAXY_FAINT_DENSITY_CAP_ARCMIN2_MAG
    )
    assert generated_density[-20:] == pytest.approx(
        [GALAXY_FAINT_DENSITY_CAP_ARCMIN2_MAG] * 20
    )
    assert "model" not in payload
    assert "redshift" not in json.dumps(payload).lower()
