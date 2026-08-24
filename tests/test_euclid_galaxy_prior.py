from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ndtr

from euclid_polish.config import Config
from euclid_polish.population.euclid_galaxy_prior import (
    BRIGHT_BRIDGE_JOIN_MAGNITUDES,
    GALAXY_FAINT_DENSITY_CAP_ARCMIN2_MAG,
    JOINT_EUCLID_GALAXY_KIND,
    JOINT_EUCLID_GALAXY_VERSION,
    RADIUS_MODEL_VERSION,
    ConditionalRadiusLaw,
    fit_linear_conditional_radius_law_from_binned_counts,
    joint_density_grid,
)
from euclid_polish.population.magnitude_law import (
    ContinuousBrightBridgeFaintCappedMagnitudeLaw,
    StraightMagnitudeLaw,
)
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


def current_radius_law() -> ConditionalRadiusLaw:
    return ConditionalRadiusLaw(
        version=RADIUS_MODEL_VERSION,
        pivot_mag=23.0,
        intercept_log10_arcsec=-0.4,
        slope_log10_arcsec_per_mag=-0.08,
        scatter_dex=0.18,
        log_radius_min=np.log10(0.03),
        log_radius_max=np.log10(10.0),
        fitted_rows=1000,
        clipped_rows=0,
        weighted_rows=800.0,
        residual_rms_dex=0.18,
        r_squared=0.3,
        covariance=((1e-4, 0.0), (0.0, 1e-5)),
        selection="fixture",
        fit_min_selected_per_magnitude_bin=20,
        fit_effective_weight_cap=1000.0,
        fit_faint_magnitude=25.5,
    )


def current_magnitude_law() -> ContinuousBrightBridgeFaintCappedMagnitudeLaw:
    return ContinuousBrightBridgeFaintCappedMagnitudeLaw(
        straight_law=magnitude_law(),
        bright_slopes=(0.8, 0.3, 0.5),
        bright_join_magnitudes=BRIGHT_BRIDGE_JOIN_MAGNITUDES,
        density_cap_arcmin2_mag=GALAXY_FAINT_DENSITY_CAP_ARCMIN2_MAG,
    )


def active_payload() -> dict:
    fitted_mag = magnitude_law()
    mag = current_magnitude_law()
    plot_x = [14.0, mag.break_magnitude, 29.0]
    return {
        "version": JOINT_EUCLID_GALAXY_VERSION,
        "kind": JOINT_EUCLID_GALAXY_KIND,
        "valid": True,
        "active": True,
        "fingerprint": "b" * 64,
        "fitted_magnitude_law": fitted_mag.to_payload(),
        "magnitude_law": mag.to_payload(),
        "radius_law": current_radius_law().to_payload(),
        "magnitude_plot": {
            "law": {"x": [14.0, 29.0], "density": [0.1, 100.0]},
            "generation_law": {
                "x": plot_x,
                "density": mag.density(plot_x).tolist(),
            },
        },
        "plots": {
            "radius": {
                "x": [-1.0, 0.0],
                "density": [1.0, 1.0],
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
            "fitted_surface_density_arcmin2": (
                fitted_mag.integrated_density()
            ),
            "vis_magnitude_min": mag.mag_bright,
            "vis_magnitude_max": mag.mag_faint,
            "fitted_vis_magnitude_max": fitted_mag.mag_faint,
            "faint_end_policy": "cap_differential_counts_after_break",
        },
    }

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pivot_mag", float("nan")),
        ("pivot_mag", float("inf")),
        ("log_radius_min", float("-inf")),
        ("log_radius_max", float("inf")),
    ],
)
def test_radius_law_rejects_nonfinite_geometry(field, value):
    payload = current_radius_law().to_payload()
    payload[field] = value

    with pytest.raises(ValueError, match="radius law is invalid"):
        ConditionalRadiusLaw.from_payload(payload)


@pytest.mark.parametrize(
    "covariance",
    [
        ((1e-4, 0.0), (0.0, float("nan"))),
        ((1e-4, 0.0, 0.0), (0.0, 1e-5, 0.0)),
    ],
)
def test_radius_law_rejects_malformed_covariance(covariance):
    payload = current_radius_law().to_payload()
    payload["covariance"] = covariance

    with pytest.raises(ValueError, match="radius law is invalid"):
        ConditionalRadiusLaw.from_payload(payload)


def test_joint_grid_integrates_to_current_brightness_density():
    mag = current_magnitude_law()
    grid = joint_density_grid(mag, current_radius_law())

    assert np.sum(grid["density"]) == pytest.approx(
        mag.integrated_density(), rel=2e-3,
    )


def test_linear_binned_radius_fit_has_no_bright_break_or_generated_tail():
    magnitude_edges = np.linspace(14.0, 28.0, 141)
    radius_edges = np.linspace(np.log10(0.03), np.log10(10.0), 51)
    magnitude = 0.5 * (magnitude_edges[:-1] + magnitude_edges[1:])
    expected_intercept, expected_slope, expected_scatter = -0.42, -0.09, 0.17
    mean = expected_intercept + expected_slope * (magnitude - 23.0)
    upper = (radius_edges[None, 1:] - mean[:, None]) / expected_scatter
    lower = (radius_edges[None, :-1] - mean[:, None]) / expected_scatter
    probability = ndtr(upper) - ndtr(lower)
    probability /= np.sum(probability, axis=1, keepdims=True)
    expected = 800.0 * probability
    selected = np.rint(1000.0 * probability)

    law = fit_linear_conditional_radius_law_from_binned_counts(
        magnitude_edges,
        radius_edges,
        selected,
        expected,
    )

    assert law.version == RADIUS_MODEL_VERSION
    assert law.intercept_log10_arcsec == pytest.approx(
        expected_intercept, abs=2e-3,
    )
    assert law.slope_log10_arcsec_per_mag == pytest.approx(
        expected_slope, abs=2e-3,
    )
    assert law.scatter_dex == pytest.approx(expected_scatter, abs=2e-3)
    assert law.mean([15.0, 23.0, 27.0]) == pytest.approx(
        expected_intercept
        + expected_slope * (np.asarray([15.0, 23.0, 27.0]) - 23.0),
        abs=2e-3,
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
    assert 14.0 <= magnitude < prior.magnitude_law.mag_faint
    assert flux > 0.0
    assert prior.morphology_mode == "balanced_random_tng_atlas"
    assert isinstance(
        prior.magnitude_law,
        ContinuousBrightBridgeFaintCappedMagnitudeLaw,
    )
    assert prior.radius_law.version == RADIUS_MODEL_VERSION
    assert prior.surface_density_arcmin2 == pytest.approx(
        current_magnitude_law().integrated_density()
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

    with pytest.raises(ValueError, match="current version"):
        JointGalaxyPopulationPrior(payload)


def test_hard_truncation_version_six_artifacts_fail_closed():
    payload = active_payload()
    payload["version"] = 6

    with pytest.raises(ValueError, match="current version"):
        JointGalaxyPopulationPrior(payload)


def test_joint_prior_rejects_shifted_v11_bright_joins():
    payload = active_payload()
    shifted = ContinuousBrightBridgeFaintCappedMagnitudeLaw(
        straight_law=magnitude_law(),
        bright_slopes=(0.8, 0.3, 0.5),
        bright_join_magnitudes=(16.5, 19.1, 20.8),
        density_cap_arcmin2_mag=GALAXY_FAINT_DENSITY_CAP_ARCMIN2_MAG,
    )
    payload["magnitude_law"] = shifted.to_payload()
    payload["generation"]["surface_density_arcmin2"] = (
        shifted.integrated_density()
    )

    with pytest.raises(ValueError, match="brightness contract"):
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
            current_magnitude_law().integrated_density(),
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
    mean_log10 = -0.4 - 0.06 * (magnitude - 23.0)
    upper = (log_radius_edges[None, 1:] - mean_log10[:, None]) / 0.12
    lower = (log_radius_edges[None, :-1] - mean_log10[:, None]) / 0.12
    probability = ndtr(upper) - ndtr(lower)
    probability /= np.sum(probability, axis=1, keepdims=True)
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
    bright_count_law = current_magnitude_law()
    bright_edges = np.linspace(14.0, 21.0, 71)
    bright_count_bins = []
    for mag_lo, mag_hi in zip(bright_edges[:-1], bright_edges[1:], strict=True):
        mass = 0.0
        for bright, faint, slope, intercept in (
            bright_count_law._line_components()
        ):
            overlap_bright = max(float(mag_lo), bright)
            overlap_faint = min(float(mag_hi), faint)
            if overlap_faint > overlap_bright:
                mass += bright_count_law._line_integral(
                    slope,
                    intercept,
                    overlap_bright,
                    overlap_faint,
                )
        bright_count_bins.append({
            "mag_lo": float(mag_lo),
            "mag_hi": float(mag_hi),
            "density_arcmin2_mag": float(mass / (mag_hi - mag_lo)),
            "expected_galaxies": float(100.0 * mass),
        })
    monkeypatch.setattr(
        "euclid_polish.web.helpers.q1_galaxy_counts."
        "read_q1_galaxy_aperture_fit",
        lambda: {"apertures": {"f2": {
            "law": magnitude_law().to_payload(),
            "x": [14.0, 21.5, 29.0],
            "density": [0.1, 3.0, 100.0],
            "fit_bin_start": 59,
        }}},
    )
    monkeypatch.setattr(
        "euclid_polish.web.helpers.q1_galaxy_counts."
        "read_q1_galaxy_aperture_counts",
        lambda: {
            "complete": True,
            "faint": 28.0,
            "footprint_area_arcmin2": 100.0,
            "apertures": {"f2": {"bins": bright_count_bins}},
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
            "selection": "fixture circularized morphology-quality selection",
            "acquisition": "fixture grouped aggregate",
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
    assert payload["radius_law"]["version"] == RADIUS_MODEL_VERSION
    assert payload["plots"]["radius"]["observed_density"]
    assert payload["plots"]["radius"]["q1_weighted_density"]
    assert "circularized" in (
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
    assert payload["magnitude_plot"]["continuous_bright_interval"][0] == 14.0
    assert payload["magnitude_plot"]["continuous_bright_interval"][1] == (
        pytest.approx(BRIGHT_BRIDGE_JOIN_MAGNITUDES[-1])
    )
    fitted_bright_bins = [
        item for item in bright_count_bins
        if item["mag_lo"] < BRIGHT_BRIDGE_JOIN_MAGNITUDES[-1] - 1e-10
    ]
    assert payload["magnitude_plot"]["bright_fit_diagnostics"] == {
        "bright_fit_bin_count": len(fitted_bright_bins),
        "bright_fit_zero_bin_count": 0,
        "bright_fit_expected_galaxies": pytest.approx(sum(
            item["expected_galaxies"] for item in fitted_bright_bins
        )),
        "bright_fit_poisson_deviance": pytest.approx(0.0, abs=1e-5),
        "bright_fit_deviance_per_bin": pytest.approx(0.0, abs=1e-7),
        "bright_fit_parameter_count": 3,
        "bright_bridge_join_magnitudes": list(
            BRIGHT_BRIDGE_JOIN_MAGNITUDES
        ),
        "bright_bridge_slopes": pytest.approx(
            bright_count_law.bright_slopes, abs=1e-5,
        ),
    }
    assert payload["magnitude_law"]["kind"] == (
        "continuous_three_slope_bright_bridge_main_flat_faint_counts"
    )
    assert payload["generation"]["faint_radius_policy"] == (
        "straight_truncated_gaussian_at_all_magnitudes_no_tail"
    )
    generated_density = payload["magnitude_plot"]["generation_law"]["density"]
    assert generated_density[-1] == pytest.approx(
        GALAXY_FAINT_DENSITY_CAP_ARCMIN2_MAG
    )
    assert generated_density[-20:] == pytest.approx(
        [GALAXY_FAINT_DENSITY_CAP_ARCMIN2_MAG] * 20
    )
    assert "model" not in payload
    assert "redshift" not in json.dumps(payload).lower()
