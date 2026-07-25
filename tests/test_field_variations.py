"""Tests for field-scale bright-star wings and pointing noise variation."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.sky.observation.artifacts import ArtifactConfig
from euclid_polish.sky.observation.field_variations import (
    DistantStarWing,
    add_distant_star_wings,
    draw_distant_star_wings,
    draw_noise_scale_map,
)
from euclid_polish.sky.observation.noise import apply_archive_noise
from euclid_polish.sky.observation.observation_simulator import (
    ObservationSimulator,
    ObservationSimulatorConfig,
)


def _blank_hr(side: int = 256) -> Image:
    return Image(
        data=np.zeros((side, side, 4), dtype=np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES,
        is_clean=True,
    )


def test_rotated_noise_region_covers_requested_cutout_fraction():
    scale = draw_noise_scale_map(
        (128, 160),
        np.random.default_rng(8),
        global_scale_min=1.0,
        global_scale_max=1.0,
        region_probability=1.0,
        region_fraction_min=0.25,
        region_fraction_max=0.50,
        region_scale_min=1.2,
        region_scale_max=1.2,
    )
    covered = float(np.mean(scale > 1.0))

    assert scale.shape == (128, 160)
    assert 0.25 <= covered <= 0.50
    np.testing.assert_allclose(np.unique(scale), [1.0, 1.2])


@pytest.mark.parametrize("band", [Config.BAND_VIS, Config.BAND_Y_E])
def test_archive_noise_scale_map_scales_only_noise_residual(band):
    signal = np.full((72, 75), 30.0, dtype=np.float32)
    plain = apply_archive_noise(
        signal, band, np.random.default_rng(12), add_artifacts=False,
    )
    scaled = apply_archive_noise(
        signal,
        band,
        np.random.default_rng(12),
        add_artifacts=False,
        noise_scale_map=np.full(signal.shape, 1.4, dtype=np.float32),
    )

    np.testing.assert_allclose(
        scaled - signal,
        1.4 * (plain - signal),
        rtol=3e-5,
        atol=3e-5,
    )


def test_distant_star_wing_is_drawn_thousands_of_pixels_outside_field():
    wings = draw_distant_star_wings(
        (128, 128),
        np.random.default_rng(4),
        probability=1.0,
        source_distance_min_lr_pix=1000.0,
        source_distance_max_lr_pix=5000.0,
        amplitude_sigma_min=8.0,
        amplitude_sigma_max=30.0,
        width_min_lr_pix=0.8,
        width_max_lr_pix=2.0,
    )

    assert len(wings) == 1
    assert 1000.0 <= wings[0].source_distance_lr_pix <= 5000.0


def test_distant_star_wing_crosses_whole_cutout_without_visible_star():
    wing = DistantStarWing(
        angle_rad=np.pi / 4.0,
        offset_lr_pix=0.0,
        source_distance_lr_pix=3000.0,
        amplitude_sigma=20.0,
        width_lr_pix=1.0,
    )
    image = add_distant_star_wings(
        np.zeros((128, 128), dtype=np.float32),
        (wing,),
        local_sigma_e=10.0,
    )

    assert image.max() == pytest.approx(200.0, rel=0.04)
    assert image[0, 0] > 100.0
    assert image[-1, -1] > 100.0


def test_simulator_adds_off_field_wing_without_local_star_or_hr_target():
    no_detector_artifacts = ArtifactConfig(
        add_cosmic_rays=False,
        add_hot_pixels=False,
        add_dead_pixels=False,
        add_streaks=False,
    )
    simulator = ObservationSimulator(config=ObservationSimulatorConfig(
        add_noise=True,
        add_artifacts=True,
        artifact_config=no_detector_artifacts,
        add_saturation=False,
        randomize_psf=False,
        add_noise_variation=False,
        add_distant_star_wings=True,
        distant_star_wing_probability=1.0,
        distant_star_wing_amplitude_sigma_min=100.0,
        distant_star_wing_amplitude_sigma_max=100.0,
    ))
    hr = _blank_hr(256)

    lr, target = simulator.process(hr, np.random.default_rng(5))

    assert np.count_nonzero(lr.data[..., 0] > 500.0) > 100
    np.testing.assert_array_equal(target.data, hr.data)


@pytest.mark.parametrize("kwargs", [
    {"distant_star_wing_probability": 1.1},
    {"distant_star_wing_source_distance_min_lr_pix": 5000.0,
     "distant_star_wing_source_distance_max_lr_pix": 1000.0},
    {"distant_star_wing_amplitude_sigma_min": 0.0},
    {"distant_star_wing_width_min_lr_pix": 2.0,
     "distant_star_wing_width_max_lr_pix": 1.0},
    {"noise_global_scale_min": 0.0},
    {"noise_region_probability": 1.1},
    {"noise_region_fraction_min": 0.6,
     "noise_region_fraction_max": 0.5},
    {"noise_region_scale_min": 1.2, "noise_region_scale_max": 1.1},
])
def test_invalid_field_variation_config_rejected(kwargs):
    with pytest.raises(ValueError):
        ObservationSimulatorConfig(**kwargs)
