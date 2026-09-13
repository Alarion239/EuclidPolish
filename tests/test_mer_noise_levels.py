"""Measured Q1 sky noise levels and their per-scene draw."""

from __future__ import annotations

import numpy as np
import pytest

import euclid_polish.sky.observation.observation_simulator as observation_module
from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.sky.observation.mer_noise_levels import (
    MERNoiseLevels,
    load_mer_noise_levels,
)
from euclid_polish.sky.observation.noise import apply_archive_noise
from euclid_polish.sky.observation.observation_simulator import (
    ObservationSimulator,
    ObservationSimulatorConfig,
)

BANDS = ("VIS", "Y_E", "J_E", "H_E")


def _is_table_row(levels: MERNoiseLevels, values: np.ndarray) -> bool:
    return bool(np.any(np.all(np.isclose(levels.levels_e, values), axis=1)))


def test_committed_table_covers_the_three_q1_fields_in_all_bands():
    levels = load_mer_noise_levels()
    assert levels.bands == BANDS
    assert len(levels.levels_e) >= 100
    assert set(levels.fields) == {"EDF-N", "EDF-S", "EDF-F"}


def test_band_constants_are_the_table_medians():
    levels = load_mer_noise_levels()
    for name in BANDS:
        assert Config.get_band(name).mer_rms_e == pytest.approx(
            levels.median(name), rel=0.01,
        )


def test_draw_returns_one_real_position_for_all_bands():
    levels = load_mer_noise_levels()
    drawn = levels.draw(np.random.default_rng(4))
    assert _is_table_row(levels, np.array([drawn[name] for name in BANDS]))


def test_table_rejects_nonpositive_levels():
    with pytest.raises(ValueError, match="positive"):
        MERNoiseLevels(
            bands=BANDS,
            levels_e=np.array([[1.0, 2.0, 0.0, 3.0]]),
            fields=("EDF-N",),
        )


def test_sky_level_override_scales_blank_sky_noise_exactly():
    blank = np.zeros((48, 48), dtype=np.float32)
    band = Config.BAND_Y_E
    base = apply_archive_noise(blank, band, np.random.default_rng(6))
    doubled = apply_archive_noise(
        blank, band, np.random.default_rng(6), sky_rms_e=2.0 * band.mer_rms_e,
    )
    np.testing.assert_allclose(doubled, 2.0 * base, rtol=1e-5, atol=1e-6)


def _captured_sky_levels(monkeypatch, **config_kwargs) -> dict:
    seen: dict = {}

    def fake_noise(signal, band, rng, **kwargs):
        del rng
        seen[band.name] = kwargs["sky_rms_e"]
        return signal

    monkeypatch.setattr(observation_module, "apply_archive_noise", fake_noise)
    simulator = ObservationSimulator(config=ObservationSimulatorConfig(
        add_artifacts=False,
        add_saturation=False,
        randomize_psf=False,
        add_distant_star_wings=False,
        **config_kwargs,
    ))
    blank = Image(
        data=np.zeros((32, 32, 4), dtype=np.float32),
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=Config.LR_INPUT_BAND_NAMES,
        is_clean=True,
    )
    simulator.process(blank, rng=np.random.default_rng(11))
    return seen


def test_simulator_gives_every_band_the_same_drawn_position(monkeypatch):
    seen = _captured_sky_levels(monkeypatch)
    levels = load_mer_noise_levels()
    assert _is_table_row(levels, np.array([seen[name] for name in BANDS]))


def test_simulator_can_use_band_medians_instead(monkeypatch):
    seen = _captured_sky_levels(monkeypatch, draw_mer_noise_levels=False)
    assert all(seen[name] is None for name in BANDS)
