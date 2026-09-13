"""Delivered-mosaic noise: Euclid's MER noise level with dither correlation."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import euclid_polish.sky.observation.noise as noise_module
from euclid_polish.config import Config
from euclid_polish.sky.observation.noise import (
    _shifted_bilinear,
    apply_archive_noise,
    dithered_unit_noise,
)

BANDS = [Config.BAND_VIS, Config.BAND_Y_E, Config.BAND_J_E, Config.BAND_H_E]
BAND_IDS = [band.name for band in BANDS]


def _lag1_correlation(field: np.ndarray) -> float:
    x = field - field.mean()
    variance = float(np.mean(x * x))
    horizontal = float(np.mean(x[:, 1:] * x[:, :-1]))
    vertical = float(np.mean(x[1:, :] * x[:-1, :]))
    return 0.5 * (horizontal + vertical) / variance


@pytest.mark.parametrize("factor", [1, 3])
def test_each_detector_pixel_spreads_total_weight_factor_squared(factor):
    """Resampling conserves per-area noise, so large-area sums keep it."""
    detector = np.zeros((20, 20))
    detector[9, 11] = 1.0
    shape = (12 * factor, 12 * factor)
    for offset_y, offset_x in ((0.0, 0.0), (0.37, 0.81), (0.99, 0.5)):
        out = _shifted_bilinear(detector, factor, offset_y, offset_x, shape, pad=2)
        assert float(out.sum()) == pytest.approx(factor * factor, rel=1e-12)


@pytest.mark.parametrize("band", BANDS, ids=BAND_IDS)
def test_unit_noise_scatter_and_correlation_match_a_dithered_stack(band):
    rng = np.random.default_rng(5)
    fields = [dithered_unit_noise((256, 256), band, rng) for _ in range(8)]
    pixel_sigma = float(np.median([field.std() for field in fields]))
    lag1 = float(np.median([_lag1_correlation(field) for field in fields]))
    if band.name == "VIS":
        # Bilinear weights averaged over sub-pixel offsets keep (2/3)**2 of
        # the variance in one pixel and share the rest with neighbours.
        assert pixel_sigma == pytest.approx(2.0 / 3.0, rel=0.08)
        assert 0.15 < lag1 < 0.35
    else:
        # A 0.30" detector pixel also spreads over 3x3 archive pixels.
        assert pixel_sigma == pytest.approx(2.0 / 9.0, rel=0.10)
        assert 0.75 < lag1 < 0.93


@pytest.mark.parametrize("band", BANDS, ids=BAND_IDS)
def test_unit_noise_has_unit_variance_over_large_areas(band):
    rng = np.random.default_rng(9)
    box = 48
    ratios = []
    for _ in range(24):
        field = dithered_unit_noise((384, 384), band, rng)
        sums = field.reshape(384 // box, box, 384 // box, box).sum(axis=(1, 3))
        ratios.append(float(sums.var()) / (box * box))
    # Finite boxes lose a few percent of correlated variance at their edges.
    assert 0.85 < float(np.mean(ratios)) < 1.10


@pytest.mark.parametrize("band", BANDS, ids=BAND_IDS)
def test_blank_sky_pixel_scatter_matches_real_mosaics(band):
    """Real Q1 mosaics: pixel scatter is ~0.66 of the MER map in VIS, ~0.24 in NISP."""
    noise = apply_archive_noise(
        np.zeros((256, 256), dtype=np.float32), band, np.random.default_rng(21),
    )
    expected = 0.66 if band.name == "VIS" else 0.235
    assert float(noise.std()) / band.mer_rms_e == pytest.approx(expected, rel=0.10)


def test_source_photon_noise_adds_to_the_sky_level():
    band = Config.BAND_VIS
    bright = np.full((64, 64), 4000.0, dtype=np.float32)
    sky = apply_archive_noise(np.zeros_like(bright), band, np.random.default_rng(3))
    source = apply_archive_noise(bright, band, np.random.default_rng(3)) - bright
    factor = np.sqrt(band.mer_rms_e ** 2 + 4000.0) / band.mer_rms_e
    np.testing.assert_allclose(source, sky * factor, rtol=1e-4, atol=2e-3)


def test_sky_noise_does_not_depend_on_sources_elsewhere():
    band = Config.BAND_VIS
    blank = np.zeros((64, 64), dtype=np.float32)
    crowded = blank.copy()
    crowded[:, 32:] = 5000.0
    a = apply_archive_noise(blank, band, np.random.default_rng(8))
    b = apply_archive_noise(crowded, band, np.random.default_rng(8))
    np.testing.assert_array_equal(a[:, :32], b[:, :32])


@pytest.mark.parametrize("band", [Config.BAND_VIS, Config.BAND_Y_E], ids=["VIS", "Y_E"])
def test_depth_map_scales_noise_before_artifacts(monkeypatch, band):
    signal = np.zeros((96, 96), dtype=np.float32)
    scale = np.full(signal.shape, 0.5, dtype=np.float32)
    seen: dict[str, np.ndarray | float] = {}

    def fake_inject(observed, band_arg, rng, config, *, local_sigma_e):
        del band_arg, rng, config
        seen["input"] = np.asarray(observed).copy()
        seen["sigma"] = float(local_sigma_e)
        return np.asarray(observed) + np.float32(7.0)

    monkeypatch.setattr(noise_module, "inject_artifacts", fake_inject)
    plain = apply_archive_noise(signal, band, np.random.default_rng(45))
    actual = apply_archive_noise(
        signal, band, np.random.default_rng(45),
        add_artifacts=True, noise_scale_map=scale,
    )

    np.testing.assert_allclose(seen["input"], 0.5 * plain, atol=1e-5)
    np.testing.assert_allclose(actual, seen["input"] + 7.0, atol=1e-5)
    assert seen["sigma"] == pytest.approx(0.5 * float(plain.std()), rel=1e-5)


def test_archive_noise_keeps_shape_and_is_reproducible():
    signal = np.zeros((32, 35), dtype=np.float32)
    a = apply_archive_noise(signal, Config.BAND_J_E, np.random.default_rng(9))
    b = apply_archive_noise(signal, Config.BAND_J_E, np.random.default_rng(9))
    assert a.shape == signal.shape
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a, b)


def test_archive_noise_requires_a_mer_noise_level():
    band = dataclasses.replace(Config.BAND_VIS, mer_rms_e=0.0)
    with pytest.raises(ValueError, match="mer_rms_e"):
        apply_archive_noise(
            np.zeros((8, 8), dtype=np.float32), band, np.random.default_rng(0),
        )
