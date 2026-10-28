"""Delivered-mosaic noise: Euclid's MER noise level with the texture of a
dithered, bilinearly resampled, median-combined stack."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import euclid_polish.sky.observation.noise as noise_module
from euclid_polish.config import Config
from euclid_polish.sky.observation.mer_noise_levels import load_mer_noise_levels
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


def _lag_correlations(field: np.ndarray) -> np.ndarray:
    """Lag-1 (axis mean), diagonal and lag-2 (axis mean) correlation."""
    x = field - field.mean()
    variance = float(np.mean(x * x))

    def lag(dy, dx):
        return float(np.mean(x[: x.shape[0] - dy, : x.shape[1] - dx] * x[dy:, dx:])) / variance

    return np.array([(lag(0, 1) + lag(1, 0)) / 2, lag(1, 1), (lag(0, 2) + lag(2, 0)) / 2])


@pytest.mark.parametrize("band", BANDS, ids=BAND_IDS)
def test_unit_noise_scatter_and_correlation_match_a_median_stack(band):
    rng = np.random.default_rng(5)
    fields = [dithered_unit_noise((256, 256), band, rng) for _ in range(12)]
    pixel_sigma = float(np.median([field.std() for field in fields]))
    lag1 = float(np.mean([_lag1_correlation(field) for field in fields]))
    if band.name == "VIS":
        # Bilinear weights share a third of each pixel's noise with its
        # neighbours; the median of four exposures keeps less of it than a
        # mean (lag-1 0.26).
        assert pixel_sigma == pytest.approx(0.69, rel=0.06)
        assert 0.20 < lag1 < 0.255
    else:
        # A 0.30" detector pixel also spreads over 3x3 archive pixels; a mean
        # of the exposures would keep lag-1 0.86.
        assert pixel_sigma == pytest.approx(0.23, rel=0.06)
        assert 0.80 < lag1 < 0.84


def test_stack_is_the_median_of_its_exposures(monkeypatch):
    """One exposure with a wild value cannot move a median of four."""
    band = Config.BAND_Y_E
    real = noise_module._resampled_exposures

    def with_outlier(shape, factor, n_exposures, rng):
        layers = real(shape, factor, n_exposures, rng)
        layers[0, 10, 10] = 1.0e6
        return layers

    clean = dithered_unit_noise((32, 32), band, np.random.default_rng(1))
    monkeypatch.setattr(noise_module, "_resampled_exposures", with_outlier)
    hit = dithered_unit_noise((32, 32), band, np.random.default_rng(1))
    assert abs(float(hit[10, 10])) < 2.0
    np.testing.assert_array_equal(np.delete(hit.ravel(), 10 * 32 + 10),
                                  np.delete(clean.ravel(), 10 * 32 + 10))


def test_median_scale_is_one_up_to_two_exposures_and_above_one_beyond():
    for factor in (1, 3):
        assert noise_module._median_stack_scale(factor, 1) == 1.0
        assert noise_module._median_stack_scale(factor, 2) == 1.0
        assert 1.0 < noise_module._median_stack_scale(factor, 4) < 1.15


def test_exposure_count_follows_the_scenes_noise_level():
    band = Config.BAND_J_E
    n = band.n_exposures
    assert noise_module.coverage_exposures(band, None) == n
    assert noise_module.coverage_exposures(band, band.mer_rms_e) == n
    assert noise_module.coverage_exposures(band, band.mer_rms_e / np.sqrt(2.0)) == 2 * n
    assert noise_module.coverage_exposures(band, band.mer_rms_e * 10.0) == 2       # clipped
    assert noise_module.coverage_exposures(band, band.mer_rms_e / 10.0) == 12


def test_archive_noise_stacks_the_exposures_its_level_implies(monkeypatch):
    band = Config.BAND_H_E
    seen = []
    real = noise_module.dithered_unit_noise

    def spy(shape, band_arg, rng, n_exposures=None):
        seen.append(n_exposures)
        return real(shape, band_arg, rng, n_exposures=n_exposures)

    monkeypatch.setattr(noise_module, "dithered_unit_noise", spy)
    signal = np.zeros((16, 16), dtype=np.float32)
    apply_archive_noise(signal, band, np.random.default_rng(0))
    apply_archive_noise(signal, band, np.random.default_rng(0), sky_rms_e=band.mer_rms_e / np.sqrt(2.0))
    assert seen == [band.n_exposures, 2 * band.n_exposures]


@pytest.mark.parametrize("band", [Config.BAND_VIS, Config.BAND_Y_E], ids=["VIS", "Y_E"])
def test_texture_over_the_q1_levels_matches_real_mer_sky(band):
    """Empty sky of 44 Q1 MER cutouts (2026-10-28): lag-1, diagonal and lag-2
    correlation VIS 0.205 / 0.051 / 0.00, NISP 0.80 / 0.67 / 0.50."""
    real = {"VIS": (0.205, 0.051, 0.0), "Y_E": (0.80, 0.673, 0.497)}[band.name]
    levels = load_mer_noise_levels()
    column = levels.levels_e[:, levels.bands.index(band.name)]
    rng = np.random.default_rng(17)
    signal = np.zeros((160, 160), dtype=np.float32)
    draws = [apply_archive_noise(signal, band, rng, sky_rms_e=float(level))
             for level in rng.choice(column, 60)]
    measured = np.mean([_lag_correlations(field) for field in draws], axis=0)
    np.testing.assert_allclose(measured, real, atol=0.03)


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
    """Real Q1 mosaics: pixel scatter is 0.666 of the MER map in VIS, 0.23 in NISP."""
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
