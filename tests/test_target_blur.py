"""Target Gaussian PSF conversion and implementation checks."""

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.training.augmentation import blur_target_tf
from euclid_polish.training.target_blur import (
    TARGET_GAUSSIAN_RADIUS_SIGMA,
    blur_target_array,
    target_kernel_radius_pixels,
    target_sigma_pixels,
)


def test_default_fwhm_is_point_sixty_six_arcsec():
    assert Config.TARGET_PSF_FWHM_ARCSEC == 0.66
    assert target_sigma_pixels() == pytest.approx(
        0.66 / Config.DEFAULT_PIXEL_SCALE / 2.354820045,
    )


def test_gaussian_kernel_radius_is_seven_sigma():
    assert TARGET_GAUSSIAN_RADIUS_SIGMA == 7.0
    assert target_kernel_radius_pixels() == 39
    assert target_kernel_radius_pixels(
        Config.BAND_VIS.psf_fwhm_arcsec,
    ) == 10


def test_numpy_blur_preserves_flux_and_does_not_mix_bands():
    source = np.zeros((101, 101, 2), np.float32)
    source[50, 50, 0] = 10.0
    source[50, 50, 1] = 20.0
    blurred = blur_target_array(source)
    assert blurred.shape == source.shape
    assert blurred.dtype == np.float32
    assert blurred[50, 50, 0] < source[50, 50, 0]
    assert blurred[50, 50, 1] == pytest.approx(2 * blurred[50, 50, 0], rel=1e-6)
    assert blurred[..., 0].sum() == pytest.approx(10.0, rel=1e-6)
    assert blurred[..., 1].sum() == pytest.approx(20.0, rel=1e-6)


def test_zero_fwhm_is_identity_copy():
    source = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    result = blur_target_array(source, 0.0)
    np.testing.assert_array_equal(result, source)
    assert result is not source


def test_tensorflow_and_numpy_blur_match():
    rng = np.random.default_rng(4)
    source = rng.random((101, 101, 4), dtype=np.float32)
    expected = blur_target_array(source, Config.TARGET_PSF_FWHM_ARCSEC)
    actual = blur_target_tf(tf.constant(source), Config.TARGET_PSF_FWHM_ARCSEC)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_fwhm_rejected(value):
    with pytest.raises(ValueError):
        blur_target_array(np.zeros((3, 3), np.float32), value)
