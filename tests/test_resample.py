"""Tests for the bilinear / cubic resampler."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.sky.observation.resample import (
    bilinear_upsample,
    cubic_upsample,
    upsample,
)


def test_bilinear_output_shape_and_dtype_for_factor3():
    arr = np.zeros((8, 11), dtype=np.float32)
    out = bilinear_upsample(arr, factor=3)
    assert out.shape == (24, 33)
    assert out.dtype == arr.dtype


@pytest.mark.parametrize("resampler", [bilinear_upsample, cubic_upsample])
def test_factor1_returns_copy(resampler):
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = resampler(arr, factor=1)
    np.testing.assert_array_equal(arr, out)
    assert out is not arr


def test_constant_image_is_preserved_through_edges():
    """A flat image must remain flat over the full resampled footprint."""
    arr = np.full((10, 10), 5.0, dtype=np.float32)
    for kernel in ("bilinear", "cubic"):
        out = upsample(arr, factor=3, kernel=kernel)
        np.testing.assert_allclose(out, 5.0, atol=1e-5, err_msg=kernel)


def test_bilinear_uses_pixel_centred_coordinates():
    arr = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    expected = np.array([
        [0.0, 0.25, 0.75, 1.0],
        [0.5, 0.75, 1.25, 1.5],
        [1.5, 1.75, 2.25, 2.5],
        [2.0, 2.25, 2.75, 3.0],
    ], dtype=np.float32)
    np.testing.assert_allclose(bilinear_upsample(arr, factor=2), expected)


def test_bilinear_delta_is_nonnegative_and_preserves_integral_scale():
    arr = np.zeros((20, 20), dtype=np.float64)
    arr[10, 10] = 1.0
    out = bilinear_upsample(arr, factor=3)

    assert np.all(out >= 0.0)
    assert out.sum() == pytest.approx(9.0)


def test_upsample_dispatch_defaults_to_bilinear():
    arr = np.arange(36, dtype=np.float32).reshape(6, 6)
    expected = bilinear_upsample(arr, factor=3)
    np.testing.assert_array_equal(upsample(arr, factor=3), expected)
    np.testing.assert_array_equal(
        upsample(arr, factor=3, kernel="bilinear"), expected,
    )
    assert upsample(arr, factor=3, kernel="cubic").shape == (18, 18)

    with pytest.raises(ValueError):
        upsample(arr, factor=3, kernel="lanczos3")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        upsample(arr, factor=3, kernel="bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize("resampler", [bilinear_upsample, cubic_upsample])
def test_factor_must_be_positive(resampler):
    with pytest.raises(ValueError):
        resampler(np.zeros((4, 4)), factor=0)


def test_input_must_be_two_dimensional():
    with pytest.raises(ValueError, match="must be 2-D"):
        bilinear_upsample(np.zeros((4, 4, 1)), factor=3)


def test_bilinear_smooth_image_is_close_to_cubic():
    """For a slowly varying image both kernels should nearly agree."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    arr = rng.normal(size=(20, 20)).astype(np.float64)
    arr_smooth = gaussian_filter(arr, sigma=3.0)
    out_b = bilinear_upsample(arr_smooth, factor=3)
    out_c = cubic_upsample(arr_smooth, factor=3)
    diff = out_b[5:-5, 5:-5] - out_c[5:-5, 5:-5]
    assert diff.std() < 0.2 * arr_smooth.std()
