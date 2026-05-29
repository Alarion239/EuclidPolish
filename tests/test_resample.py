"""Tests for the Lanczos-3 / cubic resampler."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.sky.resample import (
    cubic_upsample,
    lanczos3_upsample,
    upsample,
)


def test_lanczos3_output_shape_for_factor3():
    arr = np.zeros((8, 11), dtype=np.float32)
    out = lanczos3_upsample(arr, factor=3)
    assert out.shape == (24, 33)


def test_factor1_returns_copy():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = lanczos3_upsample(arr, factor=1)
    np.testing.assert_array_equal(arr, out)
    assert out is not arr  # copy, not view


def test_constant_image_is_preserved():
    """A flat image should remain flat after upsampling (sum scales by factor²)."""
    arr = np.full((10, 10), 5.0, dtype=np.float32)
    for kernel in ("lanczos3", "cubic"):
        out = upsample(arr, factor=3, kernel=kernel)
        # Mean preserved within numerical noise (modulo boundary effects).
        # Test on the central crop to avoid boundary regions.
        assert np.allclose(out[3:-3, 3:-3], 5.0, atol=1e-3), kernel


def test_lanczos3_partition_of_unity_interior():
    """Sum of weights along the kernel = 1 → mean preserved in the interior."""
    arr = np.zeros((20, 20), dtype=np.float64)
    arr[10, 10] = 1.0
    out = lanczos3_upsample(arr, factor=3)
    # Integral over output ≈ factor² × input integral (3² = 9).
    # Equivalent to: sum is preserved per axis × per axis = factor² for a delta.
    assert out.sum() == pytest.approx(9.0, rel=1e-3)


def test_lanczos3_central_pixel_value_of_delta():
    """A delta upsampled by 3× should peak at the input pixel's new position."""
    arr = np.zeros((9, 9), dtype=np.float64)
    arr[4, 4] = 1.0
    out = lanczos3_upsample(arr, factor=3)
    peak_yx = np.unravel_index(out.argmax(), out.shape)
    # Output pixel (j_out + 0.5) / 3 - 0.5 ≈ input 4 → j_out ≈ 13.5
    # i.e. peak between 13 and 14 in output.
    assert peak_yx[0] in (13, 14)
    assert peak_yx[1] in (13, 14)


def test_upsample_dispatch():
    arr = np.ones((6, 6), dtype=np.float32)
    out_l = upsample(arr, factor=3, kernel="lanczos3")
    out_c = upsample(arr, factor=3, kernel="cubic")
    assert out_l.shape == (18, 18)
    assert out_c.shape == (18, 18)
    with pytest.raises(ValueError):
        upsample(arr, factor=3, kernel="bogus")


def test_factor_must_be_positive():
    with pytest.raises(ValueError):
        lanczos3_upsample(np.zeros((4, 4)), factor=0)
    with pytest.raises(ValueError):
        cubic_upsample(np.zeros((4, 4)), factor=-1)


def test_conserve_flux_splits_total_across_finer_grid():
    """Value-preserving upsample inflates the total by ~factor²; conserve_flux
    splits flux so the total is (approximately) preserved — the behaviour the
    forward model needs for the NISP 0.30″→0.10″ electron-count resample."""
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    arr = gaussian_filter(np.abs(rng.normal(size=(30, 30))), sigma=3.0).astype(np.float64)
    total = arr.sum()
    val_preserving = upsample(arr, factor=3, kernel="lanczos3")
    flux_conserving = upsample(arr, factor=3, kernel="lanczos3", conserve_flux=True)
    # Value-preserving total grows ~factor² = 9×.
    assert val_preserving.sum() == pytest.approx(9 * total, rel=0.05)
    # Flux-conserving total is preserved to ~%.
    assert flux_conserving.sum() == pytest.approx(total, rel=0.02)
    # factor==1 is a no-op even with conserve_flux.
    same = upsample(arr, factor=1, kernel="lanczos3", conserve_flux=True)
    np.testing.assert_array_equal(same, arr)


def test_lanczos3_smooth_image_close_to_cubic():
    """For a slowly-varying image both kernels give nearly the same result."""
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(20, 20)).astype(np.float64)
    from scipy.ndimage import gaussian_filter
    arr_smooth = gaussian_filter(arr, sigma=3.0)
    out_l = lanczos3_upsample(arr_smooth, factor=3)
    out_c = cubic_upsample(arr_smooth, factor=3)
    # Difference on interior should be small (typically ≤ 0.1 of std).
    diff = (out_l[5:-5, 5:-5] - out_c[5:-5, 5:-5])
    assert diff.std() < 0.2 * arr_smooth.std()
