"""Unit tests for euclid_polish.sky.apparent_size.ApparentSizeModel.

Network-free; the only heavy dependency is the Planck15 d_A spline built once
at construction. Checks the literature-tuned defaults reproduce deep-field
medians and a continuous, rare big tail.
"""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.sky.apparent_size import ApparentSizeModel


def test_scalar_and_array_shapes():
    m = ApparentSizeModel()
    rng = np.random.default_rng(0)
    s = m.sample(rng)
    assert isinstance(s, float)
    a = m.sample(rng, size=10)
    assert isinstance(a, np.ndarray) and a.shape == (10,)


def test_determinism():
    m = ApparentSizeModel()
    a = m.sample(np.random.default_rng(42), size=1000)
    b = m.sample(np.random.default_rng(42), size=1000)
    assert np.array_equal(a, b)


def test_clip_bounds_respected():
    m = ApparentSizeModel(theta_min_arcsec=0.05, theta_max_arcsec=5.0)
    a = m.sample(np.random.default_rng(1), size=20000)
    assert a.min() >= 0.05 - 1e-9
    assert a.max() <= 5.0 + 1e-9


def test_median_matches_deep_field():
    # Literature anchor: median apparent R_e ≈ 0.2–0.25″ for a mag<25 sample.
    m = ApparentSizeModel()
    a = m.sample(np.random.default_rng(7), size=200000)
    assert 0.19 < np.median(a) < 0.27


def test_continuous_rare_big_tail():
    # Big galaxies present but rare, on a continuous tail (no spike): a few %
    # above 1″, well under 1% above 3″.
    m = ApparentSizeModel()
    a = m.sample(np.random.default_rng(3), size=400000)
    f1 = np.mean(a > 1.0)
    f3 = np.mean(a > 3.0)
    assert 0.005 < f1 < 0.08          # ~2% expected
    assert f3 < 0.01                  # ~0.1% expected
    assert f1 > f3                     # monotone decline


def test_bigger_r0_shifts_distribution_up():
    rng = lambda: np.random.default_rng(11)
    small = ApparentSizeModel(r0_kpc=1.5).sample(rng(), size=50000)
    big = ApparentSizeModel(r0_kpc=4.0).sample(rng(), size=50000)
    assert np.median(big) > np.median(small)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        ApparentSizeModel(r0_kpc=0.0)
    with pytest.raises(ValueError):
        ApparentSizeModel(theta_min_arcsec=2.0, theta_max_arcsec=1.0)
