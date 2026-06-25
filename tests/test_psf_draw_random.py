# tests/test_psf_draw_random.py
from __future__ import annotations

import numpy as np

from euclid_polish.psf.core import PSF
from euclid_polish.psf.psf_set import PSFSet


def _make_psf(val: float = 1.0) -> PSF:
    data = np.full((5, 5), val / 25.0, dtype=np.float32)
    return PSF(data=data, pixel_scale=0.05)


def _make_set(n: int = 3) -> PSFSet:
    return PSFSet.from_psfs([_make_psf(float(i + 1)) for i in range(n)])


def test_draw_random_returns_psf():
    pset = _make_set()
    rng = np.random.default_rng(42)
    result = pset.draw_random(rng)
    assert isinstance(result, PSF)


def test_draw_random_single_member_is_deterministic():
    pset = PSFSet.from_psfs([_make_psf(1.0)])
    p1 = pset.draw_random(np.random.default_rng(0))
    p2 = pset.draw_random(np.random.default_rng(99))
    np.testing.assert_allclose(p1.data, p2.data)


def test_draw_random_pixel_scale_preserved():
    pset = _make_set()
    p = pset.draw_random(np.random.default_rng(7))
    assert abs(p.pixel_scale - 0.05) < 1e-6


def test_draw_random_valid_psf_over_seeds():
    pset = _make_set(n=5)
    for seed in range(10):
        p = pset.draw_random(np.random.default_rng(seed))
        assert isinstance(p, PSF)
        assert np.isfinite(p.data).all()
