"""The Fourier-ring-correlation resolution against JWST
(eval/jwst_resolution.py): identical images agree at every period, a known
shift decays the coherence as cos(2π k·δ), a common blur leaves it at 1
while the gain shows the smoothing, registration recovers a known shift
from the Euclid-like input, and windows without full JWST coverage or with
a saturated star are left out."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as nd_shift

from euclid_polish.eval import jwst_resolution as jr


def _sky(seed=0, n=512, alpha=-2.5):
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n)
    kr = np.hypot(*np.meshgrid(k, k, indexing="ij"))
    kr[0, 0] = 1.0
    field = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, n))) * kr ** (alpha / 2)))
    return 10.0 * field / field.std() + 20.0


def _accumulate(s, j):
    acc = jr.RingCorrelation(lambda x: x)
    acc.add(s, j)
    return jr.summary(acc)


def test_identical_images_agree_at_every_period():
    sky = _sky()
    out = _accumulate(sky, sky)
    assert out["windows"] == 9
    assert np.allclose(out["coherence"], 1.0) and np.allclose(out["gain"], 1.0)
    assert out["resolution_coherence"] is None and out["resolution_gain"] is None
    assert out["period_px"][0] == 128.0 and len(out["period_px"]) == 63


def test_a_shift_decays_the_coherence_like_a_cosine_and_a_blur_only_the_gain():
    sky = _sky(1)
    moved = nd_shift(sky, (0.0, 1.0), order=3, mode="wrap")
    out = _accumulate(moved, sky)
    period = np.array(out["period_px"])
    coherence = np.array(out["coherence"])
    # a pure x shift by δ = 1 px: the ring mean of cos(2π k·δ) is J0(2π k δ),
    # which falls to ½ at 2π k δ = 1.53, a 4.1-px period
    assert out["resolution_coherence"]["px"] == pytest.approx(4.1, abs=0.3)
    assert coherence[period == 128.0][0] > 0.99
    # A Gaussian blur of σ = 1 px keeps the structure (coherence ~1) and its
    # gain is the blur's MTF exp(-2π²σ²k²): ½ at k = sqrt(ln 2 / 2π²σ²), 5.34 px.
    blur = _accumulate(gaussian_filter(sky, 1.0, mode="wrap"), sky)
    assert min(blur["coherence"]) > 0.97
    assert blur["resolution_gain"]["px"] == pytest.approx(
        1.0 / np.sqrt(np.log(2.0) / (2 * np.pi ** 2)), abs=0.1)


def test_half_crossing_interpolates_in_log_period():
    period = np.array([8.0, 4.0, 2.0])
    assert jr.half_crossing(period, np.array([0.9, 0.7, 0.3])) == pytest.approx(
        np.exp(np.log(4.0) + 0.5 * (np.log(2.0) - np.log(4.0))))
    assert jr.half_crossing(period, np.array([0.9, 0.8, 0.6])) is None
    assert jr.half_crossing(period, np.array([0.4, 0.8, 0.6])) is None


def test_registration_recovers_a_known_shift_from_the_euclid_like_input():
    jwst = _sky(2, n=510)
    offset = (0.6, -0.4)
    # the Euclid-like input: the sky moved by ``offset``, blurred to ~0.45″, 2×2 binned
    euclid = gaussian_filter(nd_shift(jwst, offset, order=3, mode="wrap"), 3.8)
    lr = euclid.reshape(255, 2, 255, 2).sum(axis=(1, 3))
    shift = jr.registration_shift(lr, jwst)          # what JWST must move by
    assert shift == pytest.approx(offset, abs=0.15)
    assert jr.registration_shift(lr, np.full_like(jwst, np.nan)) is None


def test_windows_without_full_coverage_or_with_a_saturated_star_are_left_out():
    sky = _sky(3, n=510)
    jwst = sky.copy()
    jwst[:100, :100] = np.nan                                       # no coverage: window (0, 0)
    jwst[300, 300] = 2 * jr.STAR_CUT_E                              # a saturated star: window (1, 1)
    acc = jr.RingCorrelation(lambda x: x)
    assert acc.add(sky, jwst) == 7
    assert acc.add(sky, np.full_like(sky, np.nan)) == 0 and acc.windows == 7


def test_thin_coverage_gaps_share_one_apodized_mask():
    """A window with a thin gap (the early-release detector gaps) counts
    when ``min_coverage`` allows it; identical images still agree at every
    period, and a fully covered window comes out exactly as before."""
    sky = _sky(4, n=510)
    gapped = sky.copy()
    gapped[:, 100:104] = np.nan                                     # a 4-px gap through 3 windows
    strict = jr.RingCorrelation(lambda x: x)
    assert strict.add(sky, gapped) == 6
    loose = jr.RingCorrelation(lambda x: x, min_coverage=0.95)
    assert loose.add(sky, gapped) == 9
    out = jr.summary(loose)
    assert min(out["coherence"]) > 0.999
    full_strict, full_loose = jr.RingCorrelation(lambda x: x), jr.RingCorrelation(lambda x: x, 0.95)
    full_strict.add(sky, sky)
    full_loose.add(sky, sky)
    assert np.array_equal(full_strict.sj, full_loose.sj) and np.array_equal(full_strict.jj, full_loose.jj)


def test_noise_correction_scales_jwst_white_noise_per_ring():
    # JWST = SR + white noise: the raw coherence drops where the noise rises,
    # the corrected one (noise per pixel × each ring's pixel count) stays ~1.
    sky = _sky(3)
    noisy = sky + np.random.default_rng(4).normal(scale=2.0, size=sky.shape)
    out = _accumulate(sky, noisy)
    period = np.array(out["period_px"])
    mid = (period >= 4.0) & (period <= 13.0)
    raw = np.array(out["coherence"])[mid]
    corrected = np.array(out["coherence_noise_corrected"])[mid]
    assert raw.min() < 0.9
    # a ring-sum noise subtracted from every ring overshot this to ~2 (and 1e19 at σ = 5)
    assert corrected.min() > 0.95 and corrected.max() < 1.15
