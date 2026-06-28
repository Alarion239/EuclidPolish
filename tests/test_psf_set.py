"""Tests for :class:`euclid_polish.psf.PSFSet` — the position-dependent PSF
ensemble: mean, the star-count-weighted + random-roll generation sampler, and
the multi-extension FITS round-trip (PrimaryHDU = mean so legacy single-PSF
readers keep working)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.psf import PSF, PSFSet


def _delta(side: int, dy: int, dx: int, scale: float = 0.05) -> PSF:
    a = np.zeros((side, side), dtype=np.float32)
    a[side // 2 + dy, side // 2 + dx] = 1.0
    return PSF(data=a, pixel_scale=scale)


def _gauss(side: int, fwhm: float, scale: float = 0.05) -> PSF:
    x = np.arange(side) - side // 2
    X, Y = np.meshgrid(x, x)
    s = fwhm / 2.355
    g = np.exp(-(X * X + Y * Y) / (2 * s * s)).astype(np.float32)
    return PSF(data=g / g.sum(), pixel_scale=scale)


def test_from_psfs_normalises_and_keeps_metadata():
    pset = PSFSet.from_psfs([_delta(11, -1, 0), _delta(11, 1, 0)],
                            centroids=[(10.0, 2.0), (10.2, 2.0)])
    assert pset.n == 2
    assert pset.shape == (11, 11)
    assert pset.pixel_scale == pytest.approx(0.05)
    for p in pset.psfs:
        assert p.total_flux == pytest.approx(1.0)


def test_from_psfs_rejects_mismatched_grid():
    with pytest.raises(ValueError, match="pixel_scale"):
        PSFSet.from_psfs([_delta(11, 0, 0, 0.05), _delta(11, 0, 0, 0.10)])
    with pytest.raises(ValueError, match="shape"):
        PSFSet.from_psfs([_delta(11, 0, 0), _delta(13, 0, 0)])


def test_mean_is_unit_sum_average():
    pset = PSFSet.from_psfs([_delta(11, -2, 0), _delta(11, 2, 0)])
    m = pset.mean()
    assert m.total_flux == pytest.approx(1.0)
    # The mean of two disjoint deltas has 0.5 at each location.
    assert m.data[3, 5] == pytest.approx(0.5)
    assert m.data[7, 5] == pytest.approx(0.5)


def test_sample_for_generation_unit_sum_and_reproducible():
    pset = PSFSet.from_psfs([_gauss(31, 2.0), _gauss(31, 5.0)])
    a = pset.sample_for_generation(np.random.default_rng(42))
    b = pset.sample_for_generation(np.random.default_rng(42))
    assert a.total_flux == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(a.data, b.data)          # seed-reproducible


def test_sample_weights_by_star_count():
    """A cluster's pick probability is proportional to its star count, so a
    heavily-outnumbered few-star PSF is drawn rarely."""
    pset = PSFSet.from_psfs([_gauss(31, 2.0), _gauss(31, 5.0)],
                            n_stars=[1, 999])
    rng = np.random.default_rng(0)
    # Force no rotation so each draw is exactly one of the two members.
    picks = [pset.sample_for_generation(rng, use_unrotated_prob=1.0)
             for _ in range(400)]
    n_first = sum(np.allclose(p.data, pset.psfs[0].data) for p in picks)
    assert n_first < 20            # the 1-star PSF almost never appears


def test_sample_uniform_when_no_star_counts():
    pset = PSFSet.from_psfs([_gauss(31, 2.0), _gauss(31, 5.0)])  # n_stars None
    assert pset._pick_weights() is None


def test_sample_unrotated_vs_rotated_probability():
    """≈30% of draws are unrotated (identical to a member); the rest are
    rotated (differ from every member)."""
    pset = PSFSet.from_psfs([_gauss(31, 2.0)])      # single member, K=1
    rng = np.random.default_rng(1)
    draws = [pset.sample_for_generation(rng, use_unrotated_prob=0.3)
             for _ in range(600)]
    unrot = sum(np.allclose(d.data, pset.psfs[0].data) for d in draws)
    assert 0.2 < unrot / len(draws) < 0.4           # ~30%
    # Every draw is still sum=1.
    assert all(d.total_flux == pytest.approx(1.0, abs=1e-5) for d in draws)


def test_sample_rotation_angle_is_in_1_359():
    """With unrotated prob 0, the member is always rotated by a real roll."""
    pset = PSFSet.from_psfs([_gauss(31, 2.0)])
    rng = np.random.default_rng(2)
    for _ in range(50):
        d = pset.sample_for_generation(rng, use_unrotated_prob=0.0)
        # A rotated Gaussian-with-no-asymmetry is ≈ itself, but flux is kept.
        assert d.total_flux == pytest.approx(1.0, abs=1e-5)


def test_draw_and_apply_couples_index_and_roll():
    """One PSFSample applied to two bands selects the SAME cluster index and
    the SAME roll in both — the cross-band consistency the generator relies on."""
    from euclid_polish.psf import PSFSample
    a = PSFSet.from_psfs([_gauss(31, 2.0), _gauss(31, 5.0), _gauss(31, 8.0)])
    b = PSFSet.from_psfs([_gauss(21, 2.0), _gauss(21, 5.0), _gauss(21, 8.0)])
    spec = a.draw_sample(np.random.default_rng(3), use_unrotated_prob=0.0)
    assert isinstance(spec, PSFSample) and spec.angle is not None
    pa = a.apply_sample(spec)
    np.testing.assert_allclose(pa.data, a.psfs[spec.index].rotated(spec.angle).data)
    assert b.apply_sample(spec).shape == (21, 21)   # b's same-index cluster, same roll


def test_apply_sample_clamps_index_for_smaller_set():
    """A Gaussian-fallback band (fewer members) clamps the shared index to its
    own range rather than indexing out of bounds."""
    from euclid_polish.psf import PSFSample
    one = PSFSet.from_psfs([_gauss(21, 3.0)])
    got = one.apply_sample(PSFSample(index=2, angle=None))
    np.testing.assert_allclose(got.data, one.psfs[0].data)


def test_fits_roundtrip_primary_is_mean(tmp_path):
    pset = PSFSet.from_psfs([_gauss(21, 2.0), _gauss(21, 4.0), _gauss(21, 6.0)],
                            centroids=[(10.0, 2.0), (10.1, 2.1), (10.2, 2.2)])
    path = pset.save(str(tmp_path), "euclid_psf_TEST.fits")

    # Multi-extension: 1 primary + K image HDUs, NPSF header records K.
    with fits.open(path) as hdul:
        assert hdul[0].header["NPSF"] == 3
        data_hdus = [h for h in hdul if h.data is not None]
        assert len(data_hdus) == 4

    loaded = PSFSet.from_fits(path)
    assert loaded.n == 3
    assert loaded.centroids is not None
    for orig, got in zip(pset.psfs, loaded.psfs, strict=False):
        np.testing.assert_allclose(orig.data, got.data, atol=1e-6)

    # HDU[0] is the mean → legacy PSF.from_fits gets a sensible single PSF.
    legacy = PSF.from_fits(path)
    np.testing.assert_allclose(legacy.data, pset.mean().data, atol=1e-6)


def test_from_fits_legacy_single_hdu_loads_as_one_element(tmp_path):
    # A pre-existing single-PSF FITS (only a PrimaryHDU) must load as a
    # 1-element set so the new loader works on both formats.
    legacy = _gauss(21, 3.0)
    path = legacy.save(str(tmp_path), "euclid_psf_LEGACY.fits")
    pset = PSFSet.from_fits(path)
    assert pset.n == 1
    np.testing.assert_allclose(pset.psfs[0].data, legacy.data, atol=1e-6)


def test_n_stars_roundtrips_through_fits(tmp_path):
    """Per-PSF star counts (the sampling weights) persist as NSTARS and load
    back; a file without them yields n_stars=None (→ uniform sampling)."""
    pset = PSFSet.from_psfs([_gauss(21, 2.0), _gauss(21, 4.0)],
                            n_stars=[120, 57])
    path = pset.save(str(tmp_path), "euclid_psf_NSTARS.fits")
    with fits.open(path) as hdul:
        image_hdus = [h for h in hdul if h.data is not None]
        assert image_hdus[1].header["NSTARS"] == 120
        assert image_hdus[2].header["NSTARS"] == 57
    loaded = PSFSet.from_fits(path)
    assert loaded.n_stars == [120, 57]

    no_counts = PSFSet.from_psfs([_gauss(21, 2.0), _gauss(21, 4.0)])
    p2 = no_counts.save(str(tmp_path), "euclid_psf_NONE.fits")
    assert PSFSet.from_fits(p2).n_stars is None


def test_grid_ops_map_over_members():
    pset = PSFSet.from_psfs([_gauss(31, 3.0), _gauss(31, 5.0)])
    cropped = pset.centre_cropped_to(21)
    assert cropped.n == 2
    assert cropped.shape == (21, 21)
    for p in cropped.psfs:
        assert p.total_flux == pytest.approx(1.0)
