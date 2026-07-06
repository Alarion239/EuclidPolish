"""Pre-rotated PSF pools: build determinism, cross-band angle alignment,
NSTARS weight inheritance, and the per-training cluster bagging loader."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.psf.core import PSF
from euclid_polish.psf.psf_set import PSFSet
from euclid_polish.psf.rotpool import (
    build_rotation_pool,
    draw_angle_table,
    load_all_band_rotpools,
    load_band_rotpool,
    rotpool_path,
)

PIX = float(Config.DEFAULT_PIXEL_SCALE)
N_CLUSTERS = 6
N_ROT = 2


def _asym_kernel(seed: int, n: int = 21) -> np.ndarray:
    """A clearly non-rotation-symmetric kernel (rotation must change it)."""
    y, x = np.mgrid[:n, :n] - n // 2
    k = np.exp(-(x**2 / 4.0 + y**2 / 18.0))          # elongated
    k[n // 2, n // 2 + 3] += 0.5 * (1 + seed)        # off-centre spike
    return (k / k.sum()).astype(np.float32)


@pytest.fixture(scope="module")
def pool_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("psfdir"))
    sets = {}
    for b, band in enumerate(Config.BANDS[:2]):      # VIS + Y_E is enough
        psfs = [PSF(data=_asym_kernel(100 * b + i), pixel_scale=PIX)
                for i in range(N_CLUSTERS)]
        sets[band.name] = PSFSet.from_psfs(
            psfs, n_stars=[5 * (i + 1) for i in range(N_CLUSTERS)])
    paths = build_rotation_pool(sets, psf_dir=d, rotations=N_ROT, seed=3,
                                workers=1)
    return d, sets, paths


def test_pool_files_and_structure(pool_dir):
    d, sets, paths = pool_dir
    assert set(paths) == {Config.BANDS[0].name, Config.BANDS[1].name}
    for band in Config.BANDS[:2]:
        with fits.open(rotpool_path(band, d)) as hdul:
            assert hdul[0].header["ROTPOOL"]
            assert hdul[0].header["NSRCCLU"] == N_CLUSTERS
            assert hdul[0].header["NROT"] == N_ROT
            assert len(hdul) == 1 + N_CLUSTERS * (N_ROT + 1)


def test_angles_shared_across_bands_and_unrotated_kept(pool_dir):
    d, sets, _ = pool_dir
    per_band = []
    for band in Config.BANDS[:2]:
        with fits.open(rotpool_path(band, d)) as hdul:
            per_band.append([(h.header["SRCIDX"], h.header["ROLLDEG"])
                             for h in hdul[1:]])
    assert per_band[0] == per_band[1]                # one pointing, all bands
    # slot 0 of each cluster is the exact unrotated (unit-sum) source kernel
    band0 = Config.BANDS[0]
    with fits.open(rotpool_path(band0, d)) as hdul:
        for i in range(N_CLUSTERS):
            h = hdul[1 + i * (N_ROT + 1)]
            assert h.header["ROLLDEG"] == 0.0
            np.testing.assert_allclose(
                h.data, sets[band0.name].psfs[i].with_unit_sum().data,
                rtol=1e-6)


def test_rotated_slot_matches_direct_rotation(pool_dir):
    d, sets, _ = pool_dir
    band0 = Config.BANDS[0]
    table = draw_angle_table(N_CLUSTERS, N_ROT, seed=3)
    with fits.open(rotpool_path(band0, d)) as hdul:
        h = hdul[1 + 0 * (N_ROT + 1) + 1]            # cluster 0, first roll
        assert h.header["ROLLDEG"] == pytest.approx(table[0, 0])
        expect = (sets[band0.name].psfs[0].with_unit_sum()
                  .rotated(float(table[0, 0]), order=3).with_unit_sum().data)
        np.testing.assert_allclose(h.data, expect, rtol=1e-5, atol=1e-8)


def test_loader_full_pool_carries_weights(pool_dir):
    d, _, _ = pool_dir
    pset = load_band_rotpool(Config.BANDS[0], psf_dir=d)
    assert pset.n == N_CLUSTERS * (N_ROT + 1)
    # every rotation slot inherits its source cluster's star count
    assert pset.n_stars[:  N_ROT + 1] == [5] * (N_ROT + 1)
    assert pset.n_stars[-(N_ROT + 1):] == [5 * N_CLUSTERS] * (N_ROT + 1)


def test_loader_bagging_is_seeded_and_cluster_level(pool_dir):
    d, _, _ = pool_dir
    a = load_band_rotpool(Config.BANDS[0], psf_dir=d,
                          subset_clusters=3, subset_seed=11)
    b = load_band_rotpool(Config.BANDS[0], psf_dir=d,
                          subset_clusters=3, subset_seed=11)
    assert a.n == b.n == 3 * (N_ROT + 1)
    for pa, pb in zip(a.psfs, b.psfs):
        np.testing.assert_array_equal(pa.data, pb.data)   # deterministic
    # a different seed picks a different cluster subset (6C3=20 — try a few)
    diff = any(
        load_band_rotpool(Config.BANDS[0], psf_dir=d, subset_clusters=3,
                          subset_seed=s).psfs[0].data.tobytes()
        != a.psfs[0].data.tobytes()
        for s in (12, 13, 14, 15))
    assert diff


def test_loader_crop_to_renormalises(pool_dir):
    d, _, _ = pool_dir
    pset = load_band_rotpool(Config.BANDS[0], psf_dir=d, crop_to=11)
    assert all(p.shape == (11, 11) for p in pset.psfs)
    for p in pset.psfs:
        assert float(p.data.sum()) == pytest.approx(1.0, rel=1e-5)


def test_load_all_requires_every_band(pool_dir):
    d, _, _ = pool_dir
    # only 2 of the 4 band pools exist → all-band loader declines
    assert load_all_band_rotpools(psf_dir=d) is None


def test_missing_pool_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="rotation pool"):
        load_band_rotpool(Config.BANDS[0], psf_dir=str(tmp_path))
