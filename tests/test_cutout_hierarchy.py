"""The behaviour-bearing Cutout hierarchy (Phase 3, net-new).

Each leaf type owns its verb and delegates to the heavy engine via an
injected callable, so these tests need no archive, no PSFs, and no model:

    EuclidLRCutout.query(...)              -> downloads (injected fetch)
    SyntheticHRCutout.convolve(forward)    -> forward model (injected)
    LRCutout.super_resolve(model, ...)     -> reconstruct (injected)

The cutout is the typed handle carrying identity + provenance; the wrapped
MultiBandSkyImage stays the pure pixel/serialization workhorse.
"""

from __future__ import annotations

import numpy as np

from euclid_polish.sky.types import MultiBandSkyImage

BANDS = ("VIS", "Y_E", "J_E", "H_E")


def _hr_image(h=8, w=8):
    rng = np.random.default_rng(0)
    return MultiBandSkyImage(
        data=rng.normal(size=(h, w, 4)).astype(np.float32),
        pixel_scale_arcsec=0.05, band_names=BANDS, is_clean=True,
    )


def _lr_image(h=4, w=4):
    rng = np.random.default_rng(1)
    return MultiBandSkyImage(
        data=rng.normal(size=(h, w, 4)).astype(np.float32),
        pixel_scale_arcsec=0.10, band_names=BANDS, is_clean=False,
    )


def _store(tmp_path):
    from euclid_polish.provenance.store import ProvStore
    return ProvStore(str(tmp_path))


# --------------------------------------------------------------------------- #

def test_cutout_proxies_pixels_to_image(tmp_path):
    from euclid_polish.cutout import HRCutout
    store = _store(tmp_path)
    img = _hr_image()
    cut = HRCutout(image=img, id=store.mint())
    assert cut.band_names == BANDS
    assert cut.pixel_scale_arcsec == 0.05
    np.testing.assert_array_equal(cut.data, img.data)


def test_cutout_prov_stamp_carries_identity(tmp_path):
    from euclid_polish.cutout import HRCutout
    store = _store(tmp_path)
    parent = store.mint()
    producer = store.mint()
    cut = HRCutout(image=_hr_image(), id=store.mint(),
                   produced_by=producer, parents=(parent,))
    s = cut.prov_stamp()
    assert s.id == cut.id
    assert s.produced_by == producer
    assert s.parents == (parent,)


def test_cutout_to_tfrecord_embeds_its_id(tmp_path):
    from euclid_polish.cutout import HRCutout
    store = _store(tmp_path)
    cut = HRCutout(image=_hr_image(), id=store.mint())
    back = MultiBandSkyImage.from_tfrecord(cut.to_tfrecord(index=0))
    assert back.prov_stamp().id == cut.id


def test_synthetic_hr_downsample_makes_lr_with_hr_parent(tmp_path):
    from euclid_polish.cutout import SyntheticHRCutout, SyntheticLRCutout
    store = _store(tmp_path)
    hr = SyntheticHRCutout(image=_hr_image(), id=store.mint())

    class FakeForward:
        def process(self, image, rng=None):
            return _lr_image(), image      # (lr, hr_out)

    lr = hr.downsample(FakeForward(), store=store)
    assert isinstance(lr, SyntheticLRCutout)
    assert lr.pixel_scale_arcsec == 0.10
    assert lr.parents == (hr.id,)
    assert lr.id != hr.id


def test_synthetic_hr_downsample_implicit_store(tmp_path):
    """downsample with no explicit store uses default_store() (guarded)."""
    import unittest.mock as mock
    from euclid_polish.cutout import SyntheticHRCutout, SyntheticLRCutout
    store = _store(tmp_path)
    hr = SyntheticHRCutout(image=_hr_image(), id=store.mint())

    class FakeForward:
        def process(self, image, rng=None):
            return _lr_image(), image

    with mock.patch("euclid_polish.cutout.base.default_store", return_value=store):
        lr = hr.downsample(FakeForward())
    assert isinstance(lr, SyntheticLRCutout)
    assert lr.parents == (hr.id,)


def test_model_upsample_makes_srcutout_with_two_parents(tmp_path):
    """Regression: upsample (ex-super_resolve) still produces an SRCutout
    with parents=(model.id, lr.id). Uses Model's _reconstruct_fn injection."""
    from euclid_polish.cutout import SyntheticLRCutout, SRCutout
    from euclid_polish.model import Model

    store = _store(tmp_path)
    lr = SyntheticLRCutout(image=_lr_image(), id=store.mint())

    def fake_reconstruct(model, data):
        h, w = data.shape[:2]
        return data[..., 0], np.zeros((h * 2, w * 2, 4), np.float32)

    m = Model.__new__(Model)
    m._tf_model = object()
    m._scale = 2
    m.id = store.mint()
    m._reconstruct_fn = fake_reconstruct

    sr = m.upsample(lr, store=store)
    assert isinstance(sr, SRCutout)
    assert sr.pixel_scale_arcsec == 0.05
    assert set(sr.parents) == {m.id, lr.id}


def test_synthetic_hr_generate_returns_hr_cutout(tmp_path):
    """generate(simulator, rng=rng, store=store) returns a SyntheticHRCutout."""
    from euclid_polish.cutout import SyntheticHRCutout
    store = _store(tmp_path)
    rng = np.random.default_rng(0)

    class FakeSimulator:
        def simulate_field(self, rng, **kwargs):
            return _hr_image(), {"n_galaxies": 0}

    hr = SyntheticHRCutout.generate(FakeSimulator(), rng=rng, store=store)
    assert isinstance(hr, SyntheticHRCutout)
    assert hr.pixel_scale_arcsec == 0.05
    assert hr.data.shape[-1] == 4


def test_synthetic_hr_generate_mints_unique_ids(tmp_path):
    from euclid_polish.cutout import SyntheticHRCutout
    store = _store(tmp_path)
    rng = np.random.default_rng(1)

    class FakeSimulator:
        def simulate_field(self, rng, **kwargs):
            return _hr_image(), {}

    hr1 = SyntheticHRCutout.generate(FakeSimulator(), rng=rng, store=store)
    hr2 = SyntheticHRCutout.generate(FakeSimulator(), rng=rng, store=store)
    assert hr1.id != hr2.id


def test_synthetic_hr_generate_implicit_store(tmp_path):
    """generate without explicit store uses default_store() (guarded)."""
    import unittest.mock as mock
    from euclid_polish.cutout import SyntheticHRCutout
    store = _store(tmp_path)

    class FakeSimulator:
        def simulate_field(self, rng, **kwargs):
            return _hr_image(), {}

    with mock.patch("euclid_polish.cutout.base.default_store", return_value=store):
        hr = SyntheticHRCutout.generate(FakeSimulator())
    assert isinstance(hr, SyntheticHRCutout)


def test_euclid_query_stacks_bands_and_mints_id(tmp_path):
    from euclid_polish.cutout import EuclidLRCutout
    store = _store(tmp_path)
    levels = {"VIS": 1.0, "Y_E": 2.0, "J_E": 3.0, "H_E": 4.0}

    def fake_fetch_plane(ra, dec, band, size):
        return np.full((size, size), levels[band], np.float32)

    cut = EuclidLRCutout.query(ra=10.0, dec=-5.0, size=4, store=store,
                               fetch_plane=fake_fetch_plane)
    assert isinstance(cut, EuclidLRCutout)
    assert cut.band_names == BANDS
    assert cut.pixel_scale_arcsec == 0.10
    assert cut.data.shape == (4, 4, 4)
    np.testing.assert_array_equal(cut.data[..., 0], np.ones((4, 4), np.float32))


def test_euclid_fetch_with_injected_fetcher(tmp_path):
    from euclid_polish.cutout import EuclidLRCutout
    store = _store(tmp_path)
    levels = {"VIS": 10.0, "Y_E": 20.0, "J_E": 30.0, "H_E": 40.0}

    def fake_plane(ra, dec, band_name, size):
        return np.full((size, size), levels[band_name], dtype=np.float32)

    cut = EuclidLRCutout.fetch(ra=10.0, dec=-5.0, size=8, store=store,
                               fetch_plane=fake_plane)
    assert isinstance(cut, EuclidLRCutout)
    assert cut.band_names == ("VIS", "Y_E", "J_E", "H_E")
    assert cut.data.shape == (8, 8, 4)
    np.testing.assert_allclose(cut.data[..., 0], 10.0)
    np.testing.assert_allclose(cut.data[..., 3], 40.0)


def test_euclid_fetch_mints_provenance_id(tmp_path):
    from euclid_polish.cutout import EuclidLRCutout
    store = _store(tmp_path)

    def fake_plane(ra, dec, band_name, size):
        return np.zeros((size, size), dtype=np.float32)

    cut = EuclidLRCutout.fetch(ra=0.0, dec=0.0, size=4, store=store,
                               fetch_plane=fake_plane)
    assert cut.id is not None
    assert str(cut.id) != ""


def test_euclid_fetch_implicit_store(tmp_path):
    import unittest.mock as mock
    from euclid_polish.cutout import EuclidLRCutout
    store = _store(tmp_path)

    def fake_plane(ra, dec, band_name, size):
        return np.zeros((size, size), dtype=np.float32)

    with mock.patch("euclid_polish.cutout.base.default_store", return_value=store):
        cut = EuclidLRCutout.fetch(ra=0.0, dec=0.0, size=4, fetch_plane=fake_plane)
    assert isinstance(cut, EuclidLRCutout)


def test_euclid_fetch_pixel_scale_is_lr(tmp_path):
    from euclid_polish.cutout import EuclidLRCutout
    from euclid_polish.config import Config
    store = _store(tmp_path)

    def fake_plane(ra, dec, band_name, size):
        return np.zeros((size, size), dtype=np.float32)

    cut = EuclidLRCutout.fetch(ra=0.0, dec=0.0, size=4, store=store,
                               fetch_plane=fake_plane)
    assert abs(cut.pixel_scale_arcsec - Config.VIS_PIXEL_SCALE_ARCSEC) < 1e-6


def test_cutout_save_fits_writes_file(tmp_path):
    from euclid_polish.cutout import HRCutout
    store = _store(tmp_path)
    cut = HRCutout(image=_hr_image(), id=store.mint())
    out = tmp_path / "test.fits"
    cut.save_fits(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_cutout_save_fits_readable_float32(tmp_path):
    from astropy.io import fits
    from euclid_polish.cutout import HRCutout
    store = _store(tmp_path)
    cut = HRCutout(image=_hr_image(), id=store.mint())
    out = tmp_path / "test.fits"
    cut.save_fits(str(out))
    with fits.open(str(out)) as hdul:
        assert hdul[0].data is not None
        assert hdul[0].data.dtype.type == np.float32


def test_cutout_save_fits_carries_provid(tmp_path):
    from astropy.io import fits
    from euclid_polish.cutout import HRCutout
    store = _store(tmp_path)
    cut = HRCutout(image=_hr_image(), id=store.mint())
    out = tmp_path / "test.fits"
    cut.save_fits(str(out))
    with fits.open(str(out)) as hdul:
        assert "PROVID" in hdul[0].header
        assert hdul[0].header["PROVID"] == str(cut.id)


def test_sr_cutout_save_fits_carries_provid(tmp_path):
    from astropy.io import fits
    from euclid_polish.cutout import SRCutout
    store = _store(tmp_path)
    cut = SRCutout(image=_hr_image(), id=store.mint())
    out = tmp_path / "sr.fits"
    cut.save_fits(str(out))
    with fits.open(str(out)) as hdul:
        assert "PROVID" in hdul[0].header
