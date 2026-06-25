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


def test_synthetic_hr_convolve_makes_lr_with_hr_parent(tmp_path):
    from euclid_polish.cutout import SyntheticHRCutout, SyntheticLRCutout
    store = _store(tmp_path)
    hr = SyntheticHRCutout(image=_hr_image(), id=store.mint())

    class FakeForward:
        def process(self, image, rng=None):
            return _lr_image(), image      # (lr, hr_out)

    lr = hr.convolve(FakeForward(), store)
    assert isinstance(lr, SyntheticLRCutout)
    assert lr.pixel_scale_arcsec == 0.10
    assert lr.parents == (hr.id,)
    assert lr.id != hr.id


def test_lr_super_resolve_makes_srcutout_with_two_parents(tmp_path):
    from euclid_polish.cutout import SyntheticLRCutout, SRCutout
    store = _store(tmp_path)
    lr = SyntheticLRCutout(image=_lr_image(), id=store.mint())
    model_id = store.mint()

    def fake_reconstruct(model, data):
        h, w = data.shape[:2]
        return data[..., 0], np.zeros((h * 2, w * 2, 4), np.float32)

    sr = lr.super_resolve(model=object(), model_id=model_id, store=store,
                          reconstruct_fn=fake_reconstruct)
    assert isinstance(sr, SRCutout)
    assert sr.pixel_scale_arcsec == 0.05
    assert set(sr.parents) == {model_id, lr.id}


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
