"""Operator verbs produce stamped, role-tagged Images.

    hr = simulator.generate(rng)      -> Image role 'hr'
    lr = forward.apply(hr)            -> Image role 'lr', parented on hr
    real = EuclidCatalog().fetch(...) -> Image role 'real'

(``Model.upsample`` is covered in test_model.py.) The operators are exercised
with their heavy engines stubbed — only the thin verb wrapper is under test.
"""
import numpy as np
from unittest.mock import patch
from astropy.io import fits as _fits

from euclid_polish.catalog import EuclidCatalog
from euclid_polish.catalog.downloader import fetch_cutout_at
from euclid_polish.image import Image, Role
from euclid_polish.provenance.records import Stamp
from euclid_polish.provenance.store import ProvStore

BANDS = ("VIS", "Y_E", "J_E", "H_E")


def _img(side, scale, *, role=Role.UNKNOWN, clean=True, stamp=None):
    return Image(data=np.zeros((side, side, 4), np.float32),
                 pixel_scale_arcsec=scale, band_names=BANDS,
                 is_clean=clean, role=role, stamp=stamp)


def test_generate_returns_stamped_hr_image(tmp_path):
    from euclid_polish.sky.generation.sky_simulator import SkySimulator
    sim = SkySimulator.__new__(SkySimulator)   # bypass heavy __init__
    sim.simulate_field = lambda rng, **kw: (_img(8, 0.05, clean=True), {})
    store = ProvStore(str(tmp_path))
    hr = sim.generate(rng=np.random.default_rng(0), store=store)
    assert isinstance(hr, Image)
    assert hr.role is Role.HR
    assert hr.stamp is not None and not hr.stamp.id.is_sentinel


def test_apply_returns_lr_image_parented_on_hr(tmp_path):
    from euclid_polish.sky.observation.observation_simulator import ObservationSimulator
    fwd = ObservationSimulator.__new__(ObservationSimulator)
    fwd.process = lambda hr, rng: (_img(4, 0.10, clean=False), hr)
    store = ProvStore(str(tmp_path))
    hr_id = store.mint()
    hr = _img(8, 0.05, role=Role.HR, stamp=Stamp(id=hr_id, schema_version=3))
    lr = fwd.apply(hr, rng=np.random.default_rng(0), store=store)
    assert isinstance(lr, Image)
    assert lr.role is Role.LR
    assert hr_id in lr.stamp.parents


def test_apply_unstamped_hr_has_no_parents(tmp_path):
    from euclid_polish.sky.observation.observation_simulator import ObservationSimulator
    fwd = ObservationSimulator.__new__(ObservationSimulator)
    fwd.process = lambda hr, rng: (_img(4, 0.10, clean=False), hr)
    store = ProvStore(str(tmp_path))
    lr = fwd.apply(_img(8, 0.05, role=Role.HR), store=store)
    assert lr.role is Role.LR
    assert lr.stamp.parents == ()


def test_fetch_returns_real_image(tmp_path):
    store = ProvStore(str(tmp_path))

    def fake_fetch_cutout(ra, dec, band_name, output_file, cutout_size_vis_pixels):
        hdr = _fits.Header()
        hdr["MAGZERO"] = 25.0
        _fits.PrimaryHDU(
            np.ones((cutout_size_vis_pixels, cutout_size_vis_pixels), np.float32),
            header=hdr,
        ).writeto(output_file)
        return True, None

    with patch("euclid_polish.catalog.client.fetch_cutout_at", side_effect=fake_fetch_cutout):
        img = EuclidCatalog._unauthenticated().fetch(ra=10.0, dec=-5.0, size=4, store=store)
    assert isinstance(img, Image)
    assert img.role is Role.REAL
    assert img.band_names == BANDS
    assert img.shape == (4, 4, 4)
    assert abs(img.pixel_scale_arcsec - 0.10) < 1e-9
    assert img.stamp is not None
