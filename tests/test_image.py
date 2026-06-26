"""Tests for the clean Image data atom (euclid_polish.image.core)."""
import os

import numpy as np
import pytest

from euclid_polish.image import Image, Role
from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import Stamp


def _img(side=4, c=4, scale=0.10, role=Role.LR, clean=False):
    data = np.arange(side * side * c, dtype=np.float32).reshape(side, side, c)
    bands = ("VIS", "Y_E", "J_E", "H_E")[:c]
    return Image(data=data, pixel_scale_arcsec=scale,
                 band_names=bands, is_clean=clean, role=role)


def test_role_defaults_to_unknown():
    im = Image(data=np.zeros((4, 4, 1), np.float32), pixel_scale_arcsec=0.05,
               band_names=("VIS",), is_clean=True)
    assert im.role is Role.UNKNOWN


def test_unknown_role_value_falls_back():
    assert Role("not-a-real-role") is Role.UNKNOWN


def test_dropped_physics_methods_are_gone():
    im = _img()
    for gone in ("convolved_with", "convolved_per_band", "with_band_noise"):
        assert not hasattr(im, gone), f"{gone} should have moved to an operator"


def test_role_roundtrips_through_tfrecord():
    im = _img(role=Role.LR)
    back = Image.from_tfrecord(im.to_tfrecord(index=3))
    assert back.role is Role.LR
    np.testing.assert_array_equal(back.data, im.data)


def test_legacy_record_without_role_parses_unknown():
    im = Image(data=np.zeros((4, 4, 4), np.float32), pixel_scale_arcsec=0.10,
               band_names=("VIS", "Y_E", "J_E", "H_E"), is_clean=False)
    back = Image.from_tfrecord(im.to_tfrecord(index=0))
    assert back.role is Role.UNKNOWN


def test_stamp_rides_in_tfrecord():
    sid = ProvId("abcd1234")
    im = _img().with_stamp(Stamp(id=sid, schema_version=3))
    back = Image.from_tfrecord(im.to_tfrecord())
    assert back.stamp is not None and back.stamp.id == sid


def test_psnr_against_identical_is_inf():
    im = _img()
    assert im.psnr_against(im) == float("inf")


def test_psnr_against_different_is_finite():
    a = _img()
    b = Image(data=a.data + 5.0, pixel_scale_arcsec=a.pixel_scale_arcsec,
              band_names=a.band_names, is_clean=a.is_clean)
    assert np.isfinite(a.psnr_against(b))


def test_save_and_load_fits_roundtrip(tmp_path):
    sid = ProvId("beef0001")
    im = _img(role=Role.SR, scale=0.05).with_stamp(Stamp(id=sid, schema_version=3))
    p = str(tmp_path / "x.fits")
    im.save_fits(p)
    back = Image.from_fits(p)
    np.testing.assert_allclose(back.data, im.data, rtol=1e-6)
    assert back.band_names == im.band_names
    assert abs(back.pixel_scale_arcsec - 0.05) < 1e-9
    assert back.role is Role.SR
    assert back.stamp is not None and back.stamp.id == sid


def test_plot_writes_file(tmp_path):
    p = str(tmp_path / "q.png")
    _img().plot(p)
    assert os.path.exists(p) and os.path.getsize(p) > 0


def test_crop_and_rebin_preserved():
    im = _img(side=8, c=1, scale=0.05, role=Role.HR, clean=True)
    assert im.centre_cropped_to(4).shape == (4, 4, 1)
    assert im.sum_rebinned(2).pixel_scale_arcsec == pytest.approx(0.10)
