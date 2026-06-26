"""Tests for the ImageSet collection (euclid_polish.image.collection)."""
import os

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.image import Image, ImageSet, Role


def _img(i, role=Role.HR):
    return Image(data=np.full((4, 4, 1), float(i), np.float32),
                 pixel_scale_arcsec=0.05, band_names=("VIS",),
                 is_clean=True, role=role)


def test_from_images_len_getitem():
    s = ImageSet.from_images([_img(0), _img(1)])
    assert len(s) == 2
    assert s[1].data[0, 0, 0] == 1.0


def test_iter_yields_images():
    s = ImageSet.from_images([_img(0), _img(1)])
    assert all(isinstance(im, Image) for im in s)


def test_write_read_roundtrip(tmp_path):
    s = ImageSet.from_images([_img(0), _img(1), _img(2)])
    path = s.write(str(tmp_path), "hr_train")
    back = ImageSet.read(path)
    assert len(back) == 3
    assert sorted(im.data[0, 0, 0] for im in back) == [0.0, 1.0, 2.0]


def test_read_limit(tmp_path):
    s = ImageSet.from_images([_img(i) for i in range(5)])
    path = s.write(str(tmp_path), "hr_train")
    back = ImageSet.read(path, limit=2)
    assert len(back) == 2


def test_by_role():
    s = ImageSet.from_images([_img(0, Role.LR), _img(1, Role.SR), _img(2, Role.LR)])
    only_lr = s.by_role(Role.LR)
    assert isinstance(only_lr, ImageSet)
    assert len(only_lr) == 2


def test_split_is_disjoint_and_sized():
    s = ImageSet.from_images([_img(i) for i in range(10)])
    a, b = s.split(0.8, rng=np.random.default_rng(0))
    assert len(a) == 8 and len(b) == 2
    seen = sorted(im.data[0, 0, 0] for im in a) + sorted(im.data[0, 0, 0] for im in b)
    assert sorted(seen) == [float(i) for i in range(10)]


def _4band(side, scale, role):
    rng = np.random.default_rng(int(side * 10 + (scale * 100)))
    data = (np.abs(rng.normal(size=(side, side, 4))) * 100.0).astype(np.float32)
    return Image(data=data, pixel_scale_arcsec=scale,
                 band_names=Config.LR_INPUT_BAND_NAMES,
                 is_clean=(role is not Role.LR), role=role)


def test_plot_reconstruction_writes_png(tmp_path):
    s = ImageSet.from_images([
        _4band(8, 0.10, Role.LR),
        _4band(16, 0.05, Role.SR),
        _4band(16, 0.05, Role.HR),
    ])
    out = str(tmp_path / "recon.png")
    assert s.plot_reconstruction(out) == out
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_reconstruction_requires_sr():
    with pytest.raises(ValueError):
        ImageSet.from_images([_4band(8, 0.10, Role.LR)]).plot_reconstruction("x.png")
