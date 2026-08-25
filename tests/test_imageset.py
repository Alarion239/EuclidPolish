"""Tests for the ImageSet collection (euclid_polish.image.collection)."""
import numpy as np

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
    back = list(ImageSet.read(path))
    assert len(back) == 3
    assert sorted(im.data[0, 0, 0] for im in back) == [0.0, 1.0, 2.0]


def test_lazy_write_streams_without_materialising(tmp_path):
    source = ImageSet.from_images([_img(0), _img(1), _img(2)])
    source_path = source.write(str(tmp_path), "source")
    lazy = ImageSet.read(source_path)

    copied_path = lazy.write(str(tmp_path), "copy")

    copied = list(ImageSet.read(copied_path))
    assert [image.data[0, 0, 0] for image in copied] == [0.0, 1.0, 2.0]


def test_read_limit(tmp_path):
    s = ImageSet.from_images([_img(i) for i in range(5)])
    path = s.write(str(tmp_path), "hr_train")
    back = list(ImageSet.read(path, num_images=2))
    assert len(back) == 2


def test_lazy_len_raises():
    """len() on a lazy (TFRecord-backed) ImageSet must raise TypeError."""
    import pytest
    lazy = ImageSet.read("/nonexistent/path.tfrecord")
    with pytest.raises(TypeError, match="len\\(\\) not supported for lazy"):
        len(lazy)


def test_lazy_getitem_raises():
    """Indexing a lazy ImageSet must raise TypeError."""
    import pytest
    lazy = ImageSet.read("/nonexistent/path.tfrecord")
    with pytest.raises(TypeError, match="Indexing not supported for lazy"):
        _ = lazy[0]


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
