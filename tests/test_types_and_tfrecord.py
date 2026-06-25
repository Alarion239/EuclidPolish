"""Tests for ``MultiBandSkyImage`` and v2 TFRecord I/O."""

from __future__ import annotations

import os

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.sky.tfrecord import (
    parse_record_graph_v2,
    read_multiband_skyimages,
    write_multiband_skyimages,
)
from euclid_polish.sky.types import MultiBandSkyImage


# ---------------------------------------------------------------------------
# MultiBandSkyImage construction
# ---------------------------------------------------------------------------

def test_accepts_3d_data():
    data = np.zeros((8, 8, 4), dtype=np.float32)
    img = MultiBandSkyImage(
        data=data, pixel_scale_arcsec=0.10,
        band_names=("VIS", "Y_E", "J_E", "H_E"), is_clean=False,
    )
    assert img.shape == (8, 8, 4)
    assert img.num_channels == 4


def test_promotes_2d_to_3d():
    """Legacy 2-D arrays are auto-promoted to (H, W, 1)."""
    data = np.zeros((8, 8), dtype=np.float32)
    img = MultiBandSkyImage(
        data=data, pixel_scale_arcsec=0.05,
        band_names=("VIS",), is_clean=True,
    )
    assert img.shape == (8, 8, 1)
    assert img.num_channels == 1


def test_rejects_non_2d_3d():
    with pytest.raises(ValueError):
        MultiBandSkyImage(
            data=np.zeros((8,), dtype=np.float32),
            pixel_scale_arcsec=0.05, band_names=("VIS",), is_clean=True,
        )


def test_rejects_band_channel_mismatch():
    with pytest.raises(ValueError, match="band_names"):
        MultiBandSkyImage(
            data=np.zeros((8, 8, 4), dtype=np.float32),
            pixel_scale_arcsec=0.10, band_names=("VIS", "Y_E"), is_clean=False,
        )


def test_preserves_data():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(16, 16, 4)).astype(np.float32)
    img = MultiBandSkyImage(
        data=data, pixel_scale_arcsec=0.10,
        band_names=("VIS", "Y_E", "J_E", "H_E"), is_clean=False,
    )
    np.testing.assert_array_equal(img.data, data)


def test_metadata_default():
    img = MultiBandSkyImage(
        data=np.zeros((4, 4, 1), dtype=np.float32),
        pixel_scale_arcsec=0.05, band_names=("VIS",), is_clean=True,
    )
    assert img.metadata == {}
    assert isinstance(img.metadata, dict)


# ---------------------------------------------------------------------------
# v2 TFRecord round-trip
# ---------------------------------------------------------------------------

@pytest.fixture
def lr_image() -> MultiBandSkyImage:
    rng = np.random.default_rng(0)
    return MultiBandSkyImage(
        data=rng.normal(size=(8, 12, 4)).astype(np.float32),
        pixel_scale_arcsec=0.10,
        band_names=("VIS", "Y_E", "J_E", "H_E"),
        is_clean=False,
    )


@pytest.fixture
def hr_image() -> MultiBandSkyImage:
    rng = np.random.default_rng(1)
    return MultiBandSkyImage(
        data=rng.normal(size=(16, 24, 1)).astype(np.float32),
        pixel_scale_arcsec=0.05,
        band_names=("VIS",),
        is_clean=True,
    )


def test_v2_roundtrip_exact(tmp_path, lr_image, hr_image):
    path_lr = write_multiband_skyimages([lr_image], "dirty_train", records_dir=str(tmp_path))
    path_hr = write_multiband_skyimages([hr_image], "clean_train", records_dir=str(tmp_path))

    [back_lr] = read_multiband_skyimages(path_lr, num_images=1)
    [back_hr] = read_multiband_skyimages(path_hr, num_images=1)

    np.testing.assert_array_equal(back_lr.data, lr_image.data)
    np.testing.assert_array_equal(back_hr.data, hr_image.data)
    assert back_lr.band_names == lr_image.band_names
    assert back_hr.band_names == hr_image.band_names
    assert back_lr.pixel_scale_arcsec == pytest.approx(lr_image.pixel_scale_arcsec)
    assert back_hr.pixel_scale_arcsec == pytest.approx(hr_image.pixel_scale_arcsec)
    assert back_lr.is_clean is False
    assert back_hr.is_clean is True


def test_handles_multiple_records(tmp_path):
    rng = np.random.default_rng(7)
    imgs = [
        MultiBandSkyImage(
            data=rng.normal(size=(4, 4, 4)).astype(np.float32),
            pixel_scale_arcsec=0.10,
            band_names=("VIS", "Y_E", "J_E", "H_E"),
            is_clean=False,
        )
        for _ in range(5)
    ]
    path = write_multiband_skyimages(imgs, "dirty_train", records_dir=str(tmp_path))
    back = read_multiband_skyimages(path, num_images=5)
    assert len(back) == 5
    for original, reread in zip(imgs, back):
        np.testing.assert_array_equal(reread.data, original.data)


def test_parse_record_graph_v2_shape(tmp_path, lr_image):
    path = write_multiband_skyimages([lr_image], "dirty_train", records_dir=str(tmp_path))
    ds = tf.data.TFRecordDataset(path).map(lambda raw: parse_record_graph_v2(raw, 4))
    t = next(iter(ds))
    assert tuple(t.shape) == lr_image.shape
    np.testing.assert_array_equal(t.numpy(), lr_image.data)


def test_parse_record_graph_v2_rejects_wrong_nchan(tmp_path, lr_image):
    path = write_multiband_skyimages([lr_image], "dirty_train", records_dir=str(tmp_path))
    ds = tf.data.TFRecordDataset(path).map(lambda raw: parse_record_graph_v2(raw, 1))
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        next(iter(ds))


def test_write_creates_expected_path(tmp_path, lr_image):
    path = write_multiband_skyimages([lr_image], "dirty_train", records_dir=str(tmp_path))
    assert path == os.path.join(str(tmp_path), "dirty_train.tfrecord")
    assert os.path.exists(path)


# ---------------------------------------------------------------------------
# v3 provenance stamp (Phase 1)
# ---------------------------------------------------------------------------

def test_v3_stamp_survives_round_trip(tmp_path):
    from euclid_polish.provenance import ProvId, Stamp
    rng = np.random.default_rng(3)
    stamp = Stamp(
        id=ProvId("4b1e7a90"),
        produced_by=ProvId("7f3a9c21"),
        parents=(ProvId("2b8e44d1"),),
        schema_version=3,
        subset="train",
    )
    img = MultiBandSkyImage(
        data=rng.normal(size=(8, 8, 4)).astype(np.float32),
        pixel_scale_arcsec=0.10,
        band_names=("VIS", "Y_E", "J_E", "H_E"),
        is_clean=False,
        stamp=stamp,
    )
    path = write_multiband_skyimages([img], "dirty_train", records_dir=str(tmp_path))
    [back] = read_multiband_skyimages(path, num_images=1)
    assert back.prov_stamp() == stamp
    assert back.subset == "train"      # subset drop fixed via the stamp


def test_unstamped_record_reads_back_with_no_stamp(tmp_path, lr_image):
    path = write_multiband_skyimages([lr_image], "dirty_train", records_dir=str(tmp_path))
    [back] = read_multiband_skyimages(path, num_images=1)
    assert back.prov_stamp() is None


def test_graph_parser_identical_with_and_without_stamp(tmp_path):
    """The training graph parser must be byte-identical regardless of stamp."""
    from euclid_polish.provenance import ProvId, Stamp
    rng = np.random.default_rng(11)
    data = rng.normal(size=(8, 8, 4)).astype(np.float32)
    plain = MultiBandSkyImage(
        data=data, pixel_scale_arcsec=0.10,
        band_names=("VIS", "Y_E", "J_E", "H_E"), is_clean=False,
    )
    stamped = MultiBandSkyImage(
        data=data, pixel_scale_arcsec=0.10,
        band_names=("VIS", "Y_E", "J_E", "H_E"), is_clean=False,
        stamp=Stamp(id=ProvId("4b1e7a90")),
    )
    p_plain = write_multiband_skyimages([plain], "dirty_train", records_dir=str(tmp_path / "a"))
    p_stamp = write_multiband_skyimages([stamped], "dirty_train", records_dir=str(tmp_path / "b"))
    t_plain = next(iter(tf.data.TFRecordDataset(p_plain).map(lambda r: parse_record_graph_v2(r, 4))))
    t_stamp = next(iter(tf.data.TFRecordDataset(p_stamp).map(lambda r: parse_record_graph_v2(r, 4))))
    np.testing.assert_array_equal(t_plain.numpy(), t_stamp.numpy())
