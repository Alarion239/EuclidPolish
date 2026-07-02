# tests/test_model.py
"""Tests for euclid_polish.model.Model — reconstruct/load are injectable so
tests run with no real TF checkpoint."""
from __future__ import annotations

import os

import numpy as np

from euclid_polish.image import Image, Role
from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import Stamp
from euclid_polish.provenance.store import ProvStore

BANDS = ("VIS", "Y_E", "J_E", "H_E")


def _lr_image(h: int = 4, w: int = 4) -> Image:
    rng = np.random.default_rng(1)
    return Image(
        data=rng.normal(size=(h, w, 4)).astype(np.float32),
        pixel_scale_arcsec=0.10, band_names=BANDS, is_clean=False,
    )


def _stamped_lr(store, h: int = 4, w: int = 4) -> Image:
    """An LR Image carrying a freshly-minted provenance stamp (so it has an id)."""
    return _lr_image(h, w).with_stamp(Stamp(id=store.mint(), schema_version=3))


def _fake_load(ckpt_dir, scale, num_res_blocks, **kwargs):
    return object()


def _fake_reconstruct(model, data):
    arr = np.asarray(data, dtype=np.float32)
    h, w = arr.shape[:2]
    vis = arr[..., 0] if arr.ndim == 3 else arr
    sr = np.zeros((h * 2, w * 2, 4), dtype=np.float32)
    return vis, sr


def _bare_model(model_id=None):
    """A Model with internals set directly, bypassing __init__/TF load."""
    from euclid_polish.model import Model
    m = Model.__new__(Model)
    m._tf_model = object()
    m.id = model_id
    m._reconstruct_fn = _fake_reconstruct
    return m


def test_model_id_none_for_legacy_checkpoint(tmp_path):
    from euclid_polish.model import Model
    ckpt_dir = str(tmp_path / "ckpt")
    os.makedirs(ckpt_dir)
    m = Model(ckpt_dir, _load_fn=_fake_load, _reconstruct_fn=_fake_reconstruct)
    assert m.id is None


def test_model_id_read_from_provenance_json(tmp_path):
    from euclid_polish.model import Model
    from euclid_polish.provenance.checkpoint import write_checkpoint_provenance
    from euclid_polish.provenance.records import Stamp
    ckpt_dir = str(tmp_path / "ckpt")
    os.makedirs(ckpt_dir)
    store = ProvStore(str(tmp_path / "prov"))
    expected_id = store.mint()
    write_checkpoint_provenance(ckpt_dir, Stamp(id=expected_id))
    m = Model(ckpt_dir, _load_fn=_fake_load, _reconstruct_fn=_fake_reconstruct)
    assert m.id == expected_id


def test_upsample_array_returns_ndarray():
    m = _bare_model()
    result = m.upsample_array(np.zeros((4, 4, 4), dtype=np.float32))
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 8 and result.shape[1] == 8


def test_upsample_array_2d_output():
    m = _bare_model()

    def _rec_2d(model, data):
        arr = np.asarray(data, dtype=np.float32)
        h, w = arr.shape[:2]
        return arr[..., 0], np.zeros((h * 2, w * 2), dtype=np.float32)

    m._reconstruct_fn = _rec_2d
    result = m.upsample_array(np.zeros((4, 4, 4), dtype=np.float32))
    assert result.shape == (8, 8)


def test_upsample_returns_sr_image(tmp_path):
    store = ProvStore(str(tmp_path))
    m = _bare_model(model_id=store.mint())
    sr = m.upsample(_stamped_lr(store), store=store)
    assert isinstance(sr, Image)
    assert sr.role is Role.SR


def test_upsample_sr_pixel_scale_is_hr(tmp_path):
    from euclid_polish.config import Config
    store = ProvStore(str(tmp_path))
    m = _bare_model()
    sr = m.upsample(_stamped_lr(store), store=store)
    assert abs(sr.pixel_scale_arcsec - Config.DEFAULT_PIXEL_SCALE) < 1e-6


def test_upsample_parents_are_model_id_and_lr_id(tmp_path):
    store = ProvStore(str(tmp_path))
    model_id = store.mint()
    m = _bare_model(model_id=model_id)
    lr_id = store.mint()
    lr = _lr_image().with_stamp(Stamp(id=lr_id, schema_version=3))
    sr = m.upsample(lr, store=store)
    assert set(sr.stamp.parents) == {model_id, lr_id}


def test_upsample_none_model_id_excludes_none(tmp_path):
    store = ProvStore(str(tmp_path))
    m = _bare_model(model_id=None)
    lr_id = store.mint()
    lr = _lr_image().with_stamp(Stamp(id=lr_id, schema_version=3))
    sr = m.upsample(lr, store=store)
    assert lr_id in sr.stamp.parents
    assert None not in sr.stamp.parents


def test_upsample_implicit_store(tmp_path):
    import unittest.mock as mock
    store = ProvStore(str(tmp_path))
    m = _bare_model()
    lr = _stamped_lr(store)
    with mock.patch("euclid_polish.provenance.defaults.default_store",
                    return_value=store):
        sr = m.upsample(lr)
    assert isinstance(sr, Image)




def test_fresh_model_builds_multiband_net(tmp_path):
    """A from-scratch Model (no checkpoint, no injected loader) must build a
    4-band WDSR matching the current pipeline, not the legacy 1-channel wdsr
    default — otherwise a fresh ensemble member crashes on 4-band data
    ('expected shape (..., 1), found (..., 4)')."""
    from euclid_polish.config import Config
    from euclid_polish.model import Model
    m = Model(str(tmp_path / "fresh_ckpt"), scale=2, num_res_blocks=2)
    assert int(m._tf_model.inputs[0].shape[-1]) == Config.NUM_LR_CHANNELS
    assert int(m._tf_model.outputs[0].shape[-1]) == Config.NUM_HR_CHANNELS
