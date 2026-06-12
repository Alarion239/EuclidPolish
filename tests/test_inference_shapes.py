"""``reconstruct()`` must accept 2D, 3D-single-channel, and 3D-multi-channel
inputs — the multi-channel path is what every web/CLI reconstruct call hits
under the current 4-band WDSR model."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from euclid_polish.training.inference import (
    infer_checkpoint_nchan_in,
    infer_checkpoint_nchan_out,
    load_model_from_checkpoint,
    reconstruct,
)
from euclid_polish.training.models.wdsr import wdsr


def _model(nchan_in: int):
    return wdsr(scale=2, num_res_blocks=2, nchan_in=nchan_in, nchan_out=1)


def test_reconstruct_2d_input():
    lr = np.random.randn(32, 32).astype(np.float32) * 1000.0
    lr_out, sr_out = reconstruct(_model(1), lr)
    assert lr_out.shape == (32, 32)
    assert sr_out.shape == (64, 64)


def test_reconstruct_3d_single_channel_input():
    lr = np.random.randn(32, 32, 1).astype(np.float32) * 1000.0
    lr_out, sr_out = reconstruct(_model(1), lr)
    assert lr_out.shape == (32, 32)
    assert sr_out.shape == (64, 64)


def test_reconstruct_3d_multi_channel_input():
    """The 4-channel cube is what dirty_* TFRecords + Euclid-cutout
    inference both produce."""
    lr = np.random.randn(32, 32, 4).astype(np.float32) * 1000.0
    lr_out, sr_out = reconstruct(_model(4), lr)
    # LR returned for display is VIS-only (channel 0).
    assert lr_out.shape == (32, 32)
    assert sr_out.shape == (64, 64)


def test_reconstruct_slices_4band_cube_for_vis_only_model():
    """A VIS-only (1-channel) model fed the full 4-band cube must slice to
    VIS (channel 0) automatically — callers always pass the full cube."""
    lr = np.random.randn(32, 32, 4).astype(np.float32) * 1000.0
    lr_out, sr_out = reconstruct(_model(1), lr)
    assert lr_out.shape == (32, 32)
    assert sr_out.shape == (64, 64)


def test_reconstruct_rejects_too_few_channels():
    """A 4-channel model fed a 1-channel cube can't be widened — clear error."""
    lr = np.random.randn(32, 32, 1).astype(np.float32) * 1000.0
    try:
        reconstruct(_model(4), lr)
    except ValueError as e:
        assert "input channels" in str(e)
    else:
        raise AssertionError("expected ValueError for too-few channels")


def test_checkpoint_self_describes_channel_count(tmp_path):
    """``infer_checkpoint_nchan_in`` reads the LR channel count from the saved
    weights, and ``load_model_from_checkpoint`` rebuilds the matching model
    without the caller specifying ``nchan_in`` — the basis for loading a
    VIS-only (1-ch) and a standard (4-ch) checkpoint everywhere alike."""
    for nin in (1, 4):
        d = str(tmp_path / f"ckpt{nin}")
        ck = tf.train.Checkpoint(model=_model(nin))
        ck.save(d + "/ckpt")
        assert infer_checkpoint_nchan_in(d) == nin
        assert infer_checkpoint_nchan_out(d, scale=2, nchan_in=nin) == 1
        m = load_model_from_checkpoint(d, scale=2, num_res_blocks=2)
        assert int(m.inputs[0].shape[-1]) == nin
        assert int(m.outputs[0].shape[-1]) == 1


def test_checkpoint_self_describes_4band_output(tmp_path):
    """A 4-band (VIS+NISP → VIS+NISP, per-band skip) checkpoint must load
    as a 4-output model with no caller hints — and its per-band skip
    kernels (in-dim 1) must NOT fool the nchan_in inference into building
    a 1-channel model (the regression that motivated excluding 5×5
    kernels from the in-dim scan)."""
    d = str(tmp_path / "ckpt4x4")
    model = wdsr(scale=2, num_res_blocks=2, nchan_in=4, nchan_out=4)
    tf.train.Checkpoint(model=model).save(d + "/ckpt")
    assert infer_checkpoint_nchan_in(d) == 4
    assert infer_checkpoint_nchan_out(d, scale=2, nchan_in=4) == 4
    m = load_model_from_checkpoint(d, scale=2, num_res_blocks=2)
    assert int(m.inputs[0].shape[-1]) == 4
    assert int(m.outputs[0].shape[-1]) == 4
    # Restored weights are functionally identical to the saved model.
    x = np.random.randn(1, 16, 16, 4).astype(np.float32)
    np.testing.assert_allclose(model(x).numpy(), m(x).numpy(),
                               rtol=0, atol=1e-6)


def test_reconstruct_returns_4band_cube_for_4band_model():
    """The 4-band model's SR comes back as a (H, W, 4) cube (channel 0 =
    VIS) — the shape the color panels and per-band FITS writers consume."""
    model = wdsr(scale=2, num_res_blocks=2, nchan_in=4, nchan_out=4)
    lr = np.random.randn(32, 32, 4).astype(np.float32) * 1000.0
    lr_out, sr_out = reconstruct(model, lr)
    assert lr_out.shape == (32, 32)
    assert sr_out.shape == (64, 64, 4)
