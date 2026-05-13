"""``reconstruct()`` must accept 2D, 3D-single-channel, and 3D-multi-channel
inputs — the multi-channel path is what every web/CLI reconstruct call hits
under the current 4-band WDSR model."""

from __future__ import annotations

import numpy as np

from euclid_polish.training.inference import reconstruct
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
