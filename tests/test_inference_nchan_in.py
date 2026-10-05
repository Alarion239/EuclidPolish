"""``infer_checkpoint_nchan_in`` reads the entry conv's in-dim for any knee count.

A multi-knee member reads ``4·K`` input channels. From ``K = 7`` (28 channels)
on, that is wider than the trunk's narrowest body kernel (``int(0.8·32) = 25``),
and at ``K = 8`` it equals the trunk width (32), so neither a minimum over the
kernels nor a shape match can find the input width — only the entry conv can.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.training.inference import (
    infer_checkpoint_nchan_in,
    infer_checkpoint_nchan_out,
    load_model_from_checkpoint,
)
from euclid_polish.training.models.wdsr import wdsr


def _save(model, tmp_path) -> str:
    d = str(tmp_path / "ckpt")
    tf.train.Checkpoint(model=model).save(d + "/ckpt")
    return d


@pytest.mark.parametrize("knees", [7, 8])
def test_single_image_member_with_many_knees(tmp_path, knees):
    """Option-2 member: 4·K knee-major channels in, 4 bands out."""
    nchan_in = 4 * knees
    model = wdsr(scale=2, num_res_blocks=1, nchan_in=nchan_in, nchan_out=4,
                 input_knees=knees)
    d = _save(model, tmp_path)
    assert infer_checkpoint_nchan_in(d) == nchan_in
    assert infer_checkpoint_nchan_out(d, scale=2, nchan_in=nchan_in) == 4


def test_multi_image_member_with_seven_knees(tmp_path):
    """Option-1 member: 4·K channels in, one image per knee out (4·K)."""
    model = wdsr(scale=2, num_res_blocks=1, nchan_in=28, nchan_out=28)
    d = _save(model, tmp_path)
    assert infer_checkpoint_nchan_in(d) == 28
    assert infer_checkpoint_nchan_out(d, scale=2, nchan_in=28) == 28


def test_seven_knee_checkpoint_restores_exactly(tmp_path):
    model = wdsr(scale=2, num_res_blocks=1, nchan_in=28, nchan_out=4, input_knees=7)
    d = _save(model, tmp_path)
    loaded = load_model_from_checkpoint(d, scale=2, num_res_blocks=1)
    assert int(loaded.inputs[0].shape[-1]) == 28
    assert int(loaded.outputs[0].shape[-1]) == 4
    x = tf.constant(np.random.default_rng(7).normal(size=(1, 6, 6, 28)).astype(np.float32))
    np.testing.assert_allclose(loaded(x).numpy(), model(x).numpy(), rtol=1e-5, atol=1e-5)
