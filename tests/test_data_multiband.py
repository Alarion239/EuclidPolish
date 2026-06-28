"""Tests for per-band asinh stretch and the WDSR multi-band model."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.training.augmentation import (
    asinh_stretch_hr,
    asinh_stretch_lr,
    inverse_asinh_stretch_hr,
    inverse_asinh_stretch_lr,
)
from euclid_polish.training.models.wdsr import wdsr

# ---------------------------------------------------------------------------
# Per-band asinh stretch
# ---------------------------------------------------------------------------

def test_asinh_stretch_lr_inverse():
    rng = np.random.default_rng(0)
    raw = tf.constant(rng.normal(size=(2, 4, 4, 4), scale=500.0), dtype=tf.float32)
    stretched = asinh_stretch_lr(raw)
    recovered = inverse_asinh_stretch_lr(stretched)
    np.testing.assert_allclose(recovered.numpy(), raw.numpy(), rtol=1e-4, atol=1e-2)


def test_asinh_stretch_hr_inverse():
    rng = np.random.default_rng(1)
    raw = tf.constant(rng.normal(size=(2, 4, 4, 1), scale=500.0), dtype=tf.float32)
    stretched = asinh_stretch_hr(raw)
    recovered = inverse_asinh_stretch_hr(stretched)
    np.testing.assert_allclose(recovered.numpy(), raw.numpy(), rtol=1e-4, atol=1e-2)


def test_per_band_stretch_uses_correct_scales():
    """Each LR channel should be divided by its band's specific scale."""
    x = tf.ones((1, 1, 1, 4), dtype=tf.float32)
    y = asinh_stretch_lr(x)
    expected = np.arcsinh(
        1.0 / np.array([Config.get_band(b).asinh_stretch_scale_e
                        for b in Config.LR_INPUT_BAND_NAMES], dtype=np.float32)
    )
    np.testing.assert_allclose(y.numpy()[0, 0, 0, :], expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# WDSR with asymmetric channels
# ---------------------------------------------------------------------------

def test_wdsr_multiband_shape():
    model = wdsr(scale=2, nchan_in=4, nchan_out=1, num_res_blocks=2)
    x = tf.zeros((1, 8, 8, 4), dtype=tf.float32)
    y = model(x)
    assert tuple(y.shape) == (1, 16, 16, 1)


def test_wdsr_backcompat_via_nchan_kwarg():
    """Calling with the legacy ``nchan=1`` keyword still works."""
    model = wdsr(scale=2, nchan=1, num_res_blocks=2)
    x = tf.zeros((1, 8, 8, 1), dtype=tf.float32)
    y = model(x)
    assert tuple(y.shape) == (1, 16, 16, 1)


def test_wdsr_trainable_params_nonzero():
    model = wdsr(scale=2, nchan_in=4, nchan_out=1, num_res_blocks=2)
    n = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    assert n > 1000


def test_wdsr_4band_output_shape():
    """The 4-band model maps a 4-channel LR stack to a 4-channel SR."""
    model = wdsr(scale=2, nchan_in=4, nchan_out=4, num_res_blocks=2)
    x = tf.zeros((1, 8, 8, 4), dtype=tf.float32)
    y = model(x)
    assert tuple(y.shape) == (1, 16, 16, 4)


def test_wdsr_per_band_skip_isolates_bands():
    """With the trunk's tail conv zeroed, the model output is the skip
    branch alone — and the per-band skip must route input band k ONLY to
    output band k (all cross-band paths go through the shared trunk)."""
    model = wdsr(scale=2, nchan_in=4, nchan_out=4, num_res_blocks=2)
    for layer in model.layers:
        inner = getattr(layer, "layer", None)
        if inner is not None and inner.name == "conv2d_main_scale_2":
            layer.set_weights([np.zeros_like(w) for w in layer.get_weights()])
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 8, 8, 4)).astype(np.float32)
    y0 = model(x).numpy()
    for k in range(4):
        xk = x.copy()
        xk[..., k] += 1.0
        diff = np.abs(model(xk).numpy() - y0).reshape(-1, 4).max(axis=0)
        assert diff[k] > 1e-7, f"band {k} skip carries no signal"
        others = [diff[j] for j in range(4) if j != k]
        assert max(others) < 1e-10, (
            f"per-band skip leaked band {k} into other output bands: {diff}"
        )


def test_wdsr_per_band_skip_requires_symmetric_channels():
    with pytest.raises(ValueError, match="per_band_skip"):
        wdsr(scale=2, nchan_in=4, nchan_out=1, num_res_blocks=2,
             per_band_skip=True)


def test_wdsr_runs_one_training_step():
    """A forward+backward pass updates trainable weights."""
    rng = np.random.default_rng(0)
    lr = tf.constant(rng.normal(size=(2, 8, 8, 4)).astype(np.float32))
    hr = tf.constant(rng.normal(size=(2, 16, 16, 4)).astype(np.float32))
    model = wdsr(scale=2, nchan_in=4, nchan_out=4, num_res_blocks=2)
    opt = tf.keras.optimizers.Adam(1e-3)
    loss_fn = tf.keras.losses.MeanAbsoluteError()
    before = [v.numpy().copy() for v in model.trainable_variables[:3]]
    with tf.GradientTape() as tape:
        sr = model(lr)
        loss = loss_fn(hr, sr)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    after = [v.numpy() for v in model.trainable_variables[:3]]
    assert any(not np.allclose(a, b) for a, b in zip(before, after))
