"""Tests for the multi-band data loader and WDSR multi-band model."""

from __future__ import annotations

import os

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.sky.tfrecord import write_multiband_skyimages
from euclid_polish.sky.types import MultiBandSkyImage
from euclid_polish.training.data_multiband import (
    MultiBandEuclidDataset,
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
# Dataset end-to-end (write → read)
# ---------------------------------------------------------------------------

def _write_test_records(tmp_path) -> str:
    rng = np.random.default_rng(0)
    # Two records each: HR at 0.05" (64x64, 1ch), LR at 0.10" (32x32, 4ch).
    hr_imgs = [
        MultiBandSkyImage(
            data=(rng.normal(size=(64, 64, 1)) * 100.0).astype(np.float32),
            pixel_scale_arcsec=0.05,
            band_names=("VIS",),
            is_clean=True,
        )
        for _ in range(2)
    ]
    lr_imgs = [
        MultiBandSkyImage(
            data=(rng.normal(size=(32, 32, 4)) * 100.0).astype(np.float32),
            pixel_scale_arcsec=0.10,
            band_names=Config.LR_INPUT_BAND_NAMES,
            is_clean=False,
        )
        for _ in range(2)
    ]
    write_multiband_skyimages(hr_imgs, "clean_train", records_dir=str(tmp_path))
    write_multiband_skyimages(lr_imgs, "dirty_train", records_dir=str(tmp_path))
    return str(tmp_path)


def test_multiband_dataset_yields_correct_shapes(tmp_path):
    rdir = _write_test_records(tmp_path)
    ds = MultiBandEuclidDataset(
        subset="train", records_dir=rdir, scale=2, hr_patch_size=16,
    ).dataset(batch_size=2, random_transform=True, repeat_count=1)
    lr, hr = next(iter(ds))
    assert lr.shape == (2, 8, 8, 4)
    assert hr.shape == (2, 16, 16, 1)


def test_multiband_dataset_without_augmentation(tmp_path):
    rdir = _write_test_records(tmp_path)
    ds = MultiBandEuclidDataset(
        subset="train", records_dir=rdir,
    ).dataset(batch_size=1, random_transform=False, repeat_count=1)
    lr, hr = next(iter(ds))
    # Full-size records (no crop)
    assert lr.shape == (1, 32, 32, 4)
    assert hr.shape == (1, 64, 64, 1)


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
    assert n > 1000   # not a degenerate model


def test_wdsr_runs_one_training_step(tmp_path):
    """A forward+backward pass updates trainable weights."""
    rdir = _write_test_records(tmp_path)
    ds = MultiBandEuclidDataset(
        subset="train", records_dir=rdir, scale=2, hr_patch_size=16,
    ).dataset(batch_size=2, repeat_count=1)
    model = wdsr(scale=2, nchan_in=4, nchan_out=1, num_res_blocks=2)
    opt = tf.keras.optimizers.Adam(1e-3)
    loss_fn = tf.keras.losses.MeanAbsoluteError()
    lr, hr = next(iter(ds))
    before = [v.numpy().copy() for v in model.trainable_variables[:3]]
    with tf.GradientTape() as tape:
        sr = model(lr)
        loss = loss_fn(hr, sr)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    after = [v.numpy() for v in model.trainable_variables[:3]]
    # At least one weight should have moved.
    assert any(not np.allclose(a, b) for a, b in zip(before, after))
