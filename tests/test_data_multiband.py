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


def test_vis_only_slices_lr_to_single_channel(tmp_path):
    """``vis_only=True`` feeds 1 LR channel; HR target is unchanged, and the
    one channel is exactly VIS (index 0) of the full 4-channel stretch."""
    rdir = _write_test_records(tmp_path)
    full = MultiBandEuclidDataset(
        subset="train", records_dir=rdir,
    ).dataset(batch_size=1, random_transform=False, repeat_count=1)
    vis = MultiBandEuclidDataset(
        subset="train", records_dir=rdir, vis_only=True,
    ).dataset(batch_size=1, random_transform=False, repeat_count=1)
    lr4, hr4 = next(iter(full))
    lr1, hr1 = next(iter(vis))
    assert lr1.shape == (1, 32, 32, 1)        # VIS only
    assert hr1.shape == (1, 64, 64, 1)        # HR target unchanged
    # The retained channel is VIS (channel 0), keeping its own asinh knee.
    np.testing.assert_allclose(lr1[..., 0].numpy(), lr4[..., 0].numpy(),
                               rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 3-source mixing (synthetic + HST + star-anchor fixed layout)
# ---------------------------------------------------------------------------

def _write_anchor_records(tmp_path, subset: str = "train") -> str:
    """Write a star-anchor ``(dirty_anchor, hr_anchor)`` pair. ``hr`` is a
    sparse delta-target — zeros except one positive star pixel at the centre.
    Records are sized ≤ 2·patch−2 (HR 24, LR 12) so any aligned 16-crop keeps
    the centred star."""
    rng = np.random.default_rng(7)
    lr_imgs, hr_imgs = [], []
    for _ in range(4):
        lr = (rng.normal(size=(12, 12, 4)) * 100.0).astype(np.float32)
        hr = np.zeros((24, 24, 1), dtype=np.float32)
        hr[12, 12, 0] = 5.0e5                      # one star pixel at the centre
        lr_imgs.append(MultiBandSkyImage(
            data=lr, pixel_scale_arcsec=0.10,
            band_names=Config.LR_INPUT_BAND_NAMES, is_clean=False))
        hr_imgs.append(MultiBandSkyImage(
            data=hr, pixel_scale_arcsec=0.05, band_names=("VIS",), is_clean=True))
    write_multiband_skyimages(lr_imgs, f"dirty_anchor_{subset}", records_dir=str(tmp_path))
    write_multiband_skyimages(hr_imgs, f"hr_anchor_{subset}", records_dir=str(tmp_path))
    return str(tmp_path)


def test_dataset_default_is_2tuple(tmp_path):
    """``dataset()`` yields the pure-synthetic ``(lr, hr)`` 2-tuple — the
    pure-supervised path used by run_pipeline.py / cli / web inference and
    every validation stream (all destructure ``lr, hr = batch``)."""
    rdir = _write_test_records(tmp_path)
    ds = MultiBandEuclidDataset(
        subset="train", records_dir=rdir, scale=2, hr_patch_size=16,
    ).dataset(batch_size=2, random_transform=True, repeat_count=1)
    batch = next(iter(ds))
    assert len(batch) == 2
    lr, hr = batch
    assert lr.shape == (2, 8, 8, 4)
    assert hr.shape == (2, 16, 16, 1)


def test_fixed_layout_syn_only_shapes(tmp_path):
    """``dataset_fixed_layout`` with only a synthetic lane yields a single
    block of ``(lr, hr)`` 2-tuples at the requested count."""
    rdir = _write_test_records(tmp_path)
    ds = MultiBandEuclidDataset(
        subset="train", records_dir=rdir, scale=2, hr_patch_size=16,
    ).dataset_fixed_layout(2, 0, 0, random_transform=True)
    lr, hr = next(iter(ds))
    assert lr.shape == (2, 8, 8, 4)
    assert hr.shape == (2, 16, 16, 1)


def test_fixed_layout_three_way_counts(tmp_path):
    """A ``[n_syn | n_hst | n_anchor]`` layout produces a batch of exactly
    that size, contiguous in lane order, with the star-anchor block's HR
    slot carrying the sparse delta (≥1 positive pixel per anchor example)."""
    rdir    = _write_test_records(tmp_path)
    hst_dir = tmp_path / "hst_records"
    hst_dir.mkdir()
    _write_test_records(hst_dir)            # clean_train + dirty_train
    anchor_dir = tmp_path / "anchor_records"
    anchor_dir.mkdir()
    _write_anchor_records(anchor_dir, subset="train")

    n_syn, n_hst, n_anchor = 2, 1, 1
    ds = MultiBandEuclidDataset(
        subset="train", records_dir=rdir, scale=2, hr_patch_size=16,
        hst_records_dir=str(hst_dir), hst_fraction=0.25,
        anchor_records_dir=str(anchor_dir), anchor_fraction=0.25,
    ).dataset_fixed_layout(n_syn, n_hst, n_anchor, random_transform=True)

    lr, hr = next(iter(ds))
    B = n_syn + n_hst + n_anchor
    assert lr.shape == (B, 8, 8, 4)
    assert hr.shape == (B, 16, 16, 1)
    # The star-anchor lane is the last n_anchor rows; its HR delta survives
    # the crop (records are sized so the centred star is always kept).
    anchor_hr = hr.numpy()[n_syn + n_hst:]
    assert (anchor_hr > 0).any(), (
        "star-anchor HR delta was cropped out — the star pixel must survive"
    )


def test_fixed_layout_missing_hst_records_raises(tmp_path):
    """A fixed layout that requests an HST lane it wasn't given records
    for is an error — no silent single-source fallback."""
    rdir = _write_test_records(tmp_path)
    ds_obj = MultiBandEuclidDataset(
        subset="train", records_dir=rdir, scale=2, hr_patch_size=16,
    )
    with pytest.raises(ValueError, match="no HST records"):
        ds_obj.dataset_fixed_layout(2, 1, 0, random_transform=True)


def test_fixed_layout_missing_anchor_records_raises(tmp_path):
    """Likewise for a requested star-anchor lane with no anchor records."""
    rdir = _write_test_records(tmp_path)
    ds_obj = MultiBandEuclidDataset(
        subset="train", records_dir=rdir, scale=2, hr_patch_size=16,
    )
    with pytest.raises(ValueError, match="no star-anchor records"):
        ds_obj.dataset_fixed_layout(2, 0, 1, random_transform=True)


def test_fraction_sum_overflow_rejected():
    """``hst_fraction + anchor_fraction > 1`` is nonsense."""
    with pytest.raises(ValueError, match="must be ≤ 1"):
        MultiBandEuclidDataset(
            subset="train", records_dir="/tmp/does-not-matter",
            hst_fraction=0.6, anchor_fraction=0.5,
        )


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
