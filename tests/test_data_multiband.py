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
# Source-tag emission + 3-source mixing (round-trip integration)
# ---------------------------------------------------------------------------

def _write_test_dirty_only(tmp_path, subset: str = "train") -> str:
    """Write *only* the dirty TFRecord — used to fake a round-trip records dir."""
    rng = np.random.default_rng(2025)
    imgs = [
        MultiBandSkyImage(
            data=(rng.normal(size=(32, 32, 4)) * 100.0).astype(np.float32),
            pixel_scale_arcsec=0.10,
            band_names=Config.LR_INPUT_BAND_NAMES,
            is_clean=False,
        )
        for _ in range(4)
    ]
    write_multiband_skyimages(imgs, f"dirty_{subset}", records_dir=str(tmp_path))
    return str(tmp_path)


def test_dataset_emits_source_tag_when_requested(tmp_path):
    """``with_source_tag=True`` switches to (lr, hr, source) 3-tuples."""
    from euclid_polish.training.data_multiband import SOURCE_SYNTHETIC

    rdir = _write_test_records(tmp_path)
    ds = MultiBandEuclidDataset(
        subset="train", records_dir=rdir, scale=2, hr_patch_size=16,
    ).dataset(
        batch_size=2, random_transform=True, repeat_count=1,
        with_source_tag=True,
    )
    batch = next(iter(ds))
    assert len(batch) == 3, "expected (lr, hr, source) 3-tuple"
    lr, hr, src = batch
    assert lr.shape == (2, 8, 8, 4)
    assert hr.shape == (2, 16, 16, 1)
    assert src.shape == (2,)
    assert src.dtype == tf.int32
    # All examples are from the synthetic source — no HST or round-trip
    # configured in this test.
    assert (src.numpy() == SOURCE_SYNTHETIC).all()


def test_dataset_2tuple_default_preserves_backward_compat(tmp_path):
    """The default ``with_source_tag=False`` keeps the old 2-tuple API.

    Pinned because many existing call sites (scripts/run_pipeline.py,
    web/app.py, cli/main.py) destructure ``lr, hr = batch``.
    """
    rdir = _write_test_records(tmp_path)
    ds = MultiBandEuclidDataset(
        subset="train", records_dir=rdir, scale=2, hr_patch_size=16,
    ).dataset(batch_size=2, random_transform=True, repeat_count=1)
    batch = next(iter(ds))
    assert len(batch) == 2


def test_roundtrip_fraction_requires_source_tag(tmp_path):
    """Mixing round-trip data without source tags would silently treat
    the dummy zeros HR as ground truth — refuse the configuration."""
    rdir = _write_test_records(tmp_path)
    rt_dir = tmp_path / "rt_records"
    _write_test_dirty_only(rt_dir, subset="train")
    ds_obj = MultiBandEuclidDataset(
        subset="train", records_dir=rdir, scale=2, hr_patch_size=16,
        roundtrip_records_dir=str(rt_dir), roundtrip_fraction=0.2,
    )
    with pytest.raises(ValueError, match="with_source_tag=True"):
        ds_obj.dataset(batch_size=2, random_transform=True, repeat_count=1)


def test_fraction_sum_overflow_rejected():
    """``hst_fraction + roundtrip_fraction > 1`` is nonsense."""
    with pytest.raises(ValueError, match="must be ≤ 1"):
        MultiBandEuclidDataset(
            subset="train", records_dir="/tmp/does-not-matter",
            hst_fraction=0.6, roundtrip_fraction=0.5,
        )


def test_three_way_mix_distribution(tmp_path):
    """Per-example source distribution matches the configured weights.

    With 100 sampled examples and a 60/20/20 split the per-source
    count should cluster near the expected mean; we use loose
    tolerances so the test isn't flaky.
    """
    from euclid_polish.training.data_multiband import (
        SOURCE_SYNTHETIC, SOURCE_HST, SOURCE_ROUNDTRIP,
    )

    rdir    = _write_test_records(tmp_path)
    hst_dir = tmp_path / "hst_records"
    hst_dir.mkdir()
    # HST stream needs (clean + dirty) — reuse the same writer.
    _ = _write_test_records(hst_dir)   # writes clean_train + dirty_train

    rt_dir = tmp_path / "rt_records"
    _write_test_dirty_only(rt_dir, subset="train")

    ds = MultiBandEuclidDataset(
        subset="train", records_dir=rdir, scale=2, hr_patch_size=16,
        hst_records_dir=str(hst_dir), hst_fraction=0.20,
        roundtrip_records_dir=str(rt_dir), roundtrip_fraction=0.20,
    ).dataset(
        batch_size=1, random_transform=True, repeat_count=None,
        with_source_tag=True,
    )

    sources = []
    for i, batch in enumerate(ds):
        if i >= 200: break
        _, _, src = batch
        sources.extend(src.numpy().tolist())

    n = len(sources)
    n_syn = sources.count(SOURCE_SYNTHETIC)
    n_hst = sources.count(SOURCE_HST)
    n_rt  = sources.count(SOURCE_ROUNDTRIP)
    assert n_syn + n_hst + n_rt == n, "every example must carry a known source tag"
    # Sample sizes are small → wide tolerances. Just verify each source
    # contributes non-trivially and synthetic dominates.
    assert n_syn > n_hst, (
        f"synthetic (weight 0.6) should dominate HST (0.2); got "
        f"syn={n_syn} hst={n_hst}"
    )
    assert n_syn > n_rt,  (
        f"synthetic (weight 0.6) should dominate round-trip (0.2); got "
        f"syn={n_syn} rt={n_rt}"
    )
    assert n_hst > 0,     "HST source missing entirely"
    assert n_rt  > 0,     "round-trip source missing entirely"


def test_roundtrip_stream_dummy_hr_is_zero(tmp_path):
    """The HR slot for round-trip examples must be zeros — the trainer
    detects round-trip via the source tag but a non-zero dummy would
    pollute any (incorrectly applied) supervised L1."""
    from euclid_polish.training.data_multiband import SOURCE_ROUNDTRIP

    rdir    = _write_test_records(tmp_path)
    rt_dir  = tmp_path / "rt_records"
    _write_test_dirty_only(rt_dir, subset="train")

    # roundtrip_fraction=1.0 would technically be valid but we keep
    # synthetic at 1e-6 weight so we always see round-trip batches.
    ds = MultiBandEuclidDataset(
        subset="train", records_dir=rdir, scale=2, hr_patch_size=16,
        roundtrip_records_dir=str(rt_dir), roundtrip_fraction=0.99,
    ).dataset(
        batch_size=2, random_transform=True, repeat_count=None,
        with_source_tag=True,
    )

    saw_roundtrip = False
    for i, (lr, hr, src) in enumerate(ds):
        if i >= 20: break
        rt_mask = src.numpy() == SOURCE_ROUNDTRIP
        if not rt_mask.any():
            continue
        saw_roundtrip = True
        # Every round-trip example's HR slot is exact zeros (the
        # trainer ignores it, but it has to be zero to keep accidental
        # supervised paths well-defined).
        rt_hr = hr.numpy()[rt_mask]
        assert (rt_hr == 0.0).all(), (
            f"round-trip dummy HR contained non-zero values: "
            f"max abs = {np.abs(rt_hr).max()}"
        )
    assert saw_roundtrip, (
        "no round-trip examples observed across 20 batches even at "
        "fraction 0.99 — sampling may be broken"
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
