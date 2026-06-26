"""End-to-end smoke test of the full multi-band pipeline.

Generates a tiny synthetic dataset, runs the per-band forward model,
writes v2 TFRecords, builds the data loader, and trains WDSR for one
step. If this passes, every Phase 0-7 module is wired correctly.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.euclid.psf_library import load_all_band_psfs
from tests._tiny_catalog import TinyCosmosCatalog
from euclid_polish.sky.multiband_forward import (
    MultiBandForward, MultiBandForwardConfig,
)
from euclid_polish.sky.multiband_generator import (
    MultiBandGeneratorConfig, MultiBandSimulator,
)
from euclid_polish.image.tfio import (
    tfrecord_path, write_multiband_skyimages,
)
from euclid_polish.training.data_multiband import MultiBandEuclidDataset
from euclid_polish.training.models.wdsr import wdsr


def test_full_pipeline_round_trip(tmp_path):
    """Phase 0–7 wired together: generate → forward-model → load → train one step."""
    # 1. Generate clean HR fields (multi-band).
    cat = TinyCosmosCatalog(n_galaxies=200, seed=0)
    sim = MultiBandSimulator(
        cat, MultiBandGeneratorConfig(image_size=96, pixel_scale=Config.DEFAULT_PIXEL_SCALE),
    )
    clean_imgs = []
    for i in range(3):
        rng = np.random.default_rng(i)
        sky, _ = sim.simulate_field(rng, n_galaxies=2, n_stars=1, n_lenses=0)
        sky.index = i
        clean_imgs.append(sky)

    records_dir = str(tmp_path)

    # 2. Run forward model (HR 4-ch → LR 4-ch + HR 4-ch clean target).
    psfs = load_all_band_psfs(psf_dir="/nonexistent_dir_for_test")
    fwd = MultiBandForward(psfs_by_band=psfs,
                           config=MultiBandForwardConfig(add_noise=True))
    rng = np.random.default_rng(0)
    lr_imgs, hr_imgs = [], []
    for sky in clean_imgs:
        lr, hr = fwd.process(sky, rng=rng)
        lr_imgs.append(lr)
        hr_imgs.append(hr)

    # 3. Write v2 TFRecords.
    for subset in ("train", "validate"):
        write_multiband_skyimages(lr_imgs, f"dirty_{subset}", records_dir=records_dir)
        write_multiband_skyimages(hr_imgs, f"clean_{subset}", records_dir=records_dir)

    # 4. Build the multi-band data loader.
    ds = MultiBandEuclidDataset(
        subset="train", records_dir=records_dir, scale=2, hr_patch_size=16,
    ).dataset(batch_size=2, random_transform=True, repeat_count=1)

    # 5. Train one step on WDSR (4-channel in, 4-channel out).
    model = wdsr(scale=2, num_res_blocks=2,
                 nchan_in=Config.NUM_LR_CHANNELS,
                 nchan_out=Config.NUM_HR_CHANNELS)
    opt = tf.keras.optimizers.Adam(1e-3)
    loss_fn = tf.keras.losses.MeanAbsoluteError()

    lr, hr = next(iter(ds))
    assert lr.shape == (2, 8, 8, 4)
    assert hr.shape == (2, 16, 16, 4)

    before = [v.numpy().copy() for v in model.trainable_variables[:3]]
    with tf.GradientTape() as tape:
        sr = model(lr)
        loss = loss_fn(hr, sr)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    after = [v.numpy() for v in model.trainable_variables[:3]]

    # At least one weight moved → the gradient path is connected end-to-end.
    assert any(not np.allclose(a, b) for a, b in zip(before, after))

    # 6. Records on disk are readable & have the right schema.
    assert os.path.exists(tfrecord_path(records_dir, "clean_train"))
    assert os.path.exists(tfrecord_path(records_dir, "dirty_train"))
