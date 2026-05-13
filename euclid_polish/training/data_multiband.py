"""
Multi-band data loader for training.

Reads v2 TFRecords produced by the new pipeline:

  * ``dirty_{subset}.tfrecord`` — (H_lr, W_lr, 4) LR float32 electrons,
    band order :attr:`Config.LR_INPUT_BAND_NAMES`.
  * ``hr_{subset}.tfrecord`` — (H_hr, W_hr, 1) HR float32 electrons,
    VIS-only target written by the forward step.
  * ``clean_{subset}.tfrecord`` — (H_hr, W_hr, 4) full 4-band HR clean
    record kept for inspection only (not used by the loader).

Each channel is asinh-stretched with its own per-band knee from
:attr:`BandConfig.asinh_stretch_scale_e` before the network sees it. The
per-band scale is a constant, so the stretch is applied in-graph via a
broadcast multiply — no Python loop per batch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

from euclid_polish.config import Config
from euclid_polish.sky.tfrecord import parse_record_graph_v2, tfrecord_path


# ---------------------------------------------------------------------------
# Per-band asinh stretch (graph constants)
# ---------------------------------------------------------------------------
#
# The stretch scales are immutable constants per-band. We materialise them
# as float32 numpy arrays at import time and rely on TF's auto-conversion
# from numpy → tensor inside ``tf.asinh`` / ``tf.sinh``. This is faster
# than building a fresh ``tf.constant`` each call AND avoids the
# lru_cache-across-test-fixtures pitfalls that would come from holding TF
# tensors in module-level caches.

_LR_STRETCH_SCALE_NP = np.array(
    [Config.get_band(name).asinh_stretch_scale_e
     for name in Config.LR_INPUT_BAND_NAMES],
    dtype=np.float32,
)  # shape (4,)

_HR_STRETCH_SCALE_NP = np.array(
    [Config.get_band(Config.HR_TARGET_BAND_NAME).asinh_stretch_scale_e],
    dtype=np.float32,
)  # shape (1,)


def _lr_stretch_scale() -> np.ndarray:
    """Length-4 vector of asinh stretch scales, one per LR channel."""
    return _LR_STRETCH_SCALE_NP


def _hr_stretch_scale() -> np.ndarray:
    """Length-1 scalar broadcasted against ``(B, H, W, 1)``."""
    return _HR_STRETCH_SCALE_NP


def asinh_stretch_lr(x: tf.Tensor) -> tf.Tensor:
    """asinh(x / k) per channel; ``x`` has shape ``(..., 4)``."""
    return tf.asinh(x / _lr_stretch_scale())


def asinh_stretch_hr(x: tf.Tensor) -> tf.Tensor:
    """asinh(x / k) for the VIS-only HR target; ``x`` has shape ``(..., 1)``."""
    return tf.asinh(x / _hr_stretch_scale())


def inverse_asinh_stretch_lr(y: tf.Tensor) -> tf.Tensor:
    """Inverse of :func:`asinh_stretch_lr` (per-band)."""
    return tf.sinh(y) * _lr_stretch_scale()


def inverse_asinh_stretch_hr(y: tf.Tensor) -> tf.Tensor:
    """Inverse of :func:`asinh_stretch_hr` (VIS only)."""
    return tf.sinh(y) * _hr_stretch_scale()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiBandEuclidDataset:
    """Reads paired v2 (clean HR-VIS, dirty LR-4channel) records.

    Parameters
    ----------
    subset        : ``'train'`` or ``'validate'``.
    records_dir   : directory containing ``clean_{subset}.tfrecord`` etc.
    scale         : super-resolution factor (HR pixel scale / LR pixel scale).
                    For our pipeline this is 2 (0.05″ HR / 0.10″ LR).
    hr_patch_size : HR crop size during training (96 by default).
    """

    def __init__(
        self,
        subset: str = "train",
        records_dir: str = Config.RECORDS_DIR_V2,
        scale: int = Config.DEFAULT_REBIN_FACTOR,
        hr_patch_size: int = Config.DEFAULT_HR_CROP_SIZE,
    ):
        if subset not in ("train", "validate"):
            raise ValueError("subset must be 'train' or 'validate'")
        self.scale         = int(scale)
        self.hr_patch_size = int(hr_patch_size)
        # HR target: 1-channel VIS written by the forward step. Falls
        # back to the older ``clean_{subset}`` name when the forward
        # output is the 1-band overwrite from the previous schema.
        hr_candidate = tfrecord_path(records_dir, f"hr_{subset}")
        legacy_clean = tfrecord_path(records_dir, f"clean_{subset}")
        import os as _os
        if _os.path.exists(hr_candidate):
            self.clean_file = hr_candidate
        else:
            self.clean_file = legacy_clean   # 1-band clean (old layout) or 4-band (new generate-only)
        self.dirty_file    = tfrecord_path(records_dir, f"dirty_{subset}")

    def dataset(
        self,
        batch_size: int = Config.DEFAULT_BATCH_SIZE,
        random_transform: bool = True,
        repeat_count: Optional[int] = None,
    ) -> tf.data.Dataset:
        """Build the (lr_4ch, hr_1ch) ``tf.data.Dataset``."""
        n_lr = Config.NUM_LR_CHANNELS
        n_hr = Config.NUM_HR_CHANNELS

        def _parse_lr(raw):
            return asinh_stretch_lr(parse_record_graph_v2(raw, n_lr))

        def _parse_hr(raw):
            return asinh_stretch_hr(parse_record_graph_v2(raw, n_hr))

        dirty_ds = tf.data.TFRecordDataset(self.dirty_file).map(
            _parse_lr, num_parallel_calls=AUTOTUNE,
        )
        clean_ds = tf.data.TFRecordDataset(self.clean_file).map(
            _parse_hr, num_parallel_calls=AUTOTUNE,
        )
        ds = tf.data.Dataset.zip((dirty_ds, clean_ds))
        ds = ds.cache()

        if random_transform:
            hr_patch = self.hr_patch_size
            scale    = self.scale
            ds = ds.shuffle(buffer_size=200)
            ds = ds.map(
                lambda lr, hr: _augment_multiband(lr, hr, hr_patch, scale),
                num_parallel_calls=AUTOTUNE,
            )

        ds = ds.repeat(repeat_count)
        return ds.batch(batch_size).prefetch(AUTOTUNE)


def _augment_multiband(
    lr: tf.Tensor, hr: tf.Tensor, hr_patch_size: int, scale: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Random aligned LR/HR crop.

    Flips and rotations are intentionally disabled: the empirical VIS
    ePSF is non-symmetric, so a flipped HR target is not what you would
    obtain by convolving the flipped clean field with the same PSF.
    """
    lr_patch_size = hr_patch_size // scale
    hr_h = tf.shape(hr)[0]
    hr_w = tf.shape(hr)[1]

    max_x = (hr_h - hr_patch_size) // scale * scale
    max_y = (hr_w - hr_patch_size) // scale * scale
    hr_x = tf.random.uniform([], 0, max_x + 1, dtype=tf.int32)
    hr_y = tf.random.uniform([], 0, max_y + 1, dtype=tf.int32)
    hr_x = hr_x // scale * scale
    hr_y = hr_y // scale * scale

    hr = hr[hr_x : hr_x + hr_patch_size, hr_y : hr_y + hr_patch_size, :]
    lr_x = hr_x // scale
    lr_y = hr_y // scale
    lr = lr[lr_x : lr_x + lr_patch_size, lr_y : lr_y + lr_patch_size, :]
    return lr, hr
