"""Per-band asinh stretch and random-crop augmentation for the training pipeline.

These are pure TF graph functions used by :meth:`Model._build_training_pipeline`.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

from euclid_polish.config import Config
from euclid_polish.image.tfio import parse_example


# ---------------------------------------------------------------------------
# Per-band asinh stretch
# ---------------------------------------------------------------------------

_LR_STRETCH_SCALE_NP = np.array(
    [Config.get_band(name).asinh_stretch_scale_e
     for name in Config.LR_INPUT_BAND_NAMES],
    dtype=np.float32,
)

_HR_STRETCH_SCALE_NP = np.array(
    [Config.get_band(name).asinh_stretch_scale_e
     for name in Config.HR_TARGET_BAND_NAMES],
    dtype=np.float32,
)


def _lr_stretch_scale() -> np.ndarray:
    return _LR_STRETCH_SCALE_NP


def _hr_stretch_scale() -> np.ndarray:
    return _HR_STRETCH_SCALE_NP


def _hr_scale_for(x: tf.Tensor, num_channels: "int | None") -> np.ndarray:
    n = num_channels if num_channels is not None else x.shape[-1]
    k = _hr_stretch_scale()
    return k[:int(n)] if n is not None else k


def asinh_stretch_lr(x: tf.Tensor) -> tf.Tensor:
    """asinh(x / k) per LR channel; ``x`` has shape ``(..., 4)``."""
    return tf.asinh(x / _lr_stretch_scale())


def asinh_stretch_hr(x: tf.Tensor, num_channels: "int | None" = None) -> tf.Tensor:
    """asinh(x / k) per HR band; ``x`` has shape ``(..., C)``, C ≤ 4."""
    return tf.asinh(x / _hr_scale_for(x, num_channels))


def inverse_asinh_stretch_lr(y: tf.Tensor) -> tf.Tensor:
    """Inverse of :func:`asinh_stretch_lr`."""
    return tf.sinh(y) * _lr_stretch_scale()


def inverse_asinh_stretch_hr(y: tf.Tensor, num_channels: "int | None" = None) -> tf.Tensor:
    """Inverse of :func:`asinh_stretch_hr`."""
    return tf.sinh(y) * _hr_scale_for(y, num_channels)


# ---------------------------------------------------------------------------
# Random crop augmentation
# ---------------------------------------------------------------------------

def _augment_multiband(
    lr: tf.Tensor, hr: tf.Tensor, hr_patch_size: int, scale: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Random aligned LR/HR crop.

    Flips and rotations are intentionally disabled: the empirical VIS ePSF is
    non-symmetric, so a flipped HR target is not what you would obtain by
    convolving the flipped clean field with the same PSF.
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

    hr = hr[hr_x: hr_x + hr_patch_size, hr_y: hr_y + hr_patch_size, :]
    lr_x = hr_x // scale
    lr_y = hr_y // scale
    lr = lr[lr_x: lr_x + lr_patch_size, lr_y: lr_y + lr_patch_size, :]
    return lr, hr


# ---------------------------------------------------------------------------
# LR-only streaming dataset (no HR side)
# ---------------------------------------------------------------------------

def lr_only_dataset(dirty_path: str, *, batch_size: int) -> tf.data.Dataset:
    """Streaming LR-only dataset from a ``dirty_{subset}.tfrecord``.

    Applies the same per-band asinh stretch as the training path.
    Yields LR tensors of shape ``[B, H, W, 4]``.
    """
    n_lr = Config.NUM_LR_CHANNELS

    def _parse(raw):
        return asinh_stretch_lr(parse_example(raw, n_lr))

    return (tf.data.TFRecordDataset(dirty_path)
            .map(_parse, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE))
