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


def _hr_scale_for(x: tf.Tensor, num_channels: int | None) -> np.ndarray:
    n = num_channels if num_channels is not None else x.shape[-1]
    k = _hr_stretch_scale()
    return k[:int(n)] if n is not None else k


def asinh_stretch_lr(x: tf.Tensor, knee: float | None = None) -> tf.Tensor:
    """asinh(x / k) per LR channel; ``x`` has shape ``(..., C)`` where C ≤ 4.

    ``knee`` (electrons) is a per-MEMBER diversity knob: when set, it replaces
    the per-band config scale with a single scalar uniformly across all bands
    (all bands share 100 e⁻ today, so a scalar override is exact). ``None`` →
    the per-band config default — the behavior every pre-knob member trained
    under, so those stay bit-identical.
    """
    if knee is not None:
        return tf.asinh(x / tf.cast(knee, x.dtype))
    c = x.shape[-1]
    k = _lr_stretch_scale()[:int(c)] if c is not None else _lr_stretch_scale()
    return tf.asinh(x / k)


def asinh_stretch_hr(x: tf.Tensor, num_channels: int | None = None,
                     knee: float | None = None) -> tf.Tensor:
    """asinh(x / k) per HR band; ``x`` has shape ``(..., C)``, C ≤ 4.

    ``knee`` (electrons): scalar per-member override — see
    :func:`asinh_stretch_lr`. ``None`` → per-band config default.
    """
    if knee is not None:
        return tf.asinh(x / tf.cast(knee, x.dtype))
    return tf.asinh(x / _hr_scale_for(x, num_channels))


def inverse_asinh_stretch_lr(y: tf.Tensor, knee: float | None = None) -> tf.Tensor:
    """Inverse of :func:`asinh_stretch_lr` (``knee`` scalar override, else per-band)."""
    if knee is not None:
        return tf.sinh(y) * tf.cast(knee, y.dtype)
    return tf.sinh(y) * _lr_stretch_scale()


def inverse_asinh_stretch_hr(y: tf.Tensor, num_channels: int | None = None,
                             knee: float | None = None) -> tf.Tensor:
    """Inverse of :func:`asinh_stretch_hr` (``knee`` scalar override, else per-band)."""
    if knee is not None:
        return tf.sinh(y) * tf.cast(knee, y.dtype)
    return tf.sinh(y) * _hr_scale_for(y, num_channels)


# ---------------------------------------------------------------------------
# Random crop augmentation
# ---------------------------------------------------------------------------

#: Per-LR-band read noise (e⁻) — the natural unit for the noise-augmentation
#: knob (one dimensionless multiplier covers all four bands sensibly).
_LR_READ_NOISE_NP = np.array(
    [Config.get_band(name).read_noise_e
     for name in Config.LR_INPUT_BAND_NAMES],
    dtype=np.float32,
)


def random_dihedral(lr: tf.Tensor, hr: tf.Tensor,
                    ) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply the SAME random dihedral transform (rot90 ×k + optional flip) to
    an aligned LR/HR pair — 8 orientations, uniform.

    Valid since data generation randomises the PSF orientation (PSF and
    galaxies are rotated together): ``flip(clean ⊛ PSF + n) = flip(clean) ⊛
    flip(PSF) + flip(n)``, so a flipped/rotated pair is an exact (dirty,
    clean) pair under the flipped PSF — within the training PSF distribution.
    The 2×2 rebin to the LR grid commutes with these transforms exactly
    (patches are square and block-aligned).
    """
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    lr = tf.image.rot90(lr, k)
    hr = tf.image.rot90(hr, k)
    flip = tf.random.uniform([], 0, 2, dtype=tf.int32)
    lr = tf.cond(flip > 0, lambda: tf.image.flip_left_right(lr), lambda: lr)
    hr = tf.cond(flip > 0, lambda: tf.image.flip_left_right(hr), lambda: hr)
    return lr, hr


def add_lr_noise(lr: tf.Tensor, noise_aug_rn: float) -> tf.Tensor:
    """Add extra Gaussian noise to an LR patch, in ELECTRONS (pre-stretch).

    ``noise_aug_rn`` is per-band: σ_band = noise_aug_rn · read_noise_band
    (VIS 3.6 e⁻, NISP 6.1 e⁻). This is noise-LEVEL augmentation on top of the
    realization already baked into the dirty records — it decorrelates
    ensemble members (each sees different draws, and different levels when
    the knob differs per member) at the cost of training at slightly higher
    noise than test. True noise re-realization needs a noiseless-LR record
    (TFRecord regen). 0 disables (identity).
    """
    if noise_aug_rn <= 0:
        return lr
    c = lr.shape[-1]
    k = _LR_READ_NOISE_NP[:int(c)] if c is not None else _LR_READ_NOISE_NP
    return lr + tf.random.normal(tf.shape(lr)) * (float(noise_aug_rn) * k)


def _augment_multiband(
    lr: tf.Tensor, hr: tf.Tensor, hr_patch_size: int, scale: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Random aligned LR/HR crop (dihedral flips/rotations live in
    :func:`random_dihedral`, applied by the pipeline after the crop so the
    transform runs on the small patch)."""
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
