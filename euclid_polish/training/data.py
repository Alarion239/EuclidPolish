"""
Data loader module for training.

This module provides EuclidDataset for loading paired clean/dirty TFRecord data.
"""

import os
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

from euclid_polish.config import Config
from euclid_polish.sky.tfrecord import parse_record_graph, tfrecord_path
from euclid_polish.training.models.common import normalize


class EuclidDataset:
    """
    Data loader for Euclid super-resolution training.

    Reads paired clean (HR) and dirty (LR) images from TFRecords under
    Config.RECORDS_DIR and returns a tf.data.Dataset of (lr_patch, hr_patch) pairs.
    """

    def __init__(
        self,
        subset: str = 'train',
        records_dir: str = Config.RECORDS_DIR,
        scale: int = Config.DEFAULT_REBIN_FACTOR,
        hr_patch_size: int = Config.DEFAULT_HR_CROP_SIZE,
    ):
        """
        Parameters
        ----------
        subset : str
            'train' or 'validate'.
        records_dir : str
            Directory containing TFRecord files.
        scale : int
            Super-resolution scale factor (hr_patch_size // scale = lr_patch_size).
        hr_patch_size : int
            Spatial size of HR patches used during training.
        """
        if subset not in ('train', 'validate'):
            raise ValueError("subset must be 'train' or 'validate'")
        self.scale         = scale
        self.hr_patch_size = hr_patch_size
        self.clean_file    = tfrecord_path(records_dir, f'clean_{subset}')
        self.dirty_file    = tfrecord_path(records_dir, f'dirty_{subset}')

    def dataset(
        self,
        batch_size: int = Config.DEFAULT_BATCH_SIZE,
        random_transform: bool = True,
        repeat_count: int | None = None,
    ) -> tf.data.Dataset:
        """
        Build and return the tf.data.Dataset.

        Parameters
        ----------
        batch_size : int
            Number of (lr, hr) pairs per batch.
        random_transform : bool
            Apply random crop, flip, and rotation (set False for validation).
        repeat_count : int or None
            Times to repeat; None repeats indefinitely.

        Returns
        -------
        tf.data.Dataset yielding (lr_patch, hr_patch) float32 tensors.
        """
        clean_ds = tf.data.TFRecordDataset(self.clean_file).map(
            parse_record_graph, num_parallel_calls=AUTOTUNE,
        )
        dirty_ds = tf.data.TFRecordDataset(self.dirty_file).map(
            parse_record_graph, num_parallel_calls=AUTOTUNE,
        )

        # Per-image min-max normalization to [0, 1].
        # TFRecords store raw flux values; we normalize here so the model
        # always sees [0, 1] input regardless of the original dynamic range.
        clean_ds = clean_ds.map(normalize, num_parallel_calls=AUTOTUNE)
        dirty_ds = dirty_ds.map(normalize, num_parallel_calls=AUTOTUNE)

        # Cache after normalization so it's computed once
        clean_ds = clean_ds.cache()
        dirty_ds = dirty_ds.cache()

        ds = tf.data.Dataset.zip((dirty_ds, clean_ds))  # (lr, hr)

        if random_transform:
            ds = ds.shuffle(buffer_size=200)
            hr_patch = self.hr_patch_size
            scale    = self.scale
            ds = ds.map(
                lambda lr, hr: _augment(lr, hr, hr_patch, scale),
                num_parallel_calls=AUTOTUNE,
            )

        ds = ds.repeat(repeat_count)
        return ds.batch(batch_size).prefetch(AUTOTUNE)


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _augment(
    lr: tf.Tensor,
    hr: tf.Tensor,
    hr_patch_size: int,
    scale: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Random crop only. Rotation and flip are disabled because the PSF is
    non-symmetric — rotating a (LR, HR) pair would break the LR↔HR
    correspondence (the rotated LR is not what you'd get by convolving the
    rotated HR with the original PSF)."""
    lr_patch_size = hr_patch_size // scale
    hr_h = tf.shape(hr)[0]
    hr_w = tf.shape(hr)[1]

    max_x = (hr_h - hr_patch_size) // scale * scale
    max_y = (hr_w - hr_patch_size) // scale * scale
    hr_x  = tf.random.uniform([], 0, max_x + 1, dtype=tf.int32)
    hr_y  = tf.random.uniform([], 0, max_y + 1, dtype=tf.int32)
    hr_x  = hr_x // scale * scale
    hr_y  = hr_y // scale * scale

    hr = hr[hr_x : hr_x + hr_patch_size, hr_y : hr_y + hr_patch_size, :]
    lr_x = hr_x // scale
    lr_y = hr_y // scale
    lr = lr[lr_x : lr_x + lr_patch_size, lr_y : lr_y + lr_patch_size, :]

    return lr, hr
