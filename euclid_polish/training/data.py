"""
Data loader module for training.

This module provides EuclidDataset for loading paired clean/dirty TFRecord data.
"""

import os
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

from euclid_polish.config import Config
from euclid_polish.sky.tfrecord import parse_record_graph


class EuclidDataset:
    """
    Data loader for Euclid super-resolution training.

    Reads paired clean (HR) and dirty (LR) images from sharded TFRecords under
    Config.RECORDS_DIR and returns a tf.data.Dataset of (lr_patch, hr_patch) pairs.
    """

    def __init__(
        self,
        subset: str = 'train',
        records_dir: str = Config.RECORDS_DIR,
        scale: int = 4,
        hr_patch_size: int = 256,
    ):
        """
        Parameters
        ----------
        subset : str
            'train' or 'validate'.
        records_dir : str
            Directory containing sharded TFRecord files.
        scale : int
            Super-resolution scale factor (hr_patch_size // scale = lr_patch_size).
        hr_patch_size : int
            Spatial size of HR patches used during training.
        """
        if subset not in ('train', 'validate'):
            raise ValueError("subset must be 'train' or 'validate'")
        self.scale         = scale
        self.hr_patch_size = hr_patch_size
        self.clean_glob    = os.path.join(records_dir, f'clean_{subset}-*.tfrecord')
        self.dirty_glob    = os.path.join(records_dir, f'dirty_{subset}-*.tfrecord')

    def dataset(
        self,
        batch_size: int = 16,
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
        clean_files = tf.data.Dataset.list_files(self.clean_glob, shuffle=random_transform)
        dirty_files = tf.data.Dataset.list_files(self.dirty_glob, shuffle=random_transform)

        # Parallel shard reads — the primary TFRecord speedup
        clean_ds = clean_files.interleave(
            tf.data.TFRecordDataset,
            cycle_length=AUTOTUNE,
            num_parallel_calls=AUTOTUNE,
        ).map(parse_record_graph, num_parallel_calls=AUTOTUNE)

        dirty_ds = dirty_files.interleave(
            tf.data.TFRecordDataset,
            cycle_length=AUTOTUNE,
            num_parallel_calls=AUTOTUNE,
        ).map(parse_record_graph, num_parallel_calls=AUTOTUNE)

        ds = tf.data.Dataset.zip((dirty_ds, clean_ds))  # (lr, hr)

        if random_transform:
            hr_patch = self.hr_patch_size
            scale    = self.scale
            ds = ds.map(
                lambda lr, hr: _random_crop(lr, hr, hr_patch, scale),
                num_parallel_calls=AUTOTUNE,
            )
            ds = ds.map(_random_flip,   num_parallel_calls=AUTOTUNE)
            ds = ds.map(_random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.shuffle(buffer_size=200)

        ds = ds.repeat(repeat_count)
        return ds.batch(batch_size).prefetch(AUTOTUNE)


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _random_crop(
    lr: tf.Tensor,
    hr: tf.Tensor,
    hr_patch_size: int,
    scale: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Crop aligned patches from LR and HR images."""
    lr_patch_size = hr_patch_size // scale
    hr_h = tf.shape(hr)[0]
    hr_w = tf.shape(hr)[1]

    # Choose top-left in HR space, snapped to scale grid
    max_x = (hr_h - hr_patch_size) // scale * scale
    max_y = (hr_w - hr_patch_size) // scale * scale
    hr_x  = tf.random.uniform([], 0, max_x + 1, dtype=tf.int32)
    hr_y  = tf.random.uniform([], 0, max_y + 1, dtype=tf.int32)
    hr_x  = hr_x // scale * scale
    hr_y  = hr_y // scale * scale

    hr_patch = hr[hr_x : hr_x + hr_patch_size, hr_y : hr_y + hr_patch_size, :]
    lr_x     = hr_x // scale
    lr_y     = hr_y // scale
    lr_patch = lr[lr_x : lr_x + lr_patch_size, lr_y : lr_y + lr_patch_size, :]
    return lr_patch, hr_patch


def _random_flip(lr: tf.Tensor, hr: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Random left-right flip applied identically to both images."""
    if tf.random.uniform(()) < 0.5:
        lr = tf.image.flip_left_right(lr)
        hr = tf.image.flip_left_right(hr)
    return lr, hr


def _random_rotate(lr: tf.Tensor, hr: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Random rotation by 0 / 90 / 180 / 270° applied identically to both images."""
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    return tf.image.rot90(lr, k), tf.image.rot90(hr, k)
