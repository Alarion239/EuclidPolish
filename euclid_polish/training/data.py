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
        hr_patch_size: int = Config.DEFAULT_HR_CROP_SIZE,
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
        # Deterministic shard ordering so clean/dirty stay aligned after zip
        clean_files = tf.data.Dataset.list_files(self.clean_glob, shuffle=False)
        dirty_files = tf.data.Dataset.list_files(self.dirty_glob, shuffle=False)

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

        # Cache decoded images in memory — avoids re-parsing TFRecords each epoch.
        # cache() must see the full dataset before repeat/take, otherwise TF
        # discards the partial cache with a warning on every pass.
        clean_ds = clean_ds.cache()
        dirty_ds = dirty_ds.cache()

        ds = tf.data.Dataset.zip((dirty_ds, clean_ds))  # (lr, hr)

        if random_transform:
            ds = ds.shuffle(buffer_size=200)
            hr_patch = self.hr_patch_size
            scale    = self.scale
            # Single map for all augmentations — avoids 3 separate AUTOTUNE thread pools
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
    """Random crop + flip + rotate in a single map call (less threading overhead)."""
    # --- crop ---
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

    # --- flip ---
    if tf.random.uniform(()) < 0.5:
        lr = tf.image.flip_left_right(lr)
        hr = tf.image.flip_left_right(hr)

    # --- rotate ---
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    return tf.image.rot90(lr, k), tf.image.rot90(hr, k)
