"""Aligned TensorFlow input pipelines for live and fixed experiment pairs."""

from __future__ import annotations

import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.image.tfio import parse_example, tfrecord_path
from euclid_polish.training.augmentation import (
    asinh_stretch_hr,
    asinh_stretch_lr,
    random_dihedral,
)

AUTOTUNE = tf.data.AUTOTUNE


def build_live_dataset(
    records_dir: str,
    forward,
    *,
    batch_size: int,
    shuffle_buffer: int = 256,
) -> tf.data.Dataset:
    """Read paired clean layers and invoke the live full-field forward."""
    scene_ds = tf.data.TFRecordDataset(tfrecord_path(records_dir, "scene_train"))
    lens_ds = tf.data.TFRecordDataset(tfrecord_path(records_dir, "lens_train"))

    def parse(scene_raw, lens_raw):
        return (
            parse_example(scene_raw, Config.NUM_LR_CHANNELS),
            parse_example(lens_raw, Config.NUM_LR_CHANNELS),
        )

    def live(scene, lens):
        lr, target = tf.numpy_function(forward.crops, [scene, lens], Tout=(tf.float32, tf.float32))
        c = forward.hr_crop_size
        k = forward.crops_per_field
        lr.set_shape((k, c // forward.scale, c // forward.scale, Config.NUM_LR_CHANNELS))
        target.set_shape((k, c, c, Config.NUM_LR_CHANNELS))
        return lr, target

    def transform(lr, target):
        lr, target = random_dihedral(lr, target)
        return asinh_stretch_lr(lr), asinh_stretch_hr(target)

    return (
        tf.data.Dataset.zip((scene_ds, lens_ds))
        .map(parse, num_parallel_calls=AUTOTUNE)
        .map(live, num_parallel_calls=AUTOTUNE)
        .unbatch()
        .shuffle(shuffle_buffer)
        .map(transform, num_parallel_calls=AUTOTUNE)
        .repeat()
        .batch(batch_size, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )


def build_fixed_dataset(
    records_dir: str,
    subset: str,
    *,
    batch_size: int,
) -> tf.data.Dataset:
    """Read the deterministic dirty/lens validation or test pairing."""
    if subset not in {"validate", "test"}:
        raise ValueError("fixed dataset subset must be validate or test")
    dirty_ds = tf.data.TFRecordDataset(tfrecord_path(records_dir, f"dirty_{subset}"))
    lens_ds = tf.data.TFRecordDataset(tfrecord_path(records_dir, f"lens_{subset}"))

    def parse(dirty_raw, lens_raw):
        dirty = parse_example(dirty_raw, Config.NUM_LR_CHANNELS)
        lens = parse_example(lens_raw, Config.NUM_LR_CHANNELS)
        return asinh_stretch_lr(dirty), asinh_stretch_hr(lens)

    return (
        tf.data.Dataset.zip((dirty_ds, lens_ds))
        .map(parse, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
