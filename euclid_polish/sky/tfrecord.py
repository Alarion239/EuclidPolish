"""
TFRecord I/O utilities for EuclidPolish sky images.
"""

from __future__ import annotations

import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import glob as _glob

from euclid_polish.config import Config
from euclid_polish.sky.types import SkyImage

# ---------------------------------------------------------------------------
# Sharded write helpers
# ---------------------------------------------------------------------------

def shard_paths(records_dir: str, name: str, n_shards: int) -> list[str]:
    """Return the list of shard file paths for a given split name.

    Example: shard_paths('./data/images/records', 'clean_train', 8)
    → ['./data/images/records/clean_train-00000-of-00008.tfrecord', ...]
    """
    return [
        os.path.join(records_dir, f"{name}-{i:05d}-of-{n_shards:05d}.tfrecord")
        for i in range(n_shards)
    ]


def open_shard_writers(paths: list[str]) -> list[tf.io.TFRecordWriter]:
    """Open one TFRecordWriter per shard path."""
    return [tf.io.TFRecordWriter(p) for p in paths]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_tfrecord_example(raw_record: bytes) -> tuple[np.ndarray, int, int, int]:
    """
    Eager parser — returns numpy values. Not safe inside tf.data.map().

    Returns
    -------
    image : ndarray, float32, shape (H, W)
    index : int
    height : int
    width : int
    """
    img = SkyImage.from_tfrecord(raw_record)
    return img.data, img.index or 0, *img.shape


def parse_record_graph(raw_record: tf.Tensor) -> tf.Tensor:
    """
    Graph-mode parser — safe inside tf.data.map().

    Returns a float32 tensor of shape [H, W, 1].
    """
    ex = tf.io.parse_single_example(raw_record, SkyImage._TFRECORD_FEATURES)
    pixels = tf.cast(tf.io.decode_raw(ex['image'], tf.float16), tf.float32)
    h = tf.cast(ex['height'], tf.int32)
    w = tf.cast(ex['width'],  tf.int32)
    return tf.reshape(pixels, [h, w, 1])


# ---------------------------------------------------------------------------
# Convenience reader (eager, for visualization / inspection)
# ---------------------------------------------------------------------------

def read_tfrecord(
    tfrecord_path: str,
    num_images: int = 5,
    mode: str = 'first',
    indices: list[int] | None = None,
    seed: int = 42,
) -> list[tuple[np.ndarray, int, int, int]]:
    """
    Read images from a TFRecord file (or glob pattern) and return numpy arrays.

    Parameters
    ----------
    tfrecord_path : str
        Path to a TFRecord file or a glob pattern matching multiple shards.
    num_images : int
        Number of images to return when mode is 'first' or 'random'.
    mode : str
        'first' or 'random'. Ignored when indices is provided.
    indices : list of int, optional
        Specific positional indices (0-based) to select.
    seed : int
        Random seed for reproducibility when mode='random'.

    Returns
    -------
    list of (image, index, height, width)
    """
    paths = sorted(_glob.glob(tfrecord_path)) or [tfrecord_path]
    dataset = tf.data.TFRecordDataset(paths)
    all_images = [
        parse_tfrecord_example(raw)
        for raw in tqdm(dataset, desc="Reading TFRecord")
    ]

    if indices is not None:
        valid = [i for i in indices if 0 <= i < len(all_images)]
        if len(valid) < len(indices):
            print(f"Warning: ignoring out-of-range indices {set(indices) - set(valid)}")
        return [all_images[i] for i in valid]

    n = min(num_images, len(all_images))
    if mode == 'first':
        return all_images[:n]
    if mode == 'random':
        np.random.seed(seed)
        chosen = np.random.choice(len(all_images), n, replace=False)
        return [all_images[i] for i in chosen]
    raise ValueError(f"mode must be 'first' or 'random', got {mode!r}")


# ---------------------------------------------------------------------------
# Batch I/O with SkyImage
# ---------------------------------------------------------------------------

def write_skyimages(
    images: list[SkyImage],
    name: str,
    n_shards: int,
    records_dir: str = Config.RECORDS_DIR,
) -> None:
    """Write SkyImage objects to sharded TFRecords with progress bar."""
    os.makedirs(records_dir, exist_ok=True)
    paths = shard_paths(records_dir, name, n_shards)
    writers = open_shard_writers(paths)
    try:
        for idx, img in enumerate(tqdm(images, desc=f"Writing {name}", unit="img")):
            writers[idx % n_shards].write(img.to_tfrecord(index=idx))
    finally:
        for w in writers:
            w.close()


def read_skyimages(
    tfrecord_path: str,
    num_images: int = 5,
    mode: str = 'first',
) -> list[SkyImage]:
    """Read TFRecords and return SkyImage objects.

    pixel_scale and is_clean are read from the stored records.
    """
    paths = sorted(_glob.glob(tfrecord_path)) or [tfrecord_path]
    dataset = tf.data.TFRecordDataset(paths)
    all_images = [SkyImage.from_tfrecord(raw) for raw in tqdm(dataset, desc="Reading TFRecord")]

    n = min(num_images, len(all_images))
    if mode == 'first':
        return all_images[:n]
    if mode == 'random':
        chosen = np.random.default_rng(42).choice(len(all_images), n, replace=False)
        return [all_images[i] for i in chosen]
    raise ValueError(f"mode must be 'first' or 'random', got {mode!r}")
