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
# File path helpers
# ---------------------------------------------------------------------------

def tfrecord_path(records_dir: str, name: str) -> str:
    """Return the path for a single TFRecord file.

    Example: tfrecord_path('./data/images/records', 'clean_train')
    → './data/images/records/clean_train.tfrecord'
    """
    return os.path.join(records_dir, f"{name}.tfrecord")


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
    pixels = tf.io.decode_raw(ex['image'], tf.float32)
    h = tf.cast(ex['height'], tf.int32)
    w = tf.cast(ex['width'],  tf.int32)
    return tf.reshape(pixels, [h, w, 1])


# ---------------------------------------------------------------------------
# Convenience reader (eager, for visualization / inspection)
# ---------------------------------------------------------------------------

def read_tfrecord(
    tfrecord_path_or_glob: str,
    num_images: int = 5,
    mode: str = 'first',
    indices: list[int] | None = None,
    seed: int = 42,
) -> list[tuple[np.ndarray, int, int, int]]:
    """
    Read images from a TFRecord file (or glob pattern) and return numpy arrays.

    Parameters
    ----------
    tfrecord_path_or_glob : str
        Path to a TFRecord file or a glob pattern.
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
    paths = sorted(_glob.glob(tfrecord_path_or_glob)) or [tfrecord_path_or_glob]
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
    records_dir: str = Config.RECORDS_DIR,
) -> str:
    """Write SkyImage objects to a single TFRecord file.

    Returns the path to the written file.
    """
    os.makedirs(records_dir, exist_ok=True)
    path = tfrecord_path(records_dir, name)
    writer = tf.io.TFRecordWriter(path)
    try:
        for idx, img in enumerate(tqdm(images, desc=f"Writing {name}", unit="img")):
            writer.write(img.to_tfrecord(index=idx))
    finally:
        writer.close()
    return path


def read_skyimages(
    tfrecord_path_or_glob: str,
    num_images: int = 5,
    mode: str = 'first',
) -> list[SkyImage]:
    """Read TFRecords and return SkyImage objects.

    pixel_scale and is_clean are read from the stored records.
    """
    paths = sorted(_glob.glob(tfrecord_path_or_glob)) or [tfrecord_path_or_glob]
    dataset = tf.data.TFRecordDataset(paths)
    all_images = [SkyImage.from_tfrecord(raw) for raw in tqdm(dataset, desc="Reading TFRecord")]

    n = min(num_images, len(all_images))
    if mode == 'first':
        return all_images[:n]
    if mode == 'random':
        chosen = np.random.default_rng(42).choice(len(all_images), n, replace=False)
        return [all_images[i] for i in chosen]
    raise ValueError(f"mode must be 'first' or 'random', got {mode!r}")
