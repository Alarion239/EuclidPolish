"""TFRecord I/O for the image stack.

This is the persistence layer an :class:`~euclid_polish.image.collection.ImageSet`
is built on. It depends only on tensorflow + the :class:`Image` atom, so the
import direction stays ``sky/`` → ``image/`` (never the reverse).

These functions were historically in ``euclid_polish.sky.tfrecord``; that module
has been removed and call sites now import from ``euclid_polish.image.tfio``.
"""

from __future__ import annotations

import contextlib
import glob as _glob
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from euclid_polish.config import Config
from euclid_polish.image.core import Image


def tfrecord_path(records_dir: str, name: str) -> str:
    """Path for a single TFRecord file: ``<records_dir>/<name>.tfrecord``."""
    return os.path.join(records_dir, f"{name}.tfrecord")


def parse_record_graph_v2(raw_record: tf.Tensor, num_channels: int) -> tf.Tensor:
    """Graph-mode parser — safe inside ``tf.data.map``.

    Returns a float32 tensor ``[H, W, num_channels]``. The caller pre-commits to
    ``num_channels`` so the output shape is statically known downstream.
    """
    ex = tf.io.parse_single_example(raw_record, Image._TFRECORD_FEATURES)
    pixels = tf.io.decode_raw(ex['image'], tf.float32)
    h = tf.cast(ex['height'], tf.int32)
    w = tf.cast(ex['width'], tf.int32)
    c = tf.cast(ex['channels'], tf.int32)
    tf.debugging.assert_equal(
        c, tf.constant(int(num_channels), dtype=tf.int32),
        message="Channel count in TFRecord does not match expected num_channels.")
    return tf.reshape(pixels, [h, w, num_channels])


def write_multiband_skyimages(
    images: "list[Image]",
    name: str,
    records_dir: str = Config.RECORDS_DIR_V2,
) -> str:
    """Write ``Image`` objects to a single TFRecord file. Returns its path.

    Materialises the whole list in RAM; for large datasets prefer
    :func:`open_multiband_writer` and stream one image at a time.
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


class _MultiBandStreamingWriter:
    """Streaming-write side of :func:`open_multiband_writer` (auto-index)."""

    def __init__(self, writer: tf.io.TFRecordWriter, path: str) -> None:
        self._writer = writer
        self._path = path
        self._count = 0

    def write(self, img: "Image", index: int | None = None) -> None:
        if index is None:
            index = self._count
        self._writer.write(img.to_tfrecord(index=index))
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    @property
    def path(self) -> str:
        return self._path


@contextlib.contextmanager
def open_multiband_writer(name: str, records_dir: str = Config.RECORDS_DIR_V2):
    """Context manager for streaming ``Image`` writes (memory scales with one image)."""
    os.makedirs(records_dir, exist_ok=True)
    path = tfrecord_path(records_dir, name)
    writer = tf.io.TFRecordWriter(path)
    handle = _MultiBandStreamingWriter(writer, path)
    try:
        yield handle
    finally:
        writer.close()


def read_multiband_skyimages(
    tfrecord_path_or_glob: str,
    num_images: int = 5,
    mode: str = 'first',
) -> "list[Image]":
    """Read TFRecords and return ``Image`` objects (``mode``: 'first' | 'random')."""
    paths = sorted(_glob.glob(tfrecord_path_or_glob)) or [tfrecord_path_or_glob]
    dataset = tf.data.TFRecordDataset(paths)
    all_images = [
        Image.from_tfrecord(raw)
        for raw in tqdm(dataset, desc="Reading TFRecord")
    ]
    n = min(num_images, len(all_images))
    if mode == 'first':
        return all_images[:n]
    if mode == 'random':
        chosen = np.random.default_rng(42).choice(len(all_images), n, replace=False)
        return [all_images[i] for i in chosen]
    raise ValueError(f"mode must be 'first' or 'random', got {mode!r}")
