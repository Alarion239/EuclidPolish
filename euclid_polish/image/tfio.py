"""TFRecord I/O for the image stack.

The persistence layer an :class:`~euclid_polish.image.collection.ImageSet` is
built on. Depends only on tensorflow + the :class:`Image` atom, so the import
direction stays ``sky/`` → ``image/`` (never the reverse).
"""

from __future__ import annotations

import contextlib
import glob as _glob
import os

import tensorflow as tf
from tqdm import tqdm

from euclid_polish.config import Config
from euclid_polish.image.core import Image


def tfrecord_path(records_dir: str, name: str) -> str:
    """Path for a single TFRecord file: ``<records_dir>/<name>.tfrecord``."""
    return os.path.join(records_dir, f"{name}.tfrecord")


# Minimal feature subset for the hot training path: only the pixels + dims are
# needed to build the input tensor, so the graph never decodes the
# band/role/provenance strings on every record.
_GRAPH_FEATURES = {
    'image':    tf.io.FixedLenFeature([], tf.string),
    'height':   tf.io.FixedLenFeature([], tf.int64),
    'width':    tf.io.FixedLenFeature([], tf.int64),
    'channels': tf.io.FixedLenFeature([], tf.int64),
}


def parse_example(raw_record: tf.Tensor, num_channels: int) -> tf.Tensor:
    """Graph-mode parser — safe inside ``tf.data.map``.

    Returns a float32 tensor ``[H, W, num_channels]``. The caller pre-commits to
    ``num_channels`` so the output shape is statically known downstream; a
    mismatch aborts the graph.
    """
    ex = tf.io.parse_single_example(raw_record, _GRAPH_FEATURES)
    pixels = tf.io.decode_raw(ex['image'], tf.float32)
    h = tf.cast(ex['height'], tf.int32)
    w = tf.cast(ex['width'], tf.int32)
    tf.debugging.assert_equal(
        tf.cast(ex['channels'], tf.int32),
        tf.constant(int(num_channels), dtype=tf.int32),
        message="Channel count in TFRecord does not match expected num_channels.")
    return tf.reshape(pixels, [h, w, num_channels])


def write_images(
    images: list[Image],
    name: str,
    records_dir: str = Config.RECORDS_DIR_V2,
) -> str:
    """Write ``Image`` objects to a single TFRecord file; return its path.

    Materialises the whole list in RAM; for large datasets prefer
    :func:`open_writer` and stream one image at a time.
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


class _ImageWriter:
    """Streaming-write side of :func:`open_writer` (auto-incrementing index)."""

    def __init__(self, writer: tf.io.TFRecordWriter, path: str) -> None:
        self._writer = writer
        self._path = path
        self._count = 0

    def write(self, img: Image, index: int | None = None) -> None:
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
def open_writer(name: str, records_dir: str = Config.RECORDS_DIR_V2):
    """Context manager for streaming ``Image`` writes (memory scales with one image)."""
    os.makedirs(records_dir, exist_ok=True)
    path = tfrecord_path(records_dir, name)
    writer = tf.io.TFRecordWriter(path)
    handle = _ImageWriter(writer, path)
    try:
        yield handle
    finally:
        writer.close()


def read_images(path_or_glob: str, num_images: int = 5) -> list[Image]:
    """Read up to ``num_images`` ``Image`` objects from ``path_or_glob`` (in order).

    Stops as soon as ``num_images`` is reached, so reading the first few records
    of a large shard is cheap.
    """
    paths = sorted(_glob.glob(path_or_glob)) or [path_or_glob]
    out: list[Image] = []
    for raw in tf.data.TFRecordDataset(paths):
        out.append(Image.from_tfrecord(raw))
        if len(out) >= num_images:
            break
    return out
