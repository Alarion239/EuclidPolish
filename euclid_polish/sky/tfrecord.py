"""
TFRecord I/O utilities for EuclidPolish sky images.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import glob as _glob

from euclid_polish.config import Config
from euclid_polish.sky.types import MultiBandSkyImage


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
# Multi-band I/O (schema v2)
# ---------------------------------------------------------------------------

def parse_record_graph_v2(raw_record: tf.Tensor, num_channels: int) -> tf.Tensor:
    """Graph-mode v2 parser — safe inside ``tf.data.map``.

    Returns a float32 tensor of shape ``[H, W, num_channels]``. The caller
    must pre-commit to ``num_channels`` (1 for HR-VIS, 4 for LR multi-band)
    so the output shape is statically known to downstream graph ops.
    """
    ex = tf.io.parse_single_example(raw_record, MultiBandSkyImage._TFRECORD_FEATURES)
    pixels = tf.io.decode_raw(ex['image'], tf.float32)
    h = tf.cast(ex['height'], tf.int32)
    w = tf.cast(ex['width'],  tf.int32)
    # Static check against the provided num_channels — graph aborts on mismatch.
    c = tf.cast(ex['channels'], tf.int32)
    tf.debugging.assert_equal(
        c, tf.constant(int(num_channels), dtype=tf.int32),
        message="Channel count in TFRecord does not match expected num_channels.",
    )
    return tf.reshape(pixels, [h, w, num_channels])


def write_multiband_skyimages(
    images: list[MultiBandSkyImage],
    name: str,
    records_dir: str = Config.RECORDS_DIR_V2,
) -> str:
    """Write MultiBandSkyImage objects to a single v2 TFRecord file.

    Materialises the whole list in RAM. For large datasets where each
    image is several MB, prefer :func:`open_multiband_writer` and stream
    one image at a time — accumulating 6400 510² × 4-channel float32
    fields here costs ~26 GB.

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


class _MultiBandStreamingWriter:
    """Streaming-write side of :func:`open_multiband_writer`.

    Wraps a :class:`tf.io.TFRecordWriter` with an auto-incrementing index
    so callers don't have to remember which image they're on.
    """

    def __init__(self, writer: tf.io.TFRecordWriter, path: str) -> None:
        self._writer = writer
        self._path = path
        self._count = 0

    def write(self, img: MultiBandSkyImage, index: int | None = None) -> None:
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
def open_multiband_writer(
    name: str,
    records_dir: str = Config.RECORDS_DIR_V2,
):
    """Context manager for streaming MultiBandSkyImage writes.

    Use this when the producer of images is itself a loop and you don't
    want to hold the full set in RAM. Each ``.write(img)`` call
    serialises that one image to disk immediately::

        with open_multiband_writer("clean_train", records_dir) as w:
            for i in range(n):
                sky, _ = sim.simulate_field(rng)
                w.write(sky, index=i)

    Memory now scales with the size of *one* image, not the count.
    """
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
) -> list[MultiBandSkyImage]:
    """Read v2 TFRecords and return MultiBandSkyImage objects."""
    paths = sorted(_glob.glob(tfrecord_path_or_glob)) or [tfrecord_path_or_glob]
    dataset = tf.data.TFRecordDataset(paths)
    all_images = [
        MultiBandSkyImage.from_tfrecord(raw)
        for raw in tqdm(dataset, desc="Reading TFRecord")
    ]

    n = min(num_images, len(all_images))
    if mode == 'first':
        return all_images[:n]
    if mode == 'random':
        chosen = np.random.default_rng(42).choice(len(all_images), n, replace=False)
        return [all_images[i] for i in chosen]
    raise ValueError(f"mode must be 'first' or 'random', got {mode!r}")
