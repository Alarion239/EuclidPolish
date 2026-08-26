"""TensorFlow persistence for :class:`Image` and :class:`ImageSet`.

The import direction stays ``sky/`` → ``image/`` (never the reverse), while
the dependency-light :mod:`euclid_polish.image.core` remains TensorFlow-free.
"""

from __future__ import annotations

import contextlib
import dataclasses
import glob as _glob
import os
from collections.abc import Iterable

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from euclid_polish.config import Config
from euclid_polish.image.core import Image, Role
from euclid_polish.provenance.records import Stamp


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

# Full eager-mode persistence schema. Every field is written; empty provenance
# strings represent an unstamped image. The graph training path intentionally
# uses only ``_GRAPH_FEATURES`` above.
_IMAGE_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "index": tf.io.FixedLenFeature([], tf.int64),
    "height": tf.io.FixedLenFeature([], tf.int64),
    "width": tf.io.FixedLenFeature([], tf.int64),
    "channels": tf.io.FixedLenFeature([], tf.int64),
    "pixel_scale": tf.io.FixedLenFeature([], tf.float32),
    "is_clean": tf.io.FixedLenFeature([], tf.int64),
    "band_names": tf.io.FixedLenFeature([], tf.string),
    "role": tf.io.FixedLenFeature([], tf.string),
    "prov_id": tf.io.FixedLenFeature([], tf.string),
    "prov_stamp": tf.io.FixedLenFeature([], tf.string),
}


def serialize_image(image: Image, index: int | None = None) -> bytes:
    """Serialize one :class:`Image` as a TFRecord ``Example``."""
    height, width, channels = image.shape
    record_index = index if index is not None else (image.index or 0)
    if image.stamp is not None:
        embedded_stamp = image.stamp
        if embedded_stamp.subset is None and image.subset is not None:
            embedded_stamp = dataclasses.replace(
                embedded_stamp, subset=image.subset
            )
        provenance_id = str(embedded_stamp.id).encode("utf-8")
        provenance_stamp = embedded_stamp.to_json().encode("utf-8")
    else:
        provenance_id = provenance_stamp = b""

    def bytes_feature(value: bytes) -> tf.train.Feature:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def integer_feature(value: int) -> tf.train.Feature:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    features = {
        "image": bytes_feature(
            np.ascontiguousarray(image.data, dtype=np.float32).tobytes()
        ),
        "index": integer_feature(record_index),
        "height": integer_feature(height),
        "width": integer_feature(width),
        "channels": integer_feature(channels),
        "pixel_scale": tf.train.Feature(
            float_list=tf.train.FloatList(value=[image.pixel_scale_arcsec])
        ),
        "is_clean": integer_feature(int(image.is_clean)),
        "band_names": bytes_feature(",".join(image.band_names).encode("utf-8")),
        "role": bytes_feature(image.role.value.encode("utf-8")),
        "prov_id": bytes_feature(provenance_id),
        "prov_stamp": bytes_feature(provenance_stamp),
    }
    return tf.train.Example(
        features=tf.train.Features(feature=features)
    ).SerializeToString()


def deserialize_image(raw_record: bytes | tf.Tensor) -> Image:
    """Deserialize one eager TFRecord example into an :class:`Image`."""
    record = tf.convert_to_tensor(raw_record, dtype=tf.string)
    example = tf.io.parse_single_example(record, _IMAGE_FEATURES)
    height = int(example["height"].numpy())
    width = int(example["width"].numpy())
    channels = int(example["channels"].numpy())
    data = tf.reshape(
        tf.io.decode_raw(example["image"], tf.float32),
        [height, width, channels],
    ).numpy()
    role = Role(example["role"].numpy().decode("utf-8"))
    stamp = None
    subset = None
    provenance_stamp = example["prov_stamp"].numpy()
    if provenance_stamp:
        stamp = Stamp.from_json(provenance_stamp.decode("utf-8"))
        subset = stamp.subset
    return Image(
        data=data,
        pixel_scale_arcsec=round(float(example["pixel_scale"].numpy()), 6),
        band_names=tuple(
            example["band_names"].numpy().decode("utf-8").split(",")
        ),
        is_clean=bool(example["is_clean"].numpy()),
        role=role,
        index=int(example["index"].numpy()),
        subset=subset,
        stamp=stamp,
    )


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
    images: Iterable[Image],
    name: str,
    records_dir: str = Config.RECORDS_DIR_V2,
) -> str:
    """Write ``Image`` objects to a single TFRecord file; return its path.

    Consumes ``images`` once and writes each record immediately; callers may
    therefore pass either an in-memory collection or a lazy iterable.
    """
    os.makedirs(records_dir, exist_ok=True)
    path = tfrecord_path(records_dir, name)
    writer = tf.io.TFRecordWriter(path)
    try:
        for idx, img in enumerate(tqdm(images, desc=f"Writing {name}", unit="img")):
            writer.write(serialize_image(img, index=idx))
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
        self._writer.write(serialize_image(img, index=index))
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
        out.append(deserialize_image(raw))
        if len(out) >= num_images:
            break
    return out
