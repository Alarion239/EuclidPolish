"""
Typed data structures for sky images.

Carrying pixel_scale alongside the data enables validation at convolution
time: the PSF kernel must be sampled at the same scale as the image it
will be convolved with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import tensorflow as tf


@dataclass
class SkyImage:
    """A sky image with its physical metadata."""

    data: np.ndarray          # float32, shape (H, W)
    pixel_scale: float        # arcsec / pixel
    is_clean: bool            # True → HR clean image; False → LR dirty image
    index: Optional[int] = None      # position in the dataset
    subset: Optional[str] = None     # 'train' or 'validate'
    metadata: dict = field(default_factory=dict)  # galaxy/star params from simulate_field

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    # TFRecord feature schema (shared by to/from)
    _TFRECORD_FEATURES = {
        'image':       tf.io.FixedLenFeature([], tf.string),
        'index':       tf.io.FixedLenFeature([], tf.int64),
        'height':      tf.io.FixedLenFeature([], tf.int64),
        'width':       tf.io.FixedLenFeature([], tf.int64),
        'pixel_scale': tf.io.FixedLenFeature([], tf.float32),
        'is_clean':    tf.io.FixedLenFeature([], tf.int64),
    }

    def to_tfrecord(self, index: int | None = None) -> bytes:
        """Serialize to a TFRecord Example (bytes). Stores data as float16."""
        h, w = self.shape
        idx = index if index is not None else (self.index or 0)
        arr_fp16 = self.data.flatten().astype(np.float16)
        feature = {
            'image':       tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr_fp16.tobytes()])),
            'index':       tf.train.Feature(int64_list=tf.train.Int64List(value=[idx])),
            'height':      tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
            'width':       tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
            'pixel_scale': tf.train.Feature(float_list=tf.train.FloatList(value=[self.pixel_scale])),
            'is_clean':    tf.train.Feature(int64_list=tf.train.Int64List(value=[int(self.is_clean)])),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    @classmethod
    def from_tfrecord(cls, raw_record) -> SkyImage:
        """Parse a single TFRecord example into a SkyImage (eager mode)."""
        example = tf.io.parse_single_example(raw_record, cls._TFRECORD_FEATURES)
        height      = int(example['height'].numpy())
        width       = int(example['width'].numpy())
        index       = int(example['index'].numpy())
        pixel_scale = round(float(example['pixel_scale'].numpy()), 6)
        is_clean    = bool(example['is_clean'].numpy())
        image_bytes = tf.io.decode_raw(example['image'], tf.float16)
        data = tf.cast(tf.reshape(image_bytes, [height, width]), tf.float32).numpy()
        return cls(data=data, pixel_scale=pixel_scale, is_clean=is_clean, index=index)
