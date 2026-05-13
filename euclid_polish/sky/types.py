"""
Typed data structures for sky images.

Carrying pixel_scale alongside the data enables validation at convolution
time: the PSF kernel must be sampled at the same scale as the image it
will be convolved with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from euclid_polish.config import Config


# ---------------------------------------------------------------------------
# Multi-band sky image (TFRecord schema v2)
# ---------------------------------------------------------------------------

@dataclass
class MultiBandSkyImage:
    """A multi-channel sky image with explicit band metadata.

    Channel layout is always ``(H, W, C)``. The ``band_names`` tuple names
    each channel in order; for HR target tensors C=1 and band_names=('VIS',),
    for LR input tensors C=4 and band_names=('VIS','Y_E','J_E','H_E').

    A single ``pixel_scale_arcsec`` covers all channels. After the forward
    model resamples NISP onto the VIS LR grid, all four LR channels share
    the same 0.10″/pix scale; HR is 0.05″/pix VIS only.
    """

    data: np.ndarray                       # float32, shape (H, W, C)
    pixel_scale_arcsec: float              # one scale shared by all channels
    band_names: Tuple[str, ...]            # length C, names from Config.LR_INPUT_BAND_NAMES
    is_clean: bool                         # True → HR clean target; False → LR dirty input
    index: Optional[int] = None            # position in the dataset
    subset: Optional[str] = None           # 'train' or 'validate'
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Promote 2-D arrays to 3-D (H, W, 1) so single-channel callers can
        # pass a flat array without an explicit ``[..., np.newaxis]``.
        if self.data.ndim == 2:
            self.data = self.data[..., np.newaxis]
        if self.data.ndim != 3:
            raise ValueError(
                f"MultiBandSkyImage.data must be (H, W, C); got shape {self.data.shape}"
            )
        if len(self.band_names) != self.data.shape[-1]:
            raise ValueError(
                f"band_names has {len(self.band_names)} entries but data has "
                f"{self.data.shape[-1]} channels"
            )

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def num_channels(self) -> int:
        return self.data.shape[-1]

    # TFRecord feature schema v2 (single source of truth)
    _TFRECORD_FEATURES = {
        'image':         tf.io.FixedLenFeature([], tf.string),
        'index':         tf.io.FixedLenFeature([], tf.int64),
        'height':        tf.io.FixedLenFeature([], tf.int64),
        'width':         tf.io.FixedLenFeature([], tf.int64),
        'channels':      tf.io.FixedLenFeature([], tf.int64),
        'pixel_scale':   tf.io.FixedLenFeature([], tf.float32),
        'is_clean':      tf.io.FixedLenFeature([], tf.int64),
        'band_names':    tf.io.FixedLenFeature([], tf.string),  # comma-joined
        'schema_version':tf.io.FixedLenFeature([], tf.int64),
    }

    def to_tfrecord(self, index: int | None = None) -> bytes:
        """Serialize to a v2 TFRecord Example (bytes). Stores data as float32."""
        h, w, c = self.shape
        idx = index if index is not None else (self.index or 0)
        arr_fp32 = np.ascontiguousarray(self.data, dtype=np.float32)
        bytes_value = arr_fp32.tobytes()
        bands_value = ",".join(self.band_names).encode("utf-8")
        feature = {
            'image':          tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_value])),
            'index':          tf.train.Feature(int64_list=tf.train.Int64List(value=[idx])),
            'height':         tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
            'width':          tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
            'channels':       tf.train.Feature(int64_list=tf.train.Int64List(value=[c])),
            'pixel_scale':    tf.train.Feature(float_list=tf.train.FloatList(value=[self.pixel_scale_arcsec])),
            'is_clean':       tf.train.Feature(int64_list=tf.train.Int64List(value=[int(self.is_clean)])),
            'band_names':     tf.train.Feature(bytes_list=tf.train.BytesList(value=[bands_value])),
            'schema_version': tf.train.Feature(int64_list=tf.train.Int64List(value=[Config.TFRECORD_SCHEMA_VERSION])),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    @classmethod
    def from_tfrecord(cls, raw_record) -> MultiBandSkyImage:
        """Parse a single v2 TFRecord example (eager mode)."""
        example = tf.io.parse_single_example(raw_record, cls._TFRECORD_FEATURES)
        h    = int(example['height'].numpy())
        w    = int(example['width'].numpy())
        c    = int(example['channels'].numpy())
        idx  = int(example['index'].numpy())
        ps   = round(float(example['pixel_scale'].numpy()), 6)
        clean= bool(example['is_clean'].numpy())
        bands= tuple(example['band_names'].numpy().decode("utf-8").split(","))
        image_bytes = tf.io.decode_raw(example['image'], tf.float32)
        data = tf.reshape(image_bytes, [h, w, c]).numpy()
        return cls(
            data=data,
            pixel_scale_arcsec=ps,
            band_names=bands,
            is_clean=clean,
            index=idx,
        )
