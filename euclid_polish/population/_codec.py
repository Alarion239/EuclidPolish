"""Compact array codec shared by population-calibration artifacts.

Arrays are serialized as ``zlib(level 9)`` over the raw little-endian
C-order bytes, then base64-encoded so they embed in the JSON calibration
artifacts. Every encoded array carries its dtype and shape next to it in the
payload; a SHA-256 of the raw bytes pins the content so a tampered or
truncated artifact fails closed at load time.
"""

from __future__ import annotations

import base64
import hashlib
import zlib
from typing import Any

import numpy as np

_SUPPORTED_DTYPES = ("<f4", "<f8", "<i4", "|u1")


def _canonical(values: np.ndarray, dtype: str) -> np.ndarray:
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported codec dtype {dtype!r}")
    return np.asarray(values, dtype=dtype, order="C")


def array_sha256(values: np.ndarray, dtype: str) -> str:
    """SHA-256 of the raw canonical bytes of ``values``."""
    return hashlib.sha256(
        _canonical(values, dtype).tobytes(order="C")
    ).hexdigest()


def encode_array(values: np.ndarray, dtype: str) -> str:
    """Serialize an array as zlib+base64 canonical bytes."""
    raw = _canonical(values, dtype).tobytes(order="C")
    return base64.b64encode(zlib.compress(raw, level=9)).decode("ascii")


def decode_array(
    encoded: Any,
    shape: tuple[int, ...],
    dtype: str,
    *,
    sha256: str | None = None,
    name: str = "array",
) -> np.ndarray:
    """Decode one array, verifying byte count and (optionally) its digest."""
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported codec dtype {dtype!r}")
    try:
        raw = zlib.decompress(base64.b64decode(str(encoded), validate=True))
    except (TypeError, ValueError, zlib.error) as exc:
        raise ValueError(f"{name} encoding is invalid") from exc
    itemsize = np.dtype(dtype).itemsize
    expected = int(np.prod(shape, dtype=np.int64)) * itemsize
    if len(raw) != expected:
        raise ValueError(f"{name} byte count is invalid")
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != str(sha256):
        raise ValueError(f"{name} fingerprint is invalid")
    values = np.frombuffer(raw, dtype=dtype).reshape(shape)
    values = np.array(values, order="C")
    values.setflags(write=False)
    return values
