"""Bounded-memory reads of one large 2-D FITS image plane.

The NEXUS quick-release mosaics ship as gzip-compressed FITS (F200W is a
34728 × 32112 float32 plane: 4.46 GB decompressed). Reading one with
``hdu.data`` decompresses the whole plane into RAM, and an ``astype`` copy of
the big-endian pixels doubled that to ~9 GB per job. astropy cannot
memory-map a gzip file, and its ``hdu.section`` restores the file position
after every row read, which on a gzip stream means re-decompressing from the
start for each row.

:func:`open_plane` locates the first 2-D image HDU with a celestial WCS by
parsing headers only, then serves row ranges:

* an uncompressed file is memory-mapped (``np.memmap``): a crop touches only
  its own pages;
* a gzip file is streamed **forward**, reading at most :data:`CHUNK_BYTES`
  at a time; a backward jump reopens the stream. The last full-width row
  strip up to :data:`STRIP_CACHE_BYTES` is kept, so row-major tile loops
  (``download_nexus_field``) decompress each strip once.

:class:`FitsPlane` subclasses astropy's :class:`~astropy.io.fits.Section`,
which :class:`~astropy.nddata.Cutout2D` accepts in place of an array: a
cutout of the plane is pixel-identical to a cutout of the whole
``float32`` array. Every slice comes back as a C-contiguous ``float32``
copy. Only plain images are supported (no tile-compressed HDUs, no BLANK
handling; BSCALE/BZERO are applied).
"""

from __future__ import annotations

import gzip
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.io.fits.hdu.image import Section
from astropy.wcs import WCS

#: Largest decompressed read (and forward skip) issued at once on a gzip plane.
CHUNK_BYTES = 32 * 1024 * 1024
#: Full-width row strips up to this size are read whole and cached (gzip).
STRIP_CACHE_BYTES = 256 * 1024 * 1024

_BLOCK = 2880
_DTYPES = {8: "u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}
_GZIP_MAGIC = b"\x1f\x8b"


def _is_gzip(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == _GZIP_MAGIC


def _padded(size: int) -> int:
    return int(math.ceil(size / _BLOCK) * _BLOCK)


def _data_size(header: fits.Header) -> int:
    naxis = int(header.get("NAXIS", 0) or 0)
    if naxis == 0:
        return 0
    count = math.prod(int(header.get(f"NAXIS{axis}", 0) or 0) for axis in range(1, naxis + 1))
    groups = int(header.get("GCOUNT", 1) or 1)
    extra = int(header.get("PCOUNT", 0) or 0)
    return abs(int(header["BITPIX"])) // 8 * groups * (extra + count)


def _celestial(header: fits.Header) -> WCS | None:
    try:
        wcs = WCS(header).celestial
    except Exception:  # noqa: BLE001 - archive headers vary; no WCS = not this HDU
        return None
    return wcs if wcs.has_celestial else None


class FitsPlane(Section):
    """A lazily read 2-D FITS image (see the module docstring).

    ``shape`` is ``(ny, nx)``, ``header`` the image HDU's header (or the
    primary header when that one carries the WCS, like
    ``jwst_euclid._find_image``), ``wcs`` its celestial WCS and ``hdu_name``
    the HDU's name. Use it as a context manager (or :meth:`close` it).
    """

    def __init__(self, path: Path | str) -> None:  # noqa: D107 - Section's hdu is unused
        self.path = Path(path)
        self._gzip = _is_gzip(self.path)
        self._stream: Any = None
        self._memmap: np.memmap | None = None
        self._strip: tuple[int, int, np.ndarray] | None = None
        self._open_stream()
        try:
            self._locate()
        except BaseException:
            self.close()
            raise
        if not self._gzip:
            self._stream.close()
            self._stream = None
            self._memmap = np.memmap(self.path, dtype=self._raw_dtype, mode="r",
                                     offset=self._offset, shape=self._shape)

    # ------------------------------------------------------------------ open

    def _open_stream(self) -> None:
        if self._stream is not None:
            self._stream.close()
        self._stream = gzip.open(self.path, "rb") if self._gzip else self.path.open("rb")

    def _locate(self) -> None:
        primary: fits.Header | None = None
        index = 0
        while True:
            try:
                header = fits.Header.fromfile(self._stream, endcard=True, padding=True)
            except EOFError:
                break
            if index == 0:
                primary = header
            start = self._stream.tell()
            is_image = index == 0 or str(header.get("XTENSION", "")).strip() == "IMAGE"
            shape = (int(header.get("NAXIS2", 0) or 0), int(header.get("NAXIS1", 0) or 0))
            if (is_image and int(header.get("NAXIS", 0) or 0) == 2 and min(shape) > 0
                    and int(header.get("BITPIX", 0)) in _DTYPES):
                candidates = [header] if index == 0 or primary is None else [header, primary]
                for source in candidates:
                    wcs = _celestial(source)
                    if wcs is not None:
                        self._accept(header, source, wcs, shape, start, index)
                        return
            self._seek(start + _padded(_data_size(header)))
            index += 1
        raise ValueError(f"no 2-D image with celestial WCS found in {self.path.name}")

    def _accept(self, header: fits.Header, source: fits.Header, wcs: WCS,
                shape: tuple[int, int], offset: int, index: int) -> None:
        self.header = source.copy()
        self.wcs = wcs
        name = str(header.get("EXTNAME", "") or "").strip() if index else "PRIMARY"
        self.hdu_name = name or "PRIMARY"
        self._shape = shape
        self._offset = offset
        self._raw_dtype = np.dtype(_DTYPES[int(header["BITPIX"])])
        self._row_bytes = shape[1] * self._raw_dtype.itemsize
        self._bscale = float(header.get("BSCALE", 1.0) or 1.0)
        self._bzero = float(header.get("BZERO", 0.0) or 0.0)

    # ------------------------------------------------------- Section contract

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float32)

    def __getitem__(self, key: Any) -> np.ndarray:
        (y0, y1), (x0, x1) = self._bounds(key)
        if y1 <= y0 or x1 <= x0:
            return np.empty((max(0, y1 - y0), max(0, x1 - x0)), np.float32)
        if self._memmap is not None:
            return self._to_float(self._memmap[y0:y1, x0:x1])
        cached = self._cached(y0, y1)
        if cached is not None:
            return self._to_float(cached[:, x0:x1])
        if (y1 - y0) * self._row_bytes <= STRIP_CACHE_BYTES:
            return self._to_float(self.rows(y0, y1)[:, x0:x1])
        out = np.empty((y1 - y0, x1 - x0), np.float32)
        step = max(1, CHUNK_BYTES // self._row_bytes)
        for start in range(y0, y1, step):
            stop = min(y1, start + step)
            out[start - y0:stop - y0] = self._scaled(self._read_rows(start, stop)[:, x0:x1])
        return out

    # ------------------------------------------------------------------ rows

    def rows(self, y0: int, y1: int) -> np.ndarray:
        """Full-width rows ``[y0, y1)`` in the file's own dtype (unscaled).

        A memory-mapped view on an uncompressed file; on a gzip file the
        strip is read (forward) and cached for the next slices of it."""
        if not 0 <= y0 < y1 <= self._shape[0]:
            raise IndexError(f"rows [{y0}, {y1}) outside 0…{self._shape[0]}")
        if self._memmap is not None:
            return self._memmap[y0:y1]
        cached = self._cached(y0, y1)
        if cached is not None:
            return cached
        strip = self._read_rows(y0, y1)
        self._strip = (y0, y1, strip) if strip.nbytes <= STRIP_CACHE_BYTES else None
        return strip

    def _cached(self, y0: int, y1: int) -> np.ndarray | None:
        if self._strip is None:
            return None
        start, stop, strip = self._strip
        if start <= y0 and y1 <= stop:
            return strip[y0 - start:y1 - start]
        return None

    def _read_rows(self, y0: int, y1: int) -> np.ndarray:
        self._seek(self._offset + y0 * self._row_bytes)
        count = (y1 - y0) * self._row_bytes
        buffer = self._stream.read(count)
        if len(buffer) != count:
            raise ValueError(f"{self.path.name} is truncated (rows {y0}–{y1})")
        return np.frombuffer(buffer, dtype=self._raw_dtype).reshape(y1 - y0, self._shape[1])

    def _seek(self, target: int) -> None:
        """Move the stream to ``target``: forward by reading, back by reopening."""
        if not self._gzip:
            self._stream.seek(target)
            return
        position = self._stream.tell()
        if target < position:
            self._open_stream()
            position = 0
        while position < target:
            chunk = self._stream.read(min(CHUNK_BYTES, target - position))
            if not chunk:
                raise ValueError(f"{self.path.name} ends before byte {target}")
            position += len(chunk)

    # --------------------------------------------------------------- helpers

    def _bounds(self, key: Any) -> tuple[tuple[int, int], tuple[int, int]]:
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) == 1:
            key = (key[0], slice(None))
        if len(key) != 2 or not all(isinstance(item, slice) for item in key):
            raise IndexError("a FitsPlane takes a 2-D slice [y0:y1, x0:x1]")
        bounds = []
        for item, size in zip(key, self._shape, strict=True):
            start, stop, stride = item.indices(size)
            if stride != 1:
                raise IndexError("a FitsPlane slice cannot have a step")
            bounds.append((start, max(start, stop)))
        return bounds[0], bounds[1]

    def _scaled(self, raw: np.ndarray) -> np.ndarray:
        if self._bscale == 1.0 and self._bzero == 0.0:
            return raw
        return raw * np.float32(self._bscale) + np.float32(self._bzero)

    def _to_float(self, raw: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(self._scaled(raw), dtype=np.float32)

    # ------------------------------------------------------------- lifetime

    def close(self) -> None:
        self._strip = None
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self._memmap is not None:
            mapping = getattr(self._memmap, "_mmap", None)
            self._memmap = None
            if mapping is not None:
                mapping.close()

    def __enter__(self) -> FitsPlane:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"FitsPlane({os.fspath(self.path)!r}, shape={self._shape})"


def open_plane(path: Path | str) -> FitsPlane:
    """Open the first 2-D celestial image of ``path`` for bounded reads.

    :class:`ValueError` when the file has no such image."""
    return FitsPlane(path)


__all__ = ["CHUNK_BYTES", "STRIP_CACHE_BYTES", "FitsPlane", "open_plane"]
