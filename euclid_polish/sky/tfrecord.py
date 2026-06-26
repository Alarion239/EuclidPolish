"""Back-compat shim: TFRecord stack I/O now lives in
:mod:`euclid_polish.image.tfio`.

The read/write functions moved into the ``euclid_polish.image`` package (so the
import direction is ``sky`` → ``image``, never the reverse) and are now the
persistence layer behind :class:`~euclid_polish.image.collection.ImageSet`. They
are re-exported here under their original names for back-compat.
"""

from __future__ import annotations

from euclid_polish.image.tfio import (  # noqa: F401
    open_multiband_writer,
    parse_record_graph_v2,
    read_multiband_skyimages,
    tfrecord_path,
    write_multiband_skyimages,
)

__all__ = [
    "tfrecord_path",
    "parse_record_graph_v2",
    "write_multiband_skyimages",
    "open_multiband_writer",
    "read_multiband_skyimages",
]
