"""``ImageSet`` — a lazy, streamable collection of :class:`Image`.

Constructed either from an in-memory list (:meth:`from_images`) or from a
TFRecord path (:meth:`read`). In the lazy path no data is loaded until the
set is iterated; gigabyte-scale training datasets can be referenced without
consuming RAM.

Callers see a uniform collection interface regardless of which mode is active.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.image.core import Image, Role
from euclid_polish.image.tfio import write_images
from euclid_polish.provenance.records import Stamp


class ImageSet:
    """An ordered collection of :class:`Image` with an optional set-level stamp.

    Two construction modes:

    * :meth:`from_images` — eager, backed by an in-memory list.
    * :meth:`read` — **lazy**, backed by a TFRecord path; records are streamed
      on demand and never fully materialised in RAM.

    Both modes expose the same collection protocol (``__iter__``, ``__len__``,
    ``__getitem__``, :meth:`write`, :meth:`by_role`, :meth:`split`).
    """

    def __init__(
        self,
        *,
        images: list[Image] | None = None,
        path: str | None = None,
        stamp: Stamp | None = None,
        max_images: int | None = None,
    ) -> None:
        if (images is None) == (path is None):
            raise ValueError("Exactly one of 'images' or 'path' must be provided.")
        self._images = images
        self._path = path
        self._max_images = max_images
        self.stamp = stamp

    # -- construction -- #

    @classmethod
    def from_images(cls, images, *, stamp: Stamp | None = None) -> ImageSet:
        """Build an eager set from an iterable of :class:`Image`."""
        return cls(images=list(images), stamp=stamp)

    @classmethod
    def read(cls, path_or_glob: str, *, stamp: Stamp | None = None,
             num_images: int | None = None) -> ImageSet:
        """Return a **lazy** set backed by ``path_or_glob``.

        No records are read until the set is iterated. This is safe for
        training-scale TFRecord files that would otherwise exhaust RAM.
        ``num_images`` caps the number of records yielded when iterating.
        """
        return cls(path=path_or_glob, stamp=stamp, max_images=num_images)

    # -- source access -- #

    @property
    def source_path(self) -> str | None:
        """TFRecord path when constructed with :meth:`read`; ``None`` otherwise."""
        return self._path

    # -- collection protocol -- #

    def __iter__(self) -> Iterator[Image]:
        if self._images is not None:
            if self._max_images is not None:
                yield from self._images[:self._max_images]
            else:
                yield from self._images
        else:
            count = 0
            for raw in tf.data.TFRecordDataset(self._path):
                if self._max_images is not None and count >= self._max_images:
                    break
                yield Image.from_tfrecord(raw.numpy())
                count += 1

    def __len__(self) -> int:
        if self._images is not None:
            return len(self._images)
        raise TypeError(
            "len() not supported for lazy ImageSet — materialise first with "
            "list(image_set) or pass num_images= to ImageSet.read()"
        )

    def __getitem__(self, key):
        if self._images is not None:
            result = self._images[key]
            if isinstance(result, list):
                return ImageSet.from_images(result, stamp=self.stamp)
            return result
        raise TypeError(
            "Indexing not supported for lazy ImageSet — materialise first with "
            "list(image_set)"
        )

    def __repr__(self) -> str:
        if self._images is not None:
            return f"ImageSet(n={len(self._images)}, mode=eager)"
        return f"ImageSet(path={self._path!r}, mode=lazy)"

    # -- persistence -- #

    def write(self, records_dir: str = Config.RECORDS_DIR_V2,
              name: str = "images") -> str:
        """Write the set to ``<records_dir>/<name>.tfrecord``; return the path.

        Streams via ``__iter__`` so lazy sets never fully materialise in RAM.
        """
        return write_images(self, name=name, records_dir=records_dir)

    # -- queries -- #

    def by_role(self, role: Role) -> ImageSet:
        """The subset whose :attr:`Image.role` equals ``role``."""
        return ImageSet.from_images(
            (img for img in self if img.role is role), stamp=self.stamp
        )

    def split(self, train_frac: float, *,
              rng: np.random.Generator | None = None) -> tuple[ImageSet, ImageSet]:
        """Random disjoint split into ``(train, validate)`` by fraction.

        Materialises the full set once to shuffle indices. Deterministic given
        ``rng``.
        """
        if not 0.0 <= train_frac <= 1.0:
            raise ValueError(f"train_frac must be in [0, 1], got {train_frac}")
        images = list(self)
        rng = rng if rng is not None else np.random.default_rng()
        n = len(images)
        order = rng.permutation(n)
        k = int(round(train_frac * n))
        train_idx, val_idx = order[:k], order[k:]
        return (
            ImageSet.from_images([images[i] for i in train_idx], stamp=self.stamp),
            ImageSet.from_images([images[i] for i in val_idx], stamp=self.stamp),
        )
