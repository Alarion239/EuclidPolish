"""``ImageSet`` — a collection of :class:`Image` persisted as TFRecords.

The collection counterpart to the single-image atom. It owns the verbs that act
on *many* images at once — write/read the TFRecord stack, iterate, filter by
role, split into train/validate — and carries one optional set-level provenance
:class:`Stamp` (e.g. the ``GenerationRun`` that produced the whole set).

Like :class:`Image`, it imports nothing but third-party libs + the provenance
value-types + its own ``image.tfio`` persistence layer; no operator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np

from euclid_polish.config import Config
from euclid_polish.image.core import Image, Role
from euclid_polish.image.tfio import read_images, write_images
from euclid_polish.provenance.records import Stamp


@dataclass
class ImageSet:
    """An ordered collection of :class:`Image` with an optional set-level stamp.

    Construct in memory with :meth:`from_images`, or load a TFRecord stack with
    :meth:`read`. ``stamp`` identifies the whole set (the run that produced it).
    """

    images: List[Image] = field(default_factory=list)
    stamp: Optional[Stamp] = None

    # -- construction -- #

    @classmethod
    def from_images(cls, images, *, stamp: Optional[Stamp] = None) -> "ImageSet":
        """Build a set from an iterable of :class:`Image`."""
        return cls(images=list(images), stamp=stamp)

    @classmethod
    def read(cls, path_or_glob: str, *, num_images: Optional[int] = None,
             stamp: Optional[Stamp] = None) -> "ImageSet":
        """Read every record under ``path_or_glob`` (or the first ``num_images``)."""
        n = num_images if num_images is not None else (1 << 30)
        return cls(images=read_images(path_or_glob, num_images=n), stamp=stamp)

    # -- collection protocol -- #

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self) -> Iterator[Image]:
        return iter(self.images)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return ImageSet(images=self.images[i], stamp=self.stamp)
        return self.images[i]

    # -- persistence -- #

    def write(self, records_dir: str = Config.RECORDS_DIR_V2,
              name: str = "images") -> str:
        """Write the set to ``<records_dir>/<name>.tfrecord``; return the path.

        Each image carries its own stamp into its record (the set-level stamp is
        provenance metadata for the whole file, recorded in its sidecar).
        """
        return write_images(self.images, name=name, records_dir=records_dir)

    # -- queries -- #

    def by_role(self, role: Role) -> "ImageSet":
        """The subset whose :attr:`Image.role` equals ``role``."""
        return ImageSet(images=[im for im in self.images if im.role is role],
                        stamp=self.stamp)

    def split(self, train_frac: float, *,
              rng: Optional[np.random.Generator] = None) -> "tuple[ImageSet, ImageSet]":
        """Random disjoint split into ``(train, validate)`` by fraction.

        ``train`` gets ``round(train_frac * len)`` images; the rest go to
        ``validate``. Deterministic given ``rng``.
        """
        if not 0.0 <= train_frac <= 1.0:
            raise ValueError(f"train_frac must be in [0, 1], got {train_frac}")
        rng = rng if rng is not None else np.random.default_rng()
        n = len(self.images)
        order = rng.permutation(n)
        k = int(round(train_frac * n))
        train_idx, val_idx = order[:k], order[k:]
        return (
            ImageSet(images=[self.images[i] for i in train_idx], stamp=self.stamp),
            ImageSet(images=[self.images[i] for i in val_idx], stamp=self.stamp),
        )
