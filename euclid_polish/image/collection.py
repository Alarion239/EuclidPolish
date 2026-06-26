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
from euclid_polish.image.plotting import plot_reconstruction
from euclid_polish.image.tfio import read_multiband_skyimages, write_multiband_skyimages
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
    def read(cls, path_or_glob: str, *, limit: Optional[int] = None,
             stamp: Optional[Stamp] = None) -> "ImageSet":
        """Read every record under ``path_or_glob`` (or the first ``limit``)."""
        num = limit if limit is not None else (1 << 30)
        images = read_multiband_skyimages(path_or_glob, num_images=num, mode='first')
        return cls(images=images, stamp=stamp)

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
        return write_multiband_skyimages(self.images, name=name, records_dir=records_dir)

    # -- queries -- #

    def by_role(self, role: Role) -> "ImageSet":
        """The subset whose :attr:`Image.role` equals ``role``."""
        return ImageSet(images=[im for im in self.images if im.role is role],
                        stamp=self.stamp)

    def _first_role(self, role: Role) -> Optional[Image]:
        for im in self.images:
            if im.role is role:
                return im
        return None

    # -- rendering -- #

    def plot_reconstruction(self, output_path: str, *, regime: str = "eye",
                            asinh_scale: Optional[float] = None) -> str:
        """Render the LR→SR(→HR) reconstruction figure for this set.

        Picks the SR image (``role='sr'``, required), the LR input
        (``role='lr'`` or ``'real'``, optional — a 2× pooled SR-VIS proxy is
        used when absent), and the HR truth (``role='hr'``, optional) from the
        set, then delegates to the shared renderer. ``regime`` is the colour
        regime (``"eye"`` or ``"calibrated"``). Returns ``output_path``.
        """
        sr = self._first_role(Role.SR)
        if sr is None:
            raise ValueError("plot_reconstruction needs an image with role='sr'")
        lr = self._first_role(Role.LR) or self._first_role(Role.REAL)
        hr = self._first_role(Role.HR)

        sr_data = np.asarray(sr.data, dtype=np.float32)

        lr_data = lr_cube = None
        if lr is not None:
            a = np.asarray(lr.data, dtype=np.float32)
            if a.ndim == 3 and a.shape[-1] > 1:
                lr_cube, lr_data = a, a[..., 0]
            else:
                lr_data = a[..., 0] if a.ndim == 3 else a
        if lr_data is None:
            vis = sr_data[..., 0] if sr_data.ndim == 3 else sr_data
            h, w = vis.shape[:2]
            lr_data = vis[: h - h % 2, : w - w % 2].reshape(
                h // 2, 2, w // 2, 2).mean(axis=(1, 3))

        hr_data = hr_cube = None
        if hr is not None:
            a = np.asarray(hr.data, dtype=np.float32)
            if a.ndim == 3 and a.shape[-1] > 1:
                hr_cube, hr_data = a, a[..., 0]
            else:
                hr_data = a[..., 0] if a.ndim == 3 else a

        plot_reconstruction(
            lr_data=lr_data, sr_data=sr_data, hr_data=hr_data,
            output_path=output_path, lr_cube=lr_cube, hr_cube=hr_cube,
            asinh_scale=asinh_scale, rgb_mode=regime)
        return output_path

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
