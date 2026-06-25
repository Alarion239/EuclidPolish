"""The behaviour-bearing cutout hierarchy.

``Cutout`` *composes* a :class:`~euclid_polish.sky.types.MultiBandSkyImage`
(has-a, not is-a): the image stays the pure pixel / serialization / physics
workhorse, while the cutout is the typed handle that carries identity +
provenance and owns the verb that produced it. Each leaf delegates its verb to
the existing engine through an injected callable, so the type layer never
duplicates the download / forward-model / reconstruct logic.

    EuclidLRCutout.query(...)           -> the Euclid archive
    SyntheticHRCutout.convolve(forward) -> MultiBandForward.process
    LRCutout.super_resolve(model, ...)  -> training.inference.reconstruct
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Optional, Sequence, Tuple

import numpy as np

from euclid_polish.config import Config
from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import Format, Stamp
from euclid_polish.provenance.store import ProvStore
from euclid_polish.sky.types import MultiBandSkyImage
from euclid_polish.training.inference import reconstruct as _default_reconstruct

_HR_SCALE = Config.DEFAULT_PIXEL_SCALE        # 0.05 arcsec/pix
_LR_SCALE = Config.VIS_PIXEL_SCALE_ARCSEC     # 0.10 arcsec/pix


@dataclass(frozen=True)
class Cutout:
    """A typed, provenance-carrying handle around a multi-band image."""

    image: MultiBandSkyImage
    id: ProvId
    produced_by: Optional[ProvId] = None
    parents: Tuple[ProvId, ...] = ()

    PROV_FORMAT: ClassVar[Format] = Format.FITS
    EXPECTED_PIXEL_SCALE: ClassVar[Optional[float]] = None

    # -- pixel proxies -- #

    @property
    def data(self) -> np.ndarray:
        return self.image.data

    @property
    def pixel_scale_arcsec(self) -> float:
        return self.image.pixel_scale_arcsec

    @property
    def band_names(self) -> Tuple[str, ...]:
        return self.image.band_names

    # -- provenance / serialization -- #

    def prov_stamp(self) -> Stamp:
        return Stamp(
            id=self.id,
            produced_by=self.produced_by,
            parents=tuple(self.parents),
            schema_version=3,
            subset=self.image.subset,
        )

    def stamped_image(self) -> MultiBandSkyImage:
        """The wrapped image carrying this cutout's stamp (for serialization)."""
        return self.image.with_stamp(self.prov_stamp())

    def to_tfrecord(self, index: Optional[int] = None) -> bytes:
        return self.stamped_image().to_tfrecord(index=index)


@dataclass(frozen=True)
class HRCutout(Cutout):
    """A high-resolution (0.05″/pix) cutout."""

    EXPECTED_PIXEL_SCALE: ClassVar[float] = _HR_SCALE


@dataclass(frozen=True)
class LRCutout(Cutout):
    """A low-resolution (0.10″/pix) cutout — the thing a model super-resolves."""

    EXPECTED_PIXEL_SCALE: ClassVar[float] = _LR_SCALE

    def super_resolve(
        self,
        model,
        model_id: ProvId,
        store: ProvStore,
        *,
        produced_by: Optional[ProvId] = None,
        reconstruct_fn: Optional[Callable] = None,
    ) -> "SRCutout":
        """Super-resolve this LR cutout into an :class:`SRCutout`.

        Delegates to ``reconstruct_fn`` (default ``training.inference.reconstruct``)
        and records lineage: the SR cutout's parents are ``(model_id, self.id)``.
        """
        if reconstruct_fn is None:
            reconstruct_fn = _default_reconstruct
        _lr2d, sr_data = reconstruct_fn(model, self.image.data)
        sr_data = np.asarray(sr_data, dtype=np.float32)
        if sr_data.ndim == 3 and sr_data.shape[-1] == len(self.band_names):
            bands = self.band_names
        else:
            bands = ("VIS",)
        sr_img = MultiBandSkyImage(
            data=sr_data, pixel_scale_arcsec=_HR_SCALE,
            band_names=bands, is_clean=True, subset=self.image.subset,
        )
        return SRCutout(
            image=sr_img, id=store.mint(), produced_by=produced_by,
            parents=(model_id, self.id),
        )


@dataclass(frozen=True)
class SyntheticHRCutout(HRCutout):
    """A clean generated HR field from the simulator."""

    def convolve(
        self,
        forward,
        store: ProvStore,
        *,
        produced_by: Optional[ProvId] = None,
        rng=None,
    ) -> "SyntheticLRCutout":
        """Run the forward model, producing a :class:`SyntheticLRCutout`.

        Delegates to ``forward.process`` (a ``MultiBandForward``); the LR
        cutout's parent is this HR cutout.
        """
        lr_img, _hr_out = forward.process(self.image, rng)
        return SyntheticLRCutout(
            image=lr_img, id=store.mint(), produced_by=produced_by,
            parents=(self.id,),
        )


@dataclass(frozen=True)
class SyntheticLRCutout(LRCutout):
    """The forward-model output of a :class:`SyntheticHRCutout`."""


@dataclass(frozen=True)
class EuclidLRCutout(LRCutout):
    """A real Euclid 4-band cutout from the archive."""

    @classmethod
    def query(
        cls,
        ra: float,
        dec: float,
        size: int,
        store: ProvStore,
        *,
        fetch_plane: Callable[[float, float, str, int], np.ndarray],
        bands: Sequence[str] = Config.LR_INPUT_BAND_NAMES,
        produced_by: Optional[ProvId] = None,
    ) -> "EuclidLRCutout":
        """Download a 4-band cutout at ``(ra, dec)``.

        ``fetch_plane(ra, dec, band, size)`` returns one ``(H, W)`` electron
        plane — injected so the type layer stays decoupled from the archive +
        ADU→e⁻ conversion machinery (which the Phase-3 wiring supplies).
        """
        planes = [
            np.asarray(fetch_plane(ra, dec, band, size), dtype=np.float32)
            for band in bands
        ]
        data = np.stack(planes, axis=-1)
        img = MultiBandSkyImage(
            data=data, pixel_scale_arcsec=_LR_SCALE,
            band_names=tuple(bands), is_clean=False,
        )
        return cls(image=img, id=store.mint(), produced_by=produced_by)


@dataclass(frozen=True)
class SRCutout(HRCutout):
    """A super-resolved cutout — output of :meth:`LRCutout.super_resolve`."""
