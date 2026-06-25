"""The behaviour-bearing cutout hierarchy.

``Cutout`` *composes* a :class:`~euclid_polish.sky.types.MultiBandSkyImage`
(has-a, not is-a): the image stays the pure pixel / serialization / physics
workhorse, while the cutout is the typed handle that carries identity +
provenance and owns the verb that produced it. Each leaf delegates its verb to
the existing engine through an injected callable, so the type layer never
duplicates the download / forward-model / reconstruct logic.

    EuclidLRCutout.query(...)              -> the Euclid archive
    SyntheticHRCutout.downsample(forward) -> MultiBandForward.process
    Model.upsample(lr, ...)                -> training.inference.reconstruct
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits as _fits

from euclid_polish.config import Config
from euclid_polish.euclid.downloader import fetch_cutout_at
from euclid_polish.euclid.photometry import adu_per_s_to_electrons_factor
from euclid_polish.provenance.defaults import default_store
from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import Format, Stamp
from euclid_polish.provenance.store import ProvStore
from euclid_polish.sky.types import MultiBandSkyImage

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


@dataclass(frozen=True)
class SyntheticHRCutout(HRCutout):
    """A clean generated HR field from the simulator."""

    def downsample(
        self,
        forward,
        *,
        rng=None,
        store=None,
        produced_by: Optional[ProvId] = None,
    ) -> "SyntheticLRCutout":
        """Run the forward model, producing a :class:`SyntheticLRCutout`.

        Delegates to ``forward.process`` (a
        :class:`~euclid_polish.sky.multiband_forward.MultiBandForward`); the
        LR cutout's parent is this HR cutout. Renamed from ``convolve``.

        Parameters
        ----------
        forward : MultiBandForward
            The forward-model operator.
        rng : np.random.Generator, optional
            Noise RNG; a fresh ``default_rng`` is used when None.
        store : ProvStore, optional
            Provenance store. Defaults to ``default_store()`` (guarded).
        produced_by : ProvId, optional
            Id of the orchestrating process.
        """
        lr_img, _hr_out = forward.process(self.image, rng)
        try:
            _store = store if store is not None else default_store()
            new_id = _store.mint()
        except Exception:   # noqa: BLE001 — provenance is best-effort
            new_id = ProvId.sentinel()   # ProvId imported at module top
        return SyntheticLRCutout(
            image=lr_img, id=new_id, produced_by=produced_by,
            parents=(self.id,),
        )

    @classmethod
    def generate(
        cls,
        simulator,
        *,
        rng=None,
        store=None,
        produced_by: Optional[ProvId] = None,
    ) -> "SyntheticHRCutout":
        """Generate a clean HR field from ``simulator``.

        Delegates to ``simulator.simulate_field(rng)`` (a
        :class:`~euclid_polish.sky.multiband_generator.MultiBandSimulator`).
        The resulting :class:`SyntheticHRCutout` gets a fresh provenance id.

        Parameters
        ----------
        simulator : MultiBandSimulator
            The scene-generator operator.
        rng : np.random.Generator, optional
            Reproducible noise source; a fresh ``default_rng()`` is used when None.
        store : ProvStore, optional
            Provenance store. Defaults to ``default_store()`` (guarded).
        produced_by : ProvId, optional
            Id of the orchestrating process (e.g. a GenerationRun).
        """
        if rng is None:
            rng = np.random.default_rng()
        sky_image, _meta = simulator.simulate_field(rng)
        try:
            _store = store if store is not None else default_store()
            new_id = _store.mint()
        except Exception:   # noqa: BLE001 — provenance is best-effort
            new_id = ProvId.sentinel()   # ProvId imported at module top
        return cls(image=sky_image, id=new_id, produced_by=produced_by)


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

    @classmethod
    def fetch(
        cls,
        ra: float,
        dec: float,
        size: int,
        *,
        bands: Sequence[str] = Config.LR_INPUT_BAND_NAMES,
        store=None,
        produced_by: Optional[ProvId] = None,
        fetch_plane: Optional[Callable] = None,
    ) -> "EuclidLRCutout":
        """Download a 4-band real Euclid cutout and convert to electrons.

        High-level entry for real-Euclid inference. Fetches each band from
        the Euclid archive (default) or via an injected ``fetch_plane``
        (tests / offline), converts each ADU s⁻¹ image to electrons via the
        per-band ``MAGZERO`` header keyword, and stacks to
        ``(H, W, len(bands))``.

        Parameters
        ----------
        ra, dec : float
            ICRS coordinates in degrees.
        size : int
            Cutout side in VIS pixels (0.10″/pix reference grid).
        bands : sequence of str
            Band names (default ``Config.LR_INPUT_BAND_NAMES``).
        store : ProvStore, optional
            Provenance store. Defaults to ``default_store()`` (guarded).
        produced_by : ProvId, optional
            Id of the orchestrating process.
        fetch_plane : callable, optional
            ``(ra, dec, band_name, size) -> np.ndarray[float32]`` already in
            electrons. When None the real archive download is used.
        """
        if fetch_plane is not None:
            planes = [
                np.asarray(fetch_plane(ra, dec, band_name, size), dtype=np.float32)
                for band_name in bands
            ]
        else:
            planes = cls._fetch_planes_from_archive(ra, dec, size, bands)

        data = np.stack(planes, axis=-1)
        img = MultiBandSkyImage(
            data=data, pixel_scale_arcsec=_LR_SCALE,
            band_names=tuple(bands), is_clean=False,
        )
        try:
            _store = store if store is not None else default_store()
            new_id = _store.mint()
        except Exception:   # noqa: BLE001 — provenance is best-effort
            new_id = ProvId.sentinel()   # ProvId imported at module top
        return cls(image=img, id=new_id, produced_by=produced_by)

    @classmethod
    def _fetch_planes_from_archive(
        cls,
        ra: float,
        dec: float,
        size: int,
        bands: Sequence[str],
    ) -> "list[np.ndarray]":
        """Real Euclid archive download path: fetch + MAGZERO → e⁻.

        Not exercised in tests (the ``fetch_plane`` injection bypasses it).
        All imports it uses (os, tempfile, _fits, fetch_cutout_at,
        adu_per_s_to_electrons_factor) are at the module top.
        """
        planes = []
        for band_name in bands:
            band = Config.get_band(band_name)
            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
                tmp_path = tf.name
            try:
                ok, err = fetch_cutout_at(
                    ra=ra, dec=dec, band_name=band_name,
                    output_file=tmp_path, cutout_size_vis_pixels=size,
                )
                if not ok:
                    raise RuntimeError(
                        f"EuclidLRCutout.fetch: band {band_name} failed: {err}")
                with _fits.open(tmp_path) as hdul:
                    arr = hdul[0].data.astype(np.float32)
                    magzero = float(
                        hdul[0].header.get("MAGZERO", band.sim_zeropoint_e))
                factor = np.float32(adu_per_s_to_electrons_factor(magzero, band))
                planes.append((arr * factor).astype(np.float32))
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return planes


@dataclass(frozen=True)
class SRCutout(HRCutout):
    """A super-resolved cutout — output of ``Model.upsample``."""
