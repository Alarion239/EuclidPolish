"""The behaviour-bearing cutout hierarchy.

``Cutout`` *composes* a :class:`~euclid_polish.sky.types.MultiBandSkyImage`
(has-a, not is-a): the image stays the pure pixel / serialization / physics
workhorse, while the cutout is the typed handle that carries identity +
provenance and owns the verb that produced it. Each leaf delegates its verb to
the existing engine through an injected callable, so the type layer never
duplicates the download / forward-model / reconstruct logic.

    EuclidLRCutout.fetch(ra, dec, size)    -> the Euclid archive
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
from euclid_polish.provenance.fits import write_stamp_cards
from euclid_polish.euclid.downloader import fetch_cutout_at
from euclid_polish.euclid.photometry import adu_per_s_to_electrons_factor
from euclid_polish.provenance.defaults import default_store
from euclid_polish.provenance.ids import ProvId
from euclid_polish.provenance.records import Format, Stamp
from euclid_polish.provenance.store import ProvStore
from euclid_polish.sky.tfrecord import write_multiband_skyimages
from euclid_polish.sky.types import MultiBandSkyImage
from euclid_polish.training.inference import plot_reconstruction as _plot_reconstruction

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

    def write_tfrecord(self, records_dir: str, name: str) -> str:
        """Write this cutout as a single-element TFRecord; return the path.

        Delegates to
        :func:`~euclid_polish.sky.tfrecord.write_multiband_skyimages`,
        embedding this cutout's provenance stamp in the record. The file is
        ``<records_dir>/<name>.tfrecord`` (``records_dir`` is created if absent).

        Parameters
        ----------
        records_dir : str
            Output directory.
        name : str
            Record-set name (e.g. ``"hr_train"``).
        """
        return write_multiband_skyimages(
            [self.stamped_image()], name=name, records_dir=records_dir,
        )

    def save_fits(self, path: str, *, wcs_header=None) -> None:
        """Write this cutout's data to a FITS file with provenance cards.

        Pixel data is written as float32. Multi-band ``(H, W, C)`` data is
        stored as a 3D FITS image with ``C`` planes on NAXIS3 (channel order
        = :attr:`band_names`). A ``PROVID`` card (and the rest of the
        provenance stamp) is embedded.

        Parameters
        ----------
        path : str
            Output FITS path (parent directory must exist).
        wcs_header : astropy.io.fits.Header, optional
            Base header to copy (e.g. a scaled-WCS header for SR, or the VIS
            FITS header for LR). A blank header is used when None.
        """
        arr = np.asarray(self.data, dtype=np.float32)
        hdr = wcs_header.copy() if wcs_header is not None else _fits.Header()
        for _bad in ("EXTNAME", "XTENSION"):
            if _bad in hdr:
                del hdr[_bad]
        if arr.ndim == 3:
            arr = np.ascontiguousarray(np.moveaxis(arr, -1, 0))
        hdr["BUNIT"] = ("electron", "Raw electrons (sign preserved)")
        if self.band_names:
            hdr["BANDS"] = (",".join(self.band_names),
                            "Band order (NAXIS3 plane index 0 = first band)")
        try:
            write_stamp_cards(hdr, self.prov_stamp())
        except Exception:   # noqa: BLE001 — provenance cards are best-effort
            pass
        hdu = _fits.PrimaryHDU(arr, header=hdr)
        hdu.writeto(path, overwrite=True, output_verify="silentfix")


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
        LR cutout's parent is this HR cutout.

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

    def save_png(
        self,
        path: str,
        *,
        regime: str = "eye",
        lr: "Optional[LRCutout]" = None,
        hr: "Optional[HRCutout]" = None,
    ) -> None:
        """Render an LR→SR(→HR) reconstruction figure as a PNG.

        Wraps :func:`~euclid_polish.training.inference.plot_reconstruction`.

        Parameters
        ----------
        path : str
            Output PNG path.
        regime : str
            Colour regime — ``"eye"`` (physical blackbody-T colours, default)
            or ``"calibrated"`` (solar-balanced adaptive windows).
        lr : LRCutout, optional
            The dirty LR input. When None a 2× average-pooled SR-VIS proxy is
            used for the LR panel.
        hr : HRCutout, optional
            Ground-truth HR target. When given, the with-HR layout + residual
            metrics are shown.
        """
        sr_data = np.asarray(self.data, dtype=np.float32)

        lr_data = lr_cube = None
        if lr is not None:
            lr_arr = np.asarray(lr.data, dtype=np.float32)
            if lr_arr.ndim == 3 and lr_arr.shape[-1] > 1:
                lr_cube = lr_arr
                lr_data = lr_arr[..., 0]
            else:
                lr_data = lr_arr[..., 0] if lr_arr.ndim == 3 else lr_arr
        if lr_data is None:
            vis = sr_data[..., 0] if sr_data.ndim == 3 else sr_data
            h, w = vis.shape[:2]
            # no LR supplied: trim to even dims and 2x average-pool the SR VIS plane as a display proxy
            lr_data = vis[: h - h % 2, : w - w % 2].reshape(
                h // 2, 2, w // 2, 2).mean(axis=(1, 3))

        hr_data = hr_cube = None
        if hr is not None:
            hr_arr = np.asarray(hr.data, dtype=np.float32)
            if hr_arr.ndim == 3 and hr_arr.shape[-1] > 1:
                hr_cube = hr_arr
                hr_data = hr_arr[..., 0]
            else:
                hr_data = hr_arr[..., 0] if hr_arr.ndim == 3 else hr_arr

        _plot_reconstruction(
            lr_data=lr_data,
            sr_data=sr_data,
            hr_data=hr_data,
            output_path=path,
            lr_cube=lr_cube,
            hr_cube=hr_cube,
            rgb_mode=regime,
        )
