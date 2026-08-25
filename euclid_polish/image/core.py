"""The clean ``Image`` data atom.

``Image`` is the central multi-band sky-image type for the whole project and it
sits at the *bottom* of the import graph: it depends only on third-party libs
(numpy, tensorflow, astropy) plus :mod:`euclid_polish.config` and the pure
provenance value-types. It never imports an operator (simulator, forward
model, trained model, archive). Everything ``Image`` owns is self-contained —
(de)serialization, crop/rebin, measurements, metrics — so the same work reads
identically from the CLI, the WebUI, the eval runners and scripts.

Transforms that need an operator live on the operator:

    hr = simulator.generate(rng)     # SkySimulator owns generate
    lr = forward.apply(hr)           # ObservationSimulator owns the forward op
    sr = model.upsample(lr)          # Model owns upsample

Channel layout is always ``(H, W, C)``; ``band_names`` names each channel in
order and a single ``pixel_scale_arcsec`` covers all channels. ``role`` tags
what kind of image this is (clean / hr / lr / sr / real) — it replaces the old
HR/LR/SR/Euclid subclass tree. An optional provenance :class:`Stamp` rides
inside the TFRecord / FITS when present, so a bare file is self-identifying.

This class was historically named ``MultiBandSkyImage`` and lived in
``euclid_polish.sky.types``; that module has been removed and all call sites now
use ``from euclid_polish.image import Image``.
"""

from __future__ import annotations

import contextlib
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

import numpy as np
import tensorflow as tf
from astropy.io import fits as _fits

from euclid_polish.config import Config
from euclid_polish.image.cube import AngularGrid, PixelUnit
from euclid_polish.provenance.fits import read_stamp_cards, write_stamp_cards
from euclid_polish.provenance.persistable import StampCarrier
from euclid_polish.provenance.records import Format, Stamp


@dataclass
class FitsWCS:
    """WCS fields extracted from a real Euclid FITS header.

    All fields are Optional — an all-None instance means no WCS was available.
    Call :meth:`to_header` to get an astropy Header suitable for
    :func:`~euclid_polish.training.inference.scaled_wcs_header`.
    """

    crpix1: float | None = None   # reference pixel, axis 1
    crpix2: float | None = None   # reference pixel, axis 2
    crval1: float | None = None   # reference RA (deg)
    crval2: float | None = None   # reference Dec (deg)
    cd1_1: float | None = None    # CD matrix row 1 col 1
    cd1_2: float | None = None    # CD matrix row 1 col 2
    cd2_1: float | None = None    # CD matrix row 2 col 1
    cd2_2: float | None = None    # CD matrix row 2 col 2
    cdelt1: float | None = None   # pixel scale axis 1 (deg) — CDELT form
    cdelt2: float | None = None   # pixel scale axis 2 (deg) — CDELT form
    ctype1: str | None = None     # e.g. "RA---TAN"
    ctype2: str | None = None     # e.g. "DEC--TAN"
    cunit1: str | None = None     # e.g. "deg"
    cunit2: str | None = None     # e.g. "deg"

    def to_header(self) -> _fits.Header:
        """Build an astropy Header containing the non-None WCS cards."""
        hdr = _fits.Header()
        mapping = {
            "CRPIX1": self.crpix1, "CRPIX2": self.crpix2,
            "CRVAL1": self.crval1, "CRVAL2": self.crval2,
            "CD1_1":  self.cd1_1,  "CD1_2":  self.cd1_2,
            "CD2_1":  self.cd2_1,  "CD2_2":  self.cd2_2,
            "CDELT1": self.cdelt1, "CDELT2": self.cdelt2,
            "CTYPE1": self.ctype1, "CTYPE2": self.ctype2,
            "CUNIT1": self.cunit1, "CUNIT2": self.cunit2,
        }
        for key, value in mapping.items():
            if value is not None:
                hdr[key] = value
        return hdr

    @classmethod
    def from_header(cls, header) -> FitsWCS:
        """Extract WCS fields from an astropy FITS Header (missing → None)."""
        return cls(
            crpix1=header.get("CRPIX1"), crpix2=header.get("CRPIX2"),
            crval1=header.get("CRVAL1"), crval2=header.get("CRVAL2"),
            cd1_1=header.get("CD1_1"),   cd1_2=header.get("CD1_2"),
            cd2_1=header.get("CD2_1"),   cd2_2=header.get("CD2_2"),
            cdelt1=header.get("CDELT1"), cdelt2=header.get("CDELT2"),
            ctype1=header.get("CTYPE1"), ctype2=header.get("CTYPE2"),
            cunit1=header.get("CUNIT1"), cunit2=header.get("CUNIT2"),
        )


class Role(str, Enum):
    """What kind of image this is — gates which operator verb applies.

    The single tag that replaced the old ``HRCutout``/``LRCutout``/``SRCutout``/
    ``EuclidLRCutout`` subclass tree. ``str`` base so the value serialises as a
    plain string in TFRecord / FITS. Unknown values fall back to ``UNKNOWN``.
    """

    CLEAN = "clean"     # clean high-res scene straight from the simulator
    HR = "hr"           # high-res training target
    LR = "lr"           # low-res dirty input (synthetic forward-model output)
    SR = "sr"           # super-resolved model output
    REAL = "real"       # real Euclid cutout from the archive
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value):  # noqa: D401 — Enum hook
        return cls.UNKNOWN


@dataclass
class Image(StampCarrier):
    """A multi-channel sky image with explicit band + provenance metadata.

    Parameters
    ----------
    data : np.ndarray
        ``(H, W, C)`` float32 pixels. 2-D input is promoted to ``(H, W, 1)``.
    pixel_scale_arcsec : float
        One scale shared by all channels.
    band_names : tuple of str
        Length ``C``; names each channel in order.
    is_clean : bool
        ``True`` → clean target; ``False`` → noised/dirty input.
    role : Role, keyword-only
        The image's role (default :attr:`Role.UNKNOWN`).
    index, subset, metadata
        Dataset position / split / free-form metadata (in-memory only).
    stamp : Stamp, keyword-only
        Optional provenance stamp (inherited from ``StampCarrier``).
    """

    PROV_FORMAT: ClassVar[Format] = Format.TFRECORD

    data: np.ndarray
    pixel_scale_arcsec: float
    band_names: tuple[str, ...]
    is_clean: bool
    role: Role = field(default=Role.UNKNOWN, kw_only=True)
    fits_wcs: FitsWCS | None = field(default=None, kw_only=True)
    index: int | None = None
    subset: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim == 2:
            self.data = self.data[..., np.newaxis]
        if self.data.ndim != 3:
            raise ValueError(
                f"Image.data must be (H, W, C); got shape {self.data.shape}")
        scale = float(self.pixel_scale_arcsec)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "pixel_scale_arcsec must be finite and positive, "
                f"got {self.pixel_scale_arcsec!r}"
            )
        self.pixel_scale_arcsec = scale
        bands = tuple(str(name) for name in self.band_names)
        if len(bands) != self.data.shape[-1]:
            raise ValueError(
                f"band_names has {len(bands)} entries but data has "
                f"{self.data.shape[-1]} channels")
        if any(not name.strip() for name in bands):
            raise ValueError("band_names must contain non-empty strings")
        if len(set(bands)) != len(bands):
            raise ValueError(f"band_names must be unique, got {bands!r}")
        self.band_names = bands
        # Tolerate a plain string role (e.g. from older constructors).
        if not isinstance(self.role, Role):
            self.role = Role(str(self.role))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def bands(self) -> tuple[str, ...]:
        """Band-name spelling shared with dependency-light image cubes."""
        return self.band_names

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.data.shape[:2]  # type: ignore[return-value]

    @property
    def num_channels(self) -> int:
        return self.data.shape[-1]

    @property
    def unit(self) -> PixelUnit:
        """Pixel calibration used by persisted and simulated project images."""
        return PixelUnit.ELECTRONS_PER_PIXEL

    @property
    def grid(self) -> AngularGrid:
        """Angular sampling represented by ``pixel_scale_arcsec``."""
        return AngularGrid(self.pixel_scale_arcsec)

    # ------------------------------------------------------------------
    # TFRecord serialization
    # ------------------------------------------------------------------
    #
    # One fixed schema, every field always written. ``prov_id``/``prov_stamp``
    # are empty strings for an unstamped image. The graph training path uses a
    # smaller feature subset (see ``image.tfio.parse_example``).
    _TFRECORD_FEATURES = {
        'image':       tf.io.FixedLenFeature([], tf.string),
        'index':       tf.io.FixedLenFeature([], tf.int64),
        'height':      tf.io.FixedLenFeature([], tf.int64),
        'width':       tf.io.FixedLenFeature([], tf.int64),
        'channels':    tf.io.FixedLenFeature([], tf.int64),
        'pixel_scale': tf.io.FixedLenFeature([], tf.float32),
        'is_clean':    tf.io.FixedLenFeature([], tf.int64),
        'band_names':  tf.io.FixedLenFeature([], tf.string),
        'role':        tf.io.FixedLenFeature([], tf.string),
        'prov_id':     tf.io.FixedLenFeature([], tf.string),
        'prov_stamp':  tf.io.FixedLenFeature([], tf.string),
    }

    def to_tfrecord(self, index: int | None = None) -> bytes:
        """Serialize to a TFRecord Example (bytes). Stores data as float32."""
        h, w, c = self.shape
        idx = index if index is not None else (self.index or 0)
        if self.stamp is not None:
            emb = self.stamp
            if emb.subset is None and self.subset is not None:
                emb = dataclasses.replace(emb, subset=self.subset)
            prov_id = str(emb.id).encode("utf-8")
            prov_stamp = emb.to_json().encode("utf-8")
        else:
            prov_id = prov_stamp = b""

        def _b(v):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

        def _i(v):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))

        feature = {
            'image':       _b(np.ascontiguousarray(self.data, dtype=np.float32).tobytes()),
            'index':       _i(idx),
            'height':      _i(h),
            'width':       _i(w),
            'channels':    _i(c),
            'pixel_scale': tf.train.Feature(
                float_list=tf.train.FloatList(value=[self.pixel_scale_arcsec])),
            'is_clean':    _i(int(self.is_clean)),
            'band_names':  _b(",".join(self.band_names).encode("utf-8")),
            'role':        _b(self.role.value.encode("utf-8")),
            'prov_id':     _b(prov_id),
            'prov_stamp':  _b(prov_stamp),
        }
        return tf.train.Example(
            features=tf.train.Features(feature=feature)).SerializeToString()

    @classmethod
    def from_tfrecord(cls, raw_record) -> Image:
        """Parse a single TFRecord example (eager mode)."""
        ex = tf.io.parse_single_example(raw_record, cls._TFRECORD_FEATURES)
        h = int(ex['height'].numpy())
        w = int(ex['width'].numpy())
        c = int(ex['channels'].numpy())
        data = tf.reshape(tf.io.decode_raw(ex['image'], tf.float32),
                          [h, w, c]).numpy()
        role = Role(ex['role'].numpy().decode("utf-8"))
        stamp = subset = None
        prov_stamp_bytes = ex['prov_stamp'].numpy()
        if prov_stamp_bytes:
            stamp = Stamp.from_json(prov_stamp_bytes.decode("utf-8"))
            subset = stamp.subset
        return cls(
            data=data,
            pixel_scale_arcsec=round(float(ex['pixel_scale'].numpy()), 6),
            band_names=tuple(ex['band_names'].numpy().decode("utf-8").split(",")),
            is_clean=bool(ex['is_clean'].numpy()),
            role=role, index=int(ex['index'].numpy()), subset=subset, stamp=stamp)

    # ------------------------------------------------------------------
    # FITS serialization (self-contained, round-trips role + provenance)
    # ------------------------------------------------------------------

    def save_fits(self, path: str, *, wcs_header=None) -> None:
        """Write to a FITS file with band/scale/role + provenance cards.

        Multi-band ``(H, W, C)`` data is stored as a 3-D image with ``C`` planes
        on NAXIS3 (plane index 0 = first band). ``wcs_header`` (optional) is a
        base header to copy — e.g. a scaled-WCS header for an SR image.
        """
        arr = np.asarray(self.data, dtype=np.float32)
        hdr = wcs_header.copy() if wcs_header is not None else _fits.Header()
        for _bad in ("EXTNAME", "XTENSION"):
            if _bad in hdr:
                del hdr[_bad]
        if arr.ndim == 3:
            arr = np.ascontiguousarray(np.moveaxis(arr, -1, 0))
        hdr["BUNIT"] = ("electron", "Raw electrons (sign preserved)")
        hdr["PIXSCALE"] = (float(self.pixel_scale_arcsec), "arcsec/pixel")
        hdr["ISCLEAN"] = (bool(self.is_clean), "Clean target (vs dirty)")
        hdr["IMGROLE"] = (self.role.value, "Image role")
        if self.band_names:
            hdr["BANDS"] = (",".join(self.band_names),
                            "Band order (NAXIS3 plane 0 = first band)")
        if self.stamp is not None:
            # provenance cards are best-effort
            with contextlib.suppress(Exception):
                write_stamp_cards(hdr, self.stamp)
        _fits.PrimaryHDU(arr, header=hdr).writeto(
            path, overwrite=True, output_verify="silentfix")

    @classmethod
    def from_fits(cls, path: str) -> Image:
        """Read an Image written by :meth:`save_fits` (recovers role + stamp)."""
        with _fits.open(path) as hdul:
            primary = hdul[0]
            if not isinstance(primary, _fits.PrimaryHDU):
                raise ValueError(f"first FITS extension is not primary: {path}")
            if primary.data is None:
                raise ValueError(f"empty primary FITS image: {path}")
            hdr = primary.header
            arr = np.asarray(primary.data, dtype=np.float32)
            if arr.ndim == 3:
                arr = np.ascontiguousarray(np.moveaxis(arr, 0, -1))
            bands_card = hdr.get("BANDS")
            ps = float(str(hdr.get("PIXSCALE", Config.DEFAULT_PIXEL_SCALE)))
            clean = bool(hdr.get("ISCLEAN", True))
            role = Role(str(hdr.get("IMGROLE", "unknown")).strip())
            stamp = read_stamp_cards(hdr)
            fits_wcs = FitsWCS.from_header(hdr)
        c = arr.shape[-1] if arr.ndim == 3 else 1
        bands_text = str(bands_card) if bands_card is not None else ""
        bands = tuple(bands_text.split(",")) if bands_text else None
        if bands is None or len(bands) != c:
            bands = ("VIS",) if c == 1 else tuple(f"b{i}" for i in range(c))
        return cls(data=arr, pixel_scale_arcsec=ps, band_names=bands,
                   is_clean=clean, role=role, stamp=stamp, fits_wcs=fits_wcs)

    # ------------------------------------------------------------------
    # Raw-bytes export (the WebUI cutout-viewer wire format)
    # ------------------------------------------------------------------

    def to_raw_bytes(self) -> bytes:
        """The pixels as a little-endian float32, C-contiguous ``(H, W, C)`` buffer.

        This is the canonical wire format the browser cutout viewer
        (``static/cutout_viewer.js``) reads straight into a ``Float32Array``:
        it is exactly the body of the ``/viewer/cube`` response. Pair it with
        :meth:`wire_meta` for the shape/band header. No HTTP or framework types
        are involved, so this stays a leaf concern.
        """
        return np.ascontiguousarray(self.data, dtype="<f4").tobytes(order="C")

    @classmethod
    def from_raw_bytes(cls, buf, *, shape, band_names, pixel_scale_arcsec,
                       is_clean: bool = False, role: Role = Role.UNKNOWN,
                       stamp: Stamp | None = None) -> Image:
        """Rebuild an Image from a :meth:`to_raw_bytes` buffer + its shape/bands.

        The inverse of :meth:`to_raw_bytes` — e.g. to reconstruct an Image from
        a viewer-cube payload (body + ``X-Cube-Shape``/``X-Cube-Bands``).
        """
        data = np.frombuffer(buf, dtype="<f4").reshape(tuple(shape)).astype(np.float32)
        return cls(data=data, pixel_scale_arcsec=float(pixel_scale_arcsec),
                   band_names=tuple(band_names), is_clean=is_clean, role=role,
                   stamp=stamp)

    def wire_meta(self) -> dict:
        """Framework-free descriptor for the viewer wire (no Flask/HTTP types).

        ``{"shape": (H, W, C), "band_names": [...], "pixel_scale_arcsec": ...}``.
        The route maps these onto its ``X-Cube-*`` headers; keeping the method
        framework-free preserves the leaf property.
        """
        h, w, c = self.shape
        return {
            "shape": (h, w, c),
            "band_names": list(self.band_names),
            "pixel_scale_arcsec": float(self.pixel_scale_arcsec),
        }

    # ------------------------------------------------------------------
    # Band convenience accessors
    # ------------------------------------------------------------------

    def band_index(self, name: str) -> int:
        if name not in self.band_names:
            raise ValueError(f"band {name!r} not in {self.band_names}")
        return self.band_names.index(name)

    def plane(self, band: str | None = None) -> np.ndarray:
        """Return one 2-D channel plane without copying it.

        The band may be omitted only for a single-channel image, matching the
        dependency-light :class:`~euclid_polish.image.ImageCube` contract.
        """
        if band is None:
            if self.num_channels != 1:
                raise ValueError("band is required for a multi-channel image")
            return self.data[..., 0]
        return self.data[..., self.band_index(band)]

    def with_role(self, role: Role) -> Image:
        """A copy tagged with ``role`` (the original is unchanged).

        ``is_clean`` is derived from ``role``: clean target roles (CLEAN, HR,
        SR) set it to ``True``; noisy/dirty roles (LR, REAL, UNKNOWN) set it
        to ``False``.
        """
        is_clean = role in (Role.CLEAN, Role.HR, Role.SR)
        return dataclasses.replace(self, role=role, is_clean=is_clean)

    # ------------------------------------------------------------------
    # Shape / sampling — pure-numpy geometry (no operator needed)
    # ------------------------------------------------------------------

    @staticmethod
    def rebin_array(arr: np.ndarray, factor: int, *,
                    trim_remainder: bool = False) -> np.ndarray:
        """Photometric sum-rebin a 2-D/3-D array by ``factor`` (counts preserved).

        ``trim_remainder=False`` (default) raises when the spatial dims aren't
        divisible by ``factor``; ``True`` trims trailing rows/cols first.
        """
        if factor < 1:
            raise ValueError(f"factor must be ≥ 1, got {factor}")
        if factor == 1:
            return np.asarray(arr).copy()
        a = np.asarray(arr)
        if a.ndim not in (2, 3):
            raise ValueError(f"expected 2-D or 3-D array, got shape {a.shape}")
        H, W = a.shape[:2]
        if H % factor != 0 or W % factor != 0:
            if not trim_remainder:
                raise ValueError(
                    f"spatial dims {(H, W)} not divisible by factor={factor}")
            Ht, Wt = (H // factor) * factor, (W // factor) * factor
            a = a[:Ht, :Wt]
            H, W = Ht, Wt
        Hn, Wn = H // factor, W // factor
        if a.ndim == 2:
            return a.reshape(Hn, factor, Wn, factor).sum(axis=(1, 3))
        C = a.shape[2]
        return a.reshape(Hn, factor, Wn, factor, C).sum(axis=(1, 3))

    @staticmethod
    def crop_array(arr: np.ndarray, side: int, *, pad_value: float = 0.0) -> np.ndarray:
        """Centre-crop a 2-D/3-D array to ``(side, side)``, padding if smaller."""
        if side < 1:
            raise ValueError(f"side must be ≥ 1, got {side}")
        a = np.asarray(arr)
        if a.ndim not in (2, 3):
            raise ValueError(f"expected 2-D or 3-D array, got shape {a.shape}")
        H, W = a.shape[:2]
        if side > H or side > W:
            pad_h = max(0, (side - H + 1) // 2)
            pad_w = max(0, (side - W + 1) // 2)
            pad_widths = ((pad_h, pad_h), (pad_w, pad_w))
            if a.ndim == 3:
                pad_widths = pad_widths + ((0, 0),)
            a = np.pad(a, pad_widths, mode="constant", constant_values=pad_value)
            H, W = a.shape[:2]
        i0, j0 = (H - side) // 2, (W - side) // 2
        if a.ndim == 2:
            return a[i0:i0 + side, j0:j0 + side]
        return a[i0:i0 + side, j0:j0 + side, :]

    def centre_cropped_to(self, side: int) -> Image:
        """Centre-crop (or zero-pad up to) a square ``(side, side, C)``."""
        return dataclasses.replace(
            self, data=self.crop_array(self.data, side).astype(self.data.dtype, copy=True))

    def sum_rebinned(self, factor: int) -> Image:
        """Photometric sum-rebin by integer ``factor``; updates ``pixel_scale_arcsec``."""
        if factor == 1:
            return self
        binned = self.rebin_array(self.data, factor, trim_remainder=False)
        return dataclasses.replace(
            self, data=binned.astype(self.data.dtype),
            pixel_scale_arcsec=self.pixel_scale_arcsec * float(factor))

    # ------------------------------------------------------------------
    # Measurements & metrics (self-contained)
    # ------------------------------------------------------------------

    def peak(self, band: str | None = None) -> float:
        return float(self.plane(band).max())

    def mean(self, band: str | None = None) -> float:
        return float(self.plane(band).mean())

    def total_flux(self, band: str | None = None) -> float:
        return float(self.plane(band).sum())

    def psnr_against(self, other: Image, *, data_range: float | None = None) -> float:
        """Peak signal-to-noise ratio (dB) of ``self`` vs ``other``.

        Identical arrays give ``inf``. ``data_range`` defaults to the dynamic
        range of ``self`` (``max - min``, or 1.0 if flat).
        """
        a = np.asarray(self.data, dtype=np.float64)
        b = np.asarray(other.data, dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
        mse = float(np.mean((a - b) ** 2))
        if mse == 0.0:
            return float("inf")
        if data_range is None:
            data_range = float(a.max() - a.min()) or 1.0
        return float(20.0 * np.log10(data_range / np.sqrt(mse)))
