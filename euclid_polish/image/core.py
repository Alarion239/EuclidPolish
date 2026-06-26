"""The clean ``Image`` data atom.

``Image`` is the central multi-band sky-image type for the whole project and it
sits at the *bottom* of the import graph: it depends only on third-party libs
(numpy, tensorflow, astropy, matplotlib) plus :mod:`euclid_polish.config` and the
pure provenance value-types. It never imports an operator (simulator, forward
model, trained model, archive). Everything ``Image`` owns is self-contained —
(de)serialization, plotting, crop/rebin, measurements, metrics — so the same
work reads identically from the CLI, the WebUI, the eval runners and scripts.

Transforms that need an operator live on the operator:

    hr = simulator.generate(rng)     # MultiBandSimulator owns generate
    lr = forward.apply(hr)           # MultiBandForward owns the forward op
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

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from astropy.io import fits as _fits

from euclid_polish.config import Config
from euclid_polish.provenance.fits import read_stamp_cards, write_stamp_cards
from euclid_polish.provenance.persistable import StampCarrier
from euclid_polish.provenance.records import Format, Stamp


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
    band_names: Tuple[str, ...]
    is_clean: bool
    role: Role = field(default=Role.UNKNOWN, kw_only=True)
    index: Optional[int] = None
    subset: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim == 2:
            self.data = self.data[..., np.newaxis]
        if self.data.ndim != 3:
            raise ValueError(
                f"Image.data must be (H, W, C); got shape {self.data.shape}")
        if len(self.band_names) != self.data.shape[-1]:
            raise ValueError(
                f"band_names has {len(self.band_names)} entries but data has "
                f"{self.data.shape[-1]} channels")
        # Tolerate a plain string role (e.g. from older constructors).
        if not isinstance(self.role, Role):
            self.role = Role(str(self.role))

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def num_channels(self) -> int:
        return self.data.shape[-1]

    # ------------------------------------------------------------------
    # TFRecord serialization (schema v2/v3 + role)
    # ------------------------------------------------------------------
    #
    # ``prov_*`` carry the optional provenance stamp (schema v3); ``role`` is an
    # additive string. All three have empty defaults so legacy v2 records — and
    # the graph parser, which reads only image/h/w/c — are unaffected.
    _TFRECORD_FEATURES = {
        'image':          tf.io.FixedLenFeature([], tf.string),
        'index':          tf.io.FixedLenFeature([], tf.int64),
        'height':         tf.io.FixedLenFeature([], tf.int64),
        'width':          tf.io.FixedLenFeature([], tf.int64),
        'channels':       tf.io.FixedLenFeature([], tf.int64),
        'pixel_scale':    tf.io.FixedLenFeature([], tf.float32),
        'is_clean':       tf.io.FixedLenFeature([], tf.int64),
        'band_names':     tf.io.FixedLenFeature([], tf.string),
        'schema_version': tf.io.FixedLenFeature([], tf.int64),
        'prov_id':        tf.io.FixedLenFeature([], tf.string, default_value=b""),
        'prov_stamp':     tf.io.FixedLenFeature([], tf.string, default_value=b""),
        'role':           tf.io.FixedLenFeature([], tf.string, default_value=b""),
    }

    def to_tfrecord(self, index: int | None = None) -> bytes:
        """Serialize to a TFRecord Example (bytes). Stores data as float32."""
        h, w, c = self.shape
        idx = index if index is not None else (self.index or 0)
        bytes_value = np.ascontiguousarray(self.data, dtype=np.float32).tobytes()
        bands_value = ",".join(self.band_names).encode("utf-8")
        feature = {
            'image':       tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_value])),
            'index':       tf.train.Feature(int64_list=tf.train.Int64List(value=[idx])),
            'height':      tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
            'width':       tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
            'channels':    tf.train.Feature(int64_list=tf.train.Int64List(value=[c])),
            'pixel_scale': tf.train.Feature(float_list=tf.train.FloatList(value=[self.pixel_scale_arcsec])),
            'is_clean':    tf.train.Feature(int64_list=tf.train.Int64List(value=[int(self.is_clean)])),
            'band_names':  tf.train.Feature(bytes_list=tf.train.BytesList(value=[bands_value])),
            'role':        tf.train.Feature(bytes_list=tf.train.BytesList(value=[self.role.value.encode("utf-8")])),
        }
        if self.stamp is not None:
            emb = self.stamp
            if emb.subset is None and self.subset is not None:
                emb = dataclasses.replace(emb, subset=self.subset)
            feature['prov_id'] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[str(emb.id).encode("utf-8")]))
            feature['prov_stamp'] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[emb.to_json().encode("utf-8")]))
            schema_version = 3
        else:
            schema_version = Config.TFRECORD_SCHEMA_VERSION
        feature['schema_version'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[schema_version]))
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    @classmethod
    def from_tfrecord(cls, raw_record) -> "Image":
        """Parse a single TFRecord example (eager mode)."""
        example = tf.io.parse_single_example(raw_record, cls._TFRECORD_FEATURES)
        h = int(example['height'].numpy())
        w = int(example['width'].numpy())
        c = int(example['channels'].numpy())
        idx = int(example['index'].numpy())
        ps = round(float(example['pixel_scale'].numpy()), 6)
        clean = bool(example['is_clean'].numpy())
        bands = tuple(example['band_names'].numpy().decode("utf-8").split(","))
        data = tf.reshape(tf.io.decode_raw(example['image'], tf.float32),
                          [h, w, c]).numpy()
        role_bytes = example['role'].numpy()
        role = Role(role_bytes.decode("utf-8")) if role_bytes else Role.UNKNOWN
        stamp = None
        subset = None
        prov_stamp_bytes = example['prov_stamp'].numpy()
        if prov_stamp_bytes:
            stamp = Stamp.from_json(prov_stamp_bytes.decode("utf-8"))
            subset = stamp.subset
        return cls(data=data, pixel_scale_arcsec=ps, band_names=bands,
                   is_clean=clean, role=role, index=idx, subset=subset, stamp=stamp)

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
            try:
                write_stamp_cards(hdr, self.stamp)
            except Exception:   # noqa: BLE001 — provenance cards are best-effort
                pass
        _fits.PrimaryHDU(arr, header=hdr).writeto(
            path, overwrite=True, output_verify="silentfix")

    @classmethod
    def from_fits(cls, path: str) -> "Image":
        """Read an Image written by :meth:`save_fits` (recovers role + stamp)."""
        with _fits.open(path) as hdul:
            hdr = hdul[0].header
            arr = np.asarray(hdul[0].data, dtype=np.float32)
            if arr.ndim == 3:
                arr = np.ascontiguousarray(np.moveaxis(arr, 0, -1))
            bands_card = hdr.get("BANDS")
            ps = float(hdr.get("PIXSCALE", Config.DEFAULT_PIXEL_SCALE))
            clean = bool(hdr.get("ISCLEAN", True))
            role = Role(str(hdr.get("IMGROLE", "unknown")).strip())
            stamp = read_stamp_cards(hdr)
        c = arr.shape[-1] if arr.ndim == 3 else 1
        bands = tuple(bands_card.split(",")) if bands_card else None
        if bands is None or len(bands) != c:
            bands = ("VIS",) if c == 1 else tuple(f"b{i}" for i in range(c))
        return cls(data=arr, pixel_scale_arcsec=ps, band_names=bands,
                   is_clean=clean, role=role, stamp=stamp)

    # ------------------------------------------------------------------
    # Band convenience accessors
    # ------------------------------------------------------------------

    def has_band(self, name: str) -> bool:
        return name in self.band_names

    def band_index(self, name: str) -> int:
        if name not in self.band_names:
            raise ValueError(f"band {name!r} not in {self.band_names}")
        return self.band_names.index(name)

    def plane(self, band: Optional[str] = None) -> np.ndarray:
        """A 2-D view of one channel, or the full ``(H, W, C)`` when ``band is None``."""
        if band is None:
            return self.data
        return self.data[..., self.band_index(band)]

    def single_band(self, name: str) -> "Image":
        """A new :class:`Image` carrying only channel ``name`` (shape ``(H, W, 1)``)."""
        k = self.band_index(name)
        return dataclasses.replace(
            self, data=self.data[..., k:k + 1].copy(), band_names=(name,))

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
        if H < side or W < side:
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

    def centre_cropped_to(self, side: int) -> "Image":
        """Centre-crop (or zero-pad up to) a square ``(side, side, C)``."""
        return dataclasses.replace(
            self, data=self.crop_array(self.data, side).astype(self.data.dtype, copy=True))

    def sum_rebinned(self, factor: int) -> "Image":
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

    def peak(self, band: Optional[str] = None) -> float:
        return float(self.plane(band).max())

    def mean(self, band: Optional[str] = None) -> float:
        return float(self.plane(band).mean())

    def total_flux(self, band: Optional[str] = None) -> float:
        return float(self.plane(band).sum())

    def background_median(self, band: Optional[str] = None, *,
                          mask_above: Optional[float] = None) -> float:
        """Per-channel median, optionally masking pixels above a threshold."""
        arr = np.asarray(self.plane(band), dtype=np.float64)
        if mask_above is not None:
            mask = arr < float(mask_above)
            if int(mask.sum()) < 10:
                return float(np.median(arr))
            return float(np.median(arr[mask]))
        return float(np.median(arr))

    def psnr_against(self, other: "Image", *, data_range: Optional[float] = None) -> float:
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

    # ------------------------------------------------------------------
    # Plotting (self-contained quick-look)
    # ------------------------------------------------------------------

    def plot(self, path: Optional[str] = None, *, band: Optional[str] = None,
             asinh: bool = True, cmap: str = "gray"):
        """Render a single-band quick-look.

        Saves a PNG to ``path`` (and returns it) when given, else draws into a
        fresh Axes and returns it. ``asinh`` applies an MAD-scaled arcsinh
        stretch so faint structure is visible.
        """
        arr = np.asarray(
            self.plane(band if band is not None else self.band_names[0]),
            dtype=np.float64)
        if asinh:
            scale = float(np.median(np.abs(arr - np.median(arr)))) or 1.0
            disp = np.arcsinh(arr / scale)
        else:
            disp = arr
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(disp, origin="lower", cmap=cmap)
        ax.set_axis_off()
        if path is not None:
            fig.savefig(path, bbox_inches="tight", dpi=100)
            plt.close(fig)
            return path
        return ax
