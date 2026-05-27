"""
Typed data structures for sky images.

Carrying pixel_scale alongside the data enables validation at convolution
time: the PSF kernel must be sampled at the same scale as the image it
will be convolved with.

:class:`MultiBandSkyImage` is the central sky-image type for the whole
project. It uniformly handles:

  * single-band HST data (``band_names=("VIS",)``, F814W as a VIS
    proxy after the rate→Euclid-electron rescale)
  * 4-band Euclid data (``band_names=("VIS","Y_E","J_E","H_E")``)

Every spatial operation (centre-crop, sum-rebin, PSF convolution,
band noise) is exposed as a method that returns a new
:class:`MultiBandSkyImage` with correctly-updated metadata
(``pixel_scale_arcsec`` after rebin, ``is_clean=False`` after noise,
``band_names`` after single-band extraction, …). Callers that used
to read ``.data`` and call free-floating helpers can now chain on
the class.

Back-compat: ``.data`` is still a plain ``np.ndarray`` accessible
to every existing caller. The class doesn't take that away — it
just adds methods that share one implementation across the
codebase instead of duplicating numpy logic inline.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.sky.noise import apply_band_noise

if TYPE_CHECKING:
    # PSF and ArtifactConfig are only referenced in type hints — keep
    # them behind TYPE_CHECKING so they don't pull their (heavy) modules
    # into runtime import of sky.types. The noise functions imported
    # above don't trigger this because sky.noise is a leaf module that
    # only depends on config + numpy.
    from euclid_polish.psf import PSF
    from euclid_polish.sky.artifacts import ArtifactConfig


# ---------------------------------------------------------------------------
# Multi-band sky image (TFRecord schema v2)
# ---------------------------------------------------------------------------

@dataclass
class MultiBandSkyImage:
    """A multi-channel sky image with explicit band metadata.

    Channel layout is always ``(H, W, C)``. The ``band_names`` tuple names
    each channel in order; for HR target tensors C=1 and band_names=('VIS',),
    for LR input tensors C=4 and band_names=('VIS','Y_E','J_E','H_E').

    A single ``pixel_scale_arcsec`` covers all channels. After the forward
    model resamples NISP onto the VIS LR grid, all four LR channels share
    the same 0.10″/pix scale; HR is 0.05″/pix VIS only.
    """

    data: np.ndarray                       # float32, shape (H, W, C)
    pixel_scale_arcsec: float              # one scale shared by all channels
    band_names: Tuple[str, ...]            # length C, names from Config.LR_INPUT_BAND_NAMES
    is_clean: bool                         # True → HR clean target; False → LR dirty input
    index: Optional[int] = None            # position in the dataset
    subset: Optional[str] = None           # 'train' or 'validate'
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Promote 2-D arrays to 3-D (H, W, 1) so single-channel callers can
        # pass a flat array without an explicit ``[..., np.newaxis]``.
        if self.data.ndim == 2:
            self.data = self.data[..., np.newaxis]
        if self.data.ndim != 3:
            raise ValueError(
                f"MultiBandSkyImage.data must be (H, W, C); got shape {self.data.shape}"
            )
        if len(self.band_names) != self.data.shape[-1]:
            raise ValueError(
                f"band_names has {len(self.band_names)} entries but data has "
                f"{self.data.shape[-1]} channels"
            )

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def num_channels(self) -> int:
        return self.data.shape[-1]

    # TFRecord feature schema v2 (single source of truth)
    _TFRECORD_FEATURES = {
        'image':         tf.io.FixedLenFeature([], tf.string),
        'index':         tf.io.FixedLenFeature([], tf.int64),
        'height':        tf.io.FixedLenFeature([], tf.int64),
        'width':         tf.io.FixedLenFeature([], tf.int64),
        'channels':      tf.io.FixedLenFeature([], tf.int64),
        'pixel_scale':   tf.io.FixedLenFeature([], tf.float32),
        'is_clean':      tf.io.FixedLenFeature([], tf.int64),
        'band_names':    tf.io.FixedLenFeature([], tf.string),  # comma-joined
        'schema_version':tf.io.FixedLenFeature([], tf.int64),
    }

    def to_tfrecord(self, index: int | None = None) -> bytes:
        """Serialize to a v2 TFRecord Example (bytes). Stores data as float32."""
        h, w, c = self.shape
        idx = index if index is not None else (self.index or 0)
        arr_fp32 = np.ascontiguousarray(self.data, dtype=np.float32)
        bytes_value = arr_fp32.tobytes()
        bands_value = ",".join(self.band_names).encode("utf-8")
        feature = {
            'image':          tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_value])),
            'index':          tf.train.Feature(int64_list=tf.train.Int64List(value=[idx])),
            'height':         tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
            'width':          tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
            'channels':       tf.train.Feature(int64_list=tf.train.Int64List(value=[c])),
            'pixel_scale':    tf.train.Feature(float_list=tf.train.FloatList(value=[self.pixel_scale_arcsec])),
            'is_clean':       tf.train.Feature(int64_list=tf.train.Int64List(value=[int(self.is_clean)])),
            'band_names':     tf.train.Feature(bytes_list=tf.train.BytesList(value=[bands_value])),
            'schema_version': tf.train.Feature(int64_list=tf.train.Int64List(value=[Config.TFRECORD_SCHEMA_VERSION])),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    @classmethod
    def from_tfrecord(cls, raw_record) -> MultiBandSkyImage:
        """Parse a single v2 TFRecord example (eager mode)."""
        example = tf.io.parse_single_example(raw_record, cls._TFRECORD_FEATURES)
        h    = int(example['height'].numpy())
        w    = int(example['width'].numpy())
        c    = int(example['channels'].numpy())
        idx  = int(example['index'].numpy())
        ps   = round(float(example['pixel_scale'].numpy()), 6)
        clean= bool(example['is_clean'].numpy())
        bands= tuple(example['band_names'].numpy().decode("utf-8").split(","))
        image_bytes = tf.io.decode_raw(example['image'], tf.float32)
        data = tf.reshape(image_bytes, [h, w, c]).numpy()
        return cls(
            data=data,
            pixel_scale_arcsec=ps,
            band_names=bands,
            is_clean=clean,
            index=idx,
        )

    # ------------------------------------------------------------------
    # Band convenience accessors
    # ------------------------------------------------------------------

    def has_band(self, name: str) -> bool:
        """``True`` if ``name`` is one of this image's channels."""
        return name in self.band_names

    def band_index(self, name: str) -> int:
        """Index of the channel named ``name``. Raises if missing."""
        if name not in self.band_names:
            raise ValueError(
                f"band {name!r} not in {self.band_names}"
            )
        return self.band_names.index(name)

    def plane(self, band: Optional[str] = None) -> np.ndarray:
        """Return a 2-D view (no copy) of one channel, or the full
        ``(H, W, C)`` array when ``band is None``.

        Convenience to replace the inline ``data[..., band_names.index(name)]``
        pattern scattered across renderers.
        """
        if band is None:
            return self.data
        return self.data[..., self.band_index(band)]

    def single_band(self, name: str) -> MultiBandSkyImage:
        """Return a new :class:`MultiBandSkyImage` carrying only ``name``.

        Preserves pixel_scale, is_clean, index, subset, metadata. Output
        shape is ``(H, W, 1)`` with ``band_names=(name,)``.
        """
        k = self.band_index(name)
        return dataclasses.replace(
            self,
            data=self.data[..., k:k + 1].copy(),
            band_names=(name,),
        )

    # ------------------------------------------------------------------
    # Shape / sampling operations
    #
    # The static ``rebin_array`` / ``crop_array`` helpers are exposed
    # on the class so callers that don't have a full
    # :class:`MultiBandSkyImage` at hand (e.g. PSF cubes, raw scene
    # tensors before metadata is attached, the renderer-side residual
    # code) can use the same implementation as the instance methods —
    # one source of truth, no risk of drift between class and
    # free-floating helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def rebin_array(
        arr: np.ndarray,
        factor: int,
        *,
        trim_remainder: bool = False,
    ) -> np.ndarray:
        """Photometric sum-rebin a 2-D or 3-D numpy array by ``factor``.

        Each output pixel is the sum of a ``factor × factor`` block of
        source pixels — total counts preserved. Used by
        :meth:`sum_rebinned` (with ``trim_remainder=False``, strict
        about divisibility) AND by every free-floating helper in the
        codebase that needs to rebin a raw array
        (:meth:`MultiBandForward.sum_rebin` delegates here with
        ``trim_remainder=True``).

        Parameters
        ----------
        arr
            ``(H, W)`` or ``(H, W, C)`` numpy array.
        factor
            Integer ≥ 1. ``factor == 1`` is a no-op (returns a copy).
        trim_remainder
            When the spatial dims aren't divisible by ``factor``:

            * ``False`` (default) — raise ``ValueError``. Safer: silent
              trimming has bitten us before (the 0.05" HR grid is
              normally 510² = divisible by 2/3/5, but a 511² cutout
              would otherwise lose its last row/col with no warning).
            * ``True``  — trim trailing rows / cols to the largest
              divisible size, then rebin. Matches the legacy behaviour
              of :meth:`MultiBandForward.sum_rebin` so callers of that
              static method don't have to change.
        """
        if factor < 1:
            raise ValueError(f"factor must be ≥ 1, got {factor}")
        if factor == 1:
            return np.asarray(arr).copy()
        a = np.asarray(arr)
        if a.ndim not in (2, 3):
            raise ValueError(
                f"expected 2-D or 3-D array, got shape {a.shape}"
            )
        H, W = a.shape[:2]
        if H % factor != 0 or W % factor != 0:
            if not trim_remainder:
                raise ValueError(
                    f"spatial dims {(H, W)} not divisible by factor={factor}"
                )
            Ht = (H // factor) * factor
            Wt = (W // factor) * factor
            a = a[:Ht, :Wt]
            H, W = Ht, Wt
        Hn, Wn = H // factor, W // factor
        if a.ndim == 2:
            return a.reshape(Hn, factor, Wn, factor).sum(axis=(1, 3))
        C = a.shape[2]
        return a.reshape(Hn, factor, Wn, factor, C).sum(axis=(1, 3))

    @staticmethod
    def crop_array(
        arr: np.ndarray,
        side: int,
        *,
        pad_value: float = 0.0,
    ) -> np.ndarray:
        """Centre-crop a 2-D or 3-D array to ``(side, side)`` (per channel
        for 3-D), padding with ``pad_value`` if the input is smaller.

        Same logic as :meth:`centre_cropped_to` but operating on raw
        arrays — so callers like ``_stream_clean_vis_scenes`` that hold
        a numpy view of a single band don't have to wrap-and-unwrap an
        entire :class:`MultiBandSkyImage` just to crop.
        """
        if side < 1:
            raise ValueError(f"side must be ≥ 1, got {side}")
        a = np.asarray(arr)
        if a.ndim not in (2, 3):
            raise ValueError(
                f"expected 2-D or 3-D array, got shape {a.shape}"
            )
        H, W = a.shape[:2]
        if H < side or W < side:
            pad_h = max(0, (side - H + 1) // 2)
            pad_w = max(0, (side - W + 1) // 2)
            pad_widths = ((pad_h, pad_h), (pad_w, pad_w))
            if a.ndim == 3:
                pad_widths = pad_widths + ((0, 0),)
            a = np.pad(
                a, pad_widths, mode="constant",
                constant_values=pad_value,
            )
            H, W = a.shape[:2]
        i0 = (H - side) // 2
        j0 = (W - side) // 2
        if a.ndim == 2:
            return a[i0:i0 + side, j0:j0 + side]
        return a[i0:i0 + side, j0:j0 + side, :]

    def centre_cropped_to(self, side: int) -> MultiBandSkyImage:
        """Centre-crop (or zero-pad up to) a square ``(side, side, C)``.

        Pads symmetrically with zeros when the input is smaller than
        ``side`` along either axis, then takes a centre crop. The output
        is always square. Preserves pixel_scale and channel layout.

        Delegates to :meth:`crop_array` so the implementation isn't
        duplicated.
        """
        return dataclasses.replace(
            self,
            data=self.crop_array(self.data, side).astype(
                self.data.dtype, copy=True,
            ),
        )

    def sum_rebinned(self, factor: int) -> MultiBandSkyImage:
        """Photometric sum-rebin by integer ``factor`` (per channel).

        Strict about divisibility — pre-crop with
        :meth:`centre_cropped_to` if you really mean to lose trailing
        rows/cols. Updates ``pixel_scale_arcsec`` to
        ``factor × source_scale``.

        Delegates to :meth:`rebin_array` (with ``trim_remainder=False``)
        so the implementation isn't duplicated.
        """
        if factor == 1:
            return self
        binned = self.rebin_array(
            self.data, factor, trim_remainder=False,
        )
        return dataclasses.replace(
            self,
            data=binned.astype(self.data.dtype),
            pixel_scale_arcsec=self.pixel_scale_arcsec * float(factor),
        )

    # ------------------------------------------------------------------
    # Physics — PSF convolution
    # ------------------------------------------------------------------

    def convolved_with(self, psf: "PSF") -> MultiBandSkyImage:
        """Convolve every channel with the same PSF.

        Use case: single-band data (HST F814W, ``band_names=("VIS",)``)
        or any application where the same kernel applies to every
        channel. The PSF's ``pixel_scale`` must match this image's
        ``pixel_scale_arcsec`` (raises otherwise — silent scale
        mismatch was the source of more than one historical bug).
        """
        self._validate_psf_scale(psf)
        H, W, C = self.data.shape
        out = np.empty_like(self.data, dtype=np.float32)
        for c in range(C):
            out[..., c] = psf.convolved_with(self.data[..., c])
        return dataclasses.replace(
            self,
            data=out.astype(self.data.dtype),
        )

    def convolved_per_band(
        self, psfs: Dict[str, "PSF"],
    ) -> MultiBandSkyImage:
        """Convolve each channel with its named PSF.

        ``psfs`` must contain a key for every band in
        :attr:`band_names`. Each PSF's ``pixel_scale`` must match this
        image. Use case: 4-band Euclid synthesis where VIS / Y_E /
        J_E / H_E each have their own PSF kernel.
        """
        # Validate all PSFs up front so a missing band errors before
        # we've done any FFTs.
        for band_name in self.band_names:
            if band_name not in psfs:
                raise ValueError(
                    f"missing PSF for band {band_name!r}; provided "
                    f"keys: {list(psfs.keys())}"
                )
            self._validate_psf_scale(psfs[band_name], band_name=band_name)
        H, W, C = self.data.shape
        out = np.empty_like(self.data, dtype=np.float32)
        for c, band_name in enumerate(self.band_names):
            out[..., c] = psfs[band_name].convolved_with(self.data[..., c])
        return dataclasses.replace(
            self,
            data=out.astype(self.data.dtype),
        )

    def _validate_psf_scale(
        self, psf: "PSF", *, band_name: Optional[str] = None,
    ) -> None:
        """Raise ``ValueError`` if the PSF's pixel_scale doesn't match
        the image's. Shared by both convolution entry points so the
        check formats the same error message either way."""
        if abs(psf.pixel_scale - self.pixel_scale_arcsec) > 1e-6:
            tag = f" for band {band_name!r}" if band_name else ""
            raise ValueError(
                f"PSF pixel_scale ({psf.pixel_scale}){tag} doesn't "
                f"match image pixel_scale_arcsec "
                f"({self.pixel_scale_arcsec}); resample the PSF first"
            )

    # ------------------------------------------------------------------
    # Physics — noise injection
    # ------------------------------------------------------------------

    def with_band_noise(
        self,
        rng: np.random.Generator,
        *,
        add_artifacts: bool = False,
        artifact_config: Optional["ArtifactConfig"] = None,
    ) -> MultiBandSkyImage:
        """Apply Euclid-side per-band noise (Poisson + sky + dark + read).

        For each channel, the matching :class:`BandConfig` is looked up
        by ``band_names[c]`` via :meth:`Config.get_band`, and
        :func:`euclid_polish.sky.multiband_forward.apply_band_noise` is
        called with that band's parameters.

        Use case: synthesising Euclid LR observations (the downstream
        super-resolution training target). Marks the result
        ``is_clean=False`` since noised data is no longer clean.

        ``add_artifacts=True`` enables the optional cosmic-ray / hot-pixel
        / interpolation-residual injection from
        :mod:`euclid_polish.sky.artifacts`. Off by default to match the
        existing pair-gen behaviour.
        """
        H, W, C = self.data.shape
        out = np.empty_like(self.data, dtype=np.float32)
        for c, band_name in enumerate(self.band_names):
            band = Config.get_band(band_name)
            out[..., c] = apply_band_noise(
                self.data[..., c], band, rng,
                add_artifacts=add_artifacts,
                artifact_config=artifact_config,
            )
        return dataclasses.replace(
            self,
            data=out.astype(self.data.dtype),
            is_clean=False,
        )

    # ------------------------------------------------------------------
    # Measurements
    # ------------------------------------------------------------------

    def peak(self, band: Optional[str] = None) -> float:
        """Max pixel value across ``band`` (or all channels)."""
        return float(self.plane(band).max())

    def mean(self, band: Optional[str] = None) -> float:
        return float(self.plane(band).mean())

    def total_flux(self, band: Optional[str] = None) -> float:
        """Sum of all pixel values — total electrons over the field."""
        return float(self.plane(band).sum())

    def background_median(
        self,
        band: Optional[str] = None,
        *,
        mask_above: Optional[float] = None,
    ) -> float:
        """Per-channel median, optionally masking pixels above a
        threshold so bright sources don't pull the estimate.

        With ``mask_above`` set, falls back to the unmasked median if
        fewer than 10 pixels remain after masking — protects against
        a saturated cutout where the whole field is "above threshold".
        """
        arr = np.asarray(self.plane(band), dtype=np.float64)
        if mask_above is not None:
            mask = arr < float(mask_above)
            if int(mask.sum()) < 10:
                return float(np.median(arr))
            return float(np.median(arr[mask]))
        return float(np.median(arr))
