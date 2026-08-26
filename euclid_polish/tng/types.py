"""Typed value objects for TNG atlas views and rendered stamps.

The numerical renderer deliberately has two distinct image domains:

* native TNG atlas images are surface brightness in MJy/sr on a physical pc/pixel
  grid; and
* rendered TNG stamps are electrons/pixel on the simulation's angular grid.

The records in this module make that boundary explicit without subclassing
``numpy.ndarray`` or storing an open-ended compatibility metadata dictionary.
They remain dependency-light: FITS I/O and stamp compositing are imported only
inside the methods that perform those operations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from euclid_polish.config import Config
from euclid_polish.image import Image, Role
from euclid_polish.photometry import electrons_to_ab_mag
from euclid_polish.tng.image import TNGSurfaceBrightnessImage

TNG_FITS_BANDS: tuple[str, ...] = ("VIS", "Y", "J", "H")
TNG_MODEL_BANDS: tuple[str, ...] = tuple(Config.LR_INPUT_BAND_NAMES)
#: Every atlas galaxy has these five viewing orientations.
N_ORIENTATIONS = 5
#: Native physical pixel pitch of every supported TNG atlas plane.
TNG_NATIVE_PC_PER_PIXEL = 100.0

type FileIdentity = tuple[tuple[str, int, int], ...]
type DriftMode = Literal["sed_interp", "parametric"]


def _positive_finite(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return number


def _nonnegative_finite(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return number


def _four_positive(
    values: Sequence[float],
    name: str,
) -> tuple[float, float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != len(TNG_MODEL_BANDS):
        raise ValueError(f"{name} must contain four values, got {len(result)}")
    if not all(np.isfinite(value) and value > 0.0 for value in result):
        raise ValueError(f"{name} must contain finite positive values")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class TNGView:
    """One specific SKIRT atlas galaxy and viewing orientation.

    Construction is metadata-only. FITS files are resolved and opened lazily by
    :meth:`load_surface_brightness`, which keeps donor enumeration cheap.
    """

    galaxy_dir: Path
    subhalo_id: str
    orientation: int
    native_re_px: float
    radius_manifest_fingerprint: str = ""

    def __post_init__(self) -> None:
        directory = Path(self.galaxy_dir)
        subhalo_id = str(self.subhalo_id).strip()
        if not subhalo_id:
            raise ValueError("subhalo_id must be non-empty")
        if isinstance(self.orientation, bool) or not isinstance(self.orientation, int):
            raise TypeError("orientation must be an integer in 1..5")
        if not 1 <= self.orientation <= 5:
            raise ValueError(f"orientation must be in 1..5, got {self.orientation!r}")
        object.__setattr__(self, "galaxy_dir", directory)
        object.__setattr__(self, "subhalo_id", subhalo_id)
        object.__setattr__(
            self,
            "native_re_px",
            _positive_finite(self.native_re_px, "native_re_px"),
        )
        object.__setattr__(
            self,
            "radius_manifest_fingerprint",
            str(self.radius_manifest_fingerprint),
        )

    def fits_path(self, fits_band: str) -> Path:
        """Resolve one atlas band, including the archive's zero-padded variant."""
        from euclid_polish.tng.atlas import _resolve_fits_path

        return _resolve_fits_path(
            self.galaxy_dir,
            self.subhalo_id,
            self.orientation,
            fits_band,
        )

    def native_re_arcsec(self, pixel_scale_arcsec: float) -> float:
        """Nominal native half-light radius when assigned to an angular grid."""
        scale = _positive_finite(pixel_scale_arcsec, "pixel_scale_arcsec")
        return self.native_re_px * scale

    def can_render(self, target_re_arcsec: float, pixel_scale_arcsec: float) -> bool:
        """Whether the requested radius is reachable without donor enlargement."""
        target = _positive_finite(target_re_arcsec, "target_re_arcsec")
        return self.native_re_arcsec(pixel_scale_arcsec) >= target

    def file_identity(self) -> FileIdentity:
        """Stable cache identity for the four contributing FITS files."""
        identities: list[tuple[str, int, int]] = []
        for fits_band in TNG_FITS_BANDS:
            path = self.fits_path(fits_band)
            status = path.stat()
            identities.append(
                (str(path.resolve()), int(status.st_size), int(status.st_mtime_ns))
            )
        return tuple(identities)

    def load_surface_brightness(self) -> TNGSurfaceBrightnessImage:
        """Load and register the four native MJy/sr planes in model-band order."""
        from euclid_polish.tng._image import _load_tng_plane

        planes = tuple(
            _load_tng_plane(self.fits_path(fits_band), model_band)
            for fits_band, model_band in zip(
                TNG_FITS_BANDS, TNG_MODEL_BANDS, strict=True
            )
        )
        image = TNGSurfaceBrightnessImage.stack(planes)
        if not np.isclose(
            image.pixel_scale_pc,
            TNG_NATIVE_PC_PER_PIXEL,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "TNG atlas source must use the native 100 pc/pixel grid, got "
                f"{image.pixel_scale_pc!r}"
            )
        if image.bands != TNG_MODEL_BANDS:
            raise ValueError(f"unexpected TNG band order {image.bands!r}")
        return image


@dataclass(frozen=True, slots=True)
class TNGRotation:
    """The image-plane augmentation applied to one atlas orientation."""

    quarter_turns: int = 0
    angle_deg: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.quarter_turns, bool) or not isinstance(self.quarter_turns, int):
            raise TypeError("quarter_turns must be an integer")
        if self.angle_deg is None:
            object.__setattr__(self, "quarter_turns", self.quarter_turns % 4)
            return
        if self.quarter_turns != 0:
            raise ValueError("an arbitrary rotation cannot also carry quarter turns")
        angle = float(self.angle_deg)
        if not np.isfinite(angle):
            raise ValueError(f"angle_deg must be finite, got {self.angle_deg!r}")
        object.__setattr__(self, "angle_deg", angle % 360.0)

    @property
    def is_arbitrary(self) -> bool:
        return self.angle_deg is not None

    def apply(
        self,
        image: TNGSurfaceBrightnessImage,
    ) -> TNGSurfaceBrightnessImage:
        """Apply exactly the rotation represented by this record."""
        if self.angle_deg is not None:
            # TNG augmentation uses its validated cubic surface-brightness
            # rotation rather than a generic image interpolation policy.
            from euclid_polish.tng._image import _rotate_surface_brightness

            return _rotate_surface_brightness(image, self.angle_deg)
        return image.rotated_quarter(self.quarter_turns)

    def record_fields(self) -> dict[str, Any]:
        return {
            "rot_k": self.quarter_turns,
            "rot_angle": self.angle_deg,
            "arbitrary_rotation": self.is_arbitrary,
        }


@dataclass(frozen=True, slots=True)
class NominalRadiusGeometry:
    """Shrink-only mapping to a sampled nominal Euclid effective radius."""

    target_re_arcsec: float
    scale_factor: float
    radius_rendering: str
    radius_renderer_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_re_arcsec",
            _positive_finite(self.target_re_arcsec, "target_re_arcsec"),
        )
        scale = _positive_finite(self.scale_factor, "scale_factor")
        if scale > 1.0:
            raise ValueError(f"nominal-radius rendering must be shrink-only, got {scale!r}")
        object.__setattr__(self, "scale_factor", scale)
        if not str(self.radius_rendering).strip():
            raise ValueError("radius_rendering must be non-empty")
        if not str(self.radius_renderer_fingerprint).strip():
            raise ValueError("radius_renderer_fingerprint must be non-empty")
        object.__setattr__(self, "radius_rendering", str(self.radius_rendering))
        object.__setattr__(
            self,
            "radius_renderer_fingerprint",
            str(self.radius_renderer_fingerprint),
        )
@dataclass(frozen=True, slots=True)
class PhysicalRedshiftGeometry:
    """Physical 100 pc/pixel placement through an integer output rebin."""

    rebin_factor: int
    continuous_rebin_factor: float
    compactness: float
    apparent_re_arcsec: float
    surface_brightness_cut_mag_arcsec2: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.rebin_factor, bool) or not isinstance(self.rebin_factor, int):
            raise TypeError("rebin_factor must be an integer")
        if self.rebin_factor < 1:
            raise ValueError("rebin_factor must be at least one")
        object.__setattr__(
            self,
            "continuous_rebin_factor",
            _positive_finite(
                self.continuous_rebin_factor,
                "continuous_rebin_factor",
            ),
        )
        object.__setattr__(
            self,
            "compactness",
            _positive_finite(self.compactness, "compactness"),
        )
        object.__setattr__(
            self,
            "apparent_re_arcsec",
            _positive_finite(self.apparent_re_arcsec, "apparent_re_arcsec"),
        )
        object.__setattr__(
            self,
            "surface_brightness_cut_mag_arcsec2",
            _nonnegative_finite(
                self.surface_brightness_cut_mag_arcsec2,
                "surface_brightness_cut_mag_arcsec2",
            ),
        )


type TNGGeometry = NominalRadiusGeometry | PhysicalRedshiftGeometry


@dataclass(frozen=True, slots=True)
class TNGRedshiftTransform:
    """Photometric redshift transform applied after the stamp geometry."""

    redshift: float
    band_factors: tuple[float, float, float, float]
    drift_mode: DriftMode
    drift_epsilon: float
    dimming_factor: float

    def __post_init__(self) -> None:
        redshift = _nonnegative_finite(self.redshift, "redshift")
        factors = _four_positive(self.band_factors, "band_factors")
        if self.drift_mode not in ("sed_interp", "parametric"):
            raise ValueError(f"unsupported drift_mode {self.drift_mode!r}")
        epsilon = float(self.drift_epsilon)
        if not np.isfinite(epsilon):
            raise ValueError("drift_epsilon must be finite")
        dimming = _positive_finite(self.dimming_factor, "dimming_factor")
        object.__setattr__(self, "redshift", redshift)
        object.__setattr__(self, "band_factors", factors)
        object.__setattr__(self, "drift_epsilon", epsilon)
        object.__setattr__(self, "dimming_factor", dimming)

    def record_fields(self) -> dict[str, Any]:
        return {
            "z": self.redshift,
            "redshift_band_factors": self.band_factors,
            "drift_mode": self.drift_mode,
            "drift_eps": self.drift_epsilon,
            "dimming": self.dimming_factor,
        }


@dataclass(frozen=True, slots=True)
class TotalVISNormalization:
    """One shared four-band scalar fixing the stamp's total VIS electrons."""

    target_flux_e: float
    achieved_flux_e: float
    brightness_scale: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_flux_e",
            _positive_finite(self.target_flux_e, "target_flux_e"),
        )
        object.__setattr__(
            self,
            "achieved_flux_e",
            _positive_finite(self.achieved_flux_e, "achieved_flux_e"),
        )
        object.__setattr__(
            self,
            "brightness_scale",
            _positive_finite(self.brightness_scale, "brightness_scale"),
        )

    def record_fields(self) -> dict[str, Any]:
        return {
            "target_vis_flux_e": self.target_flux_e,
            "achieved_vis_flux_e": self.achieved_flux_e,
            "brightness_scale": self.brightness_scale,
            "photometric_scaling": "single_shared_total_vis_anchor",
        }


@dataclass(frozen=True, slots=True)
class VIS2FWHMNormalization:
    """Shared scalar fixing the centred VIS two-FWHM aperture flux."""

    target_flux_e: float
    achieved_flux_e: float
    brightness_scale: float
    psf_fwhm_arcsec: float
    psf_source: str
    psf_fingerprint: str

    APERTURE_PSF_MODEL: ClassVar[str] = "circularized_empirical_vis_psf"
    APERTURE_RESPONSE_METHOD: ClassVar[str] = "compact_adjoint_v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_flux_e",
            _positive_finite(self.target_flux_e, "target_flux_e"),
        )
        object.__setattr__(
            self,
            "achieved_flux_e",
            _positive_finite(self.achieved_flux_e, "achieved_flux_e"),
        )
        object.__setattr__(
            self,
            "brightness_scale",
            _positive_finite(self.brightness_scale, "brightness_scale"),
        )
        object.__setattr__(
            self,
            "psf_fwhm_arcsec",
            _positive_finite(self.psf_fwhm_arcsec, "psf_fwhm_arcsec"),
        )
        object.__setattr__(self, "psf_source", str(self.psf_source))
        object.__setattr__(self, "psf_fingerprint", str(self.psf_fingerprint))

    def record_fields(self) -> dict[str, Any]:
        vis_band = Config.get_band("VIS")
        return {
            "target_vis_2fwhm_flux_e": self.target_flux_e,
            "achieved_vis_2fwhm_flux_e": self.achieved_flux_e,
            "target_vis_2fwhm_mag": float(
                electrons_to_ab_mag(self.target_flux_e, vis_band)
            ),
            "achieved_vis_2fwhm_mag": float(
                electrons_to_ab_mag(self.achieved_flux_e, vis_band)
            ),
            "brightness_scale": self.brightness_scale,
            "photometric_scaling": "vis_2fwhm_after_redshift_and_nominal_scale",
            "aperture_psf_fwhm_arcsec": self.psf_fwhm_arcsec,
            "aperture_psf_source": self.psf_source,
            "aperture_psf_fingerprint": self.psf_fingerprint,
            "aperture_psf_model": self.APERTURE_PSF_MODEL,
            "aperture_response_method": self.APERTURE_RESPONSE_METHOD,
        }


type TNGNormalization = TotalVISNormalization | VIS2FWHMNormalization


@dataclass(frozen=True, slots=True)
class TNGRenderTrace:
    """Complete typed provenance for one rendered TNG stamp."""

    view: TNGView
    rotation: TNGRotation
    geometry: TNGGeometry
    redshift: TNGRedshiftTransform | None = None
    normalization: TNGNormalization | None = None
    render_support_clipped: bool = False
    max_output_side: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.view, TNGView):
            raise TypeError("view must be a TNGView")
        if not isinstance(self.rotation, TNGRotation):
            raise TypeError("rotation must be a TNGRotation")
        if not isinstance(
            self.geometry, (NominalRadiusGeometry, PhysicalRedshiftGeometry)
        ):
            raise TypeError("geometry must be a TNG geometry record")
        if self.redshift is not None and not isinstance(
            self.redshift, TNGRedshiftTransform
        ):
            raise TypeError("redshift must be a TNGRedshiftTransform")
        if isinstance(self.geometry, PhysicalRedshiftGeometry) and self.redshift is None:
            raise ValueError("physical-redshift geometry requires a redshift transform")
        if self.normalization is not None and not isinstance(
            self.normalization, (TotalVISNormalization, VIS2FWHMNormalization)
        ):
            raise TypeError("normalization must be a TNG normalization record")
        if self.max_output_side is not None:
            if isinstance(self.max_output_side, bool) or not isinstance(
                self.max_output_side, int
            ):
                raise TypeError("max_output_side must be an integer")
            if self.max_output_side < 1:
                raise ValueError("max_output_side must be positive")

    def with_redshift(self, transform: TNGRedshiftTransform) -> TNGRenderTrace:
        return replace(self, redshift=transform)

    def with_normalization(
        self,
        normalization: TNGNormalization,
    ) -> TNGRenderTrace:
        return replace(self, normalization=normalization)

    def record_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "subhalo_id": self.view.subhalo_id,
            "orientation": self.view.orientation,
            "native_halflight_px": self.view.native_re_px,
            "radius_manifest_fingerprint": self.view.radius_manifest_fingerprint,
            "render_support_clipped": self.render_support_clipped,
            "max_output_side": self.max_output_side,
        }
        fields.update(self.rotation.record_fields())
        if isinstance(self.geometry, NominalRadiusGeometry):
            fields.update({
                "rebin_factor": 1,
                "rebin_factor_continuous": 1.0 / self.geometry.scale_factor,
                "radius_scale_factor": self.geometry.scale_factor,
                "target_re_arcsec": self.geometry.target_re_arcsec,
                "radius_rendering": self.geometry.radius_rendering,
                "radius_renderer_fingerprint": (
                    self.geometry.radius_renderer_fingerprint
                ),
            })
        else:
            fields.update({
                "rebin_factor": self.geometry.rebin_factor,
                "rebin_factor_continuous": (
                    self.geometry.continuous_rebin_factor
                ),
                "compactness": self.geometry.compactness,
                "apparent_re_arcsec": self.geometry.apparent_re_arcsec,
                "sb_cut_mag_arcsec2": (
                    self.geometry.surface_brightness_cut_mag_arcsec2
                ),
            })
        if self.redshift is not None:
            fields.update(self.redshift.record_fields())
        if self.normalization is not None:
            fields.update(self.normalization.record_fields())
        return fields


@dataclass(frozen=True, slots=True, eq=False)
class NativePhotometry:
    """Cached native VIS radial profile and four-band pixel-value sums."""

    vis_mean_profile_mjy_sr: np.ndarray = field(repr=False, compare=False)
    band_sums_mjy_sr: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        profile = np.array(
            self.vis_mean_profile_mjy_sr,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        if profile.ndim != 1 or profile.size == 0:
            raise ValueError("VIS native photometry profile must be non-empty and 1-D")
        if not np.all(np.isfinite(profile)) or np.any(profile < 0.0):
            raise ValueError("VIS native photometry profile must be finite and non-negative")
        sums = tuple(float(value) for value in self.band_sums_mjy_sr)
        if len(sums) != len(TNG_MODEL_BANDS):
            raise ValueError("native photometry requires four band sums")
        if not all(np.isfinite(value) and value >= 0.0 for value in sums):
            raise ValueError("native photometry band sums must be finite and non-negative")
        profile.setflags(write=False)
        object.__setattr__(self, "vis_mean_profile_mjy_sr", profile)
        object.__setattr__(self, "band_sums_mjy_sr", sums)

    def band_sum(self, band: str) -> float:
        try:
            index = TNG_MODEL_BANDS.index(band)
        except ValueError as exc:
            raise ValueError(f"band {band!r} not in {TNG_MODEL_BANDS!r}") from exc
        return self.band_sums_mjy_sr[index]


@dataclass(frozen=True, slots=True, eq=False, repr=False, init=False)
class RenderedTNG:
    """An immutable clean four-band :class:`Image` plus render provenance.

    The constructor snapshots the supplied ``Image``.  Pixel storage is owned
    by this value and exposed read-only; :attr:`image` returns a detached
    ``Image`` record so changing its metadata cannot invalidate the render
    trace stored here.
    """

    _image: Image = field(repr=False, compare=False)
    trace: TNGRenderTrace

    def __init__(self, image: Image, trace: TNGRenderTrace) -> None:
        self._validate_image(image)
        if not isinstance(trace, TNGRenderTrace):
            raise TypeError("trace must be a TNGRenderTrace")

        # Keep the owning base read-only as well as the public view.  A caller
        # therefore cannot re-enable writes through ``setflags`` on the view.
        pixels_owner = np.array(image.data, order="C", copy=True)
        pixels_owner.setflags(write=False)
        pixels = pixels_owner.view()
        pixels.setflags(write=False)
        snapshot = replace(
            image,
            data=pixels,
            metadata=dict(image.metadata),
        )
        object.__setattr__(self, "_image", snapshot)
        object.__setattr__(self, "trace", trace)
        self.validate()

    @staticmethod
    def _validate_image(image: Image) -> None:
        if not isinstance(image, Image):
            raise TypeError("image must be an Image")
        if image.is_clean is not True or image.role is not Role.CLEAN:
            raise ValueError(
                "rendered TNG stamps must be clean images with Role.CLEAN"
            )
        if image.band_names != TNG_MODEL_BANDS:
            raise ValueError(
                f"rendered TNG bands must be {TNG_MODEL_BANDS!r}, "
                f"got {image.band_names!r}"
            )
        if image.data.dtype != np.float32:
            raise ValueError("rendered TNG pixels must be float32")
        if not np.all(np.isfinite(image.data)):
            raise ValueError("rendered TNG pixels must be finite")
        if np.any(image.data < 0.0):
            raise ValueError("clean rendered TNG pixels must be non-negative")
        _positive_finite(
            image.pixel_scale_arcsec,
            "rendered TNG pixel_scale_arcsec",
        )

    def validate(self) -> None:
        """Recheck the complete rendered-image and trace contract."""
        self._validate_image(self._image)
        if not isinstance(self.trace, TNGRenderTrace):
            raise TypeError("trace must be a TNGRenderTrace")
        if isinstance(self.trace.geometry, NominalRadiusGeometry):
            expected = (
                self.trace.view.native_re_px
                * self.pixel_scale_arcsec
                * self.trace.geometry.scale_factor
            )
            if not np.isclose(
                expected,
                self.trace.geometry.target_re_arcsec,
                rtol=1e-9,
                atol=1e-12,
            ):
                raise ValueError(
                    "nominal radius is inconsistent with donor radius, angular "
                    "pixel scale, and shrink factor"
                )

    @property
    def image(self) -> Image:
        """Return a detached ``Image`` view of the rendered pixels."""
        return replace(
            self._image,
            data=self._image.data,
            metadata=dict(self._image.metadata),
        )

    def __repr__(self) -> str:
        return (
            f"RenderedTNG(shape={self.shape!r}, bands={self.bands!r}, "
            f"pixel_scale_arcsec={self.pixel_scale_arcsec!r}, "
            f"subhalo_id={self.trace.view.subhalo_id!r}, "
            f"orientation={self.trace.view.orientation})"
        )

    @property
    def data(self) -> np.ndarray:
        return self._image.data

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._image.shape

    @property
    def pixel_scale_arcsec(self) -> float:
        return self._image.pixel_scale_arcsec

    @property
    def bands(self) -> tuple[str, ...]:
        return self._image.band_names

    def plane(self, band: str | None = None) -> np.ndarray:
        if band is None:
            if self.shape[-1] != 1:
                raise ValueError("band is required for a multi-channel image")
            return self.data[..., 0]
        try:
            index = self.bands.index(band)
        except ValueError as exc:
            raise ValueError(f"band {band!r} not in {self.bands!r}") from exc
        return self.data[..., index]

    def flux_e(self, band: str) -> float:
        return float(np.sum(self.plane(band), dtype=np.float64))

    @property
    def fluxes_e(self) -> tuple[float, float, float, float]:
        return tuple(self.flux_e(band) for band in self.bands)  # type: ignore[return-value]

    @property
    def flux_e_per_band(self) -> dict[str, float]:
        return dict(zip(self.bands, self.fluxes_e, strict=True))

    def scaled(self, factor: float) -> RenderedTNG:
        """Return a stamp multiplied by one positive, band-shared scalar."""
        scale = _positive_finite(factor, "factor")
        return type(self)(
            image=replace(
                self._image,
                data=self.data * np.float32(scale),
            ),
            trace=self.trace,
        )

    def scaled_by_band(self, factors: Sequence[float]) -> RenderedTNG:
        """Return a stamp multiplied by four positive per-band factors."""
        values = _four_positive(factors, "factors")
        multiplier = np.asarray(values, dtype=np.float32)[None, None, :]
        return type(self)(
            image=replace(self._image, data=self.data * multiplier),
            trace=self.trace,
        )

    def with_trace(self, trace: TNGRenderTrace) -> RenderedTNG:
        return type(self)(image=self._image, trace=trace)

    def transformed_at_redshift(
        self,
        transform: TNGRedshiftTransform,
    ) -> RenderedTNG:
        transformed = self.scaled_by_band(transform.band_factors)
        return transformed.with_trace(self.trace.with_redshift(transform))

    def normalised(self, normalization: TNGNormalization) -> RenderedTNG:
        normalized = self.scaled(normalization.brightness_scale)
        return normalized.with_trace(self.trace.with_normalization(normalization))

    def normalised_to_total_vis(self, target_flux_e: float) -> RenderedTNG:
        """Apply and record one shared scalar matching total VIS electrons."""
        target = _positive_finite(target_flux_e, "target_flux_e")
        current = self.flux_e("VIS")
        if not np.isfinite(current) or current <= 0.0:
            raise ValueError("cannot normalize an empty VIS stamp")
        factor = target / current
        scaled = self.scaled(factor)
        normalization = TotalVISNormalization(
            target_flux_e=target,
            achieved_flux_e=scaled.flux_e("VIS"),
            brightness_scale=factor,
        )
        return scaled.with_trace(self.trace.with_normalization(normalization))

    def as_array(self, *, copy: bool = False) -> np.ndarray:
        return self.data.copy() if copy else self.data

    def record_fields(self) -> dict[str, Any]:
        """Materialize the serializable render record at the catalog boundary."""
        fields = self.trace.record_fields()
        fields.update({
            "shape": self.shape,
            "bands": self.bands,
            "pixel_scale_arcsec": self.pixel_scale_arcsec,
            "flux_e_per_band": self.flux_e_per_band,
        })
        return fields
