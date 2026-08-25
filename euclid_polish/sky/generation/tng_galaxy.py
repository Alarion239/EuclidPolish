"""Render validated TNG SKIRT atlas views onto the Euclid clean-sky grid.

The public boundary is deliberately typed. :class:`TNGView` identifies one
native atlas orientation, :class:`TNGRenderer` owns the expensive caches and
scientific transformations, and every render is returned as
:class:`RenderedTNG`. Raw ndarrays remain private numerical implementation
details; callers cannot accidentally confuse native MJy/sr pixels on a
physical grid with rendered electrons/pixel on an angular grid.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
from scipy.signal import fftconvolve

from euclid_polish.config import Config
from euclid_polish.image import AngularGrid, ImageCube, PhysicalGrid, PixelUnit
from euclid_polish.photometry import (
    ab_mag_to_electrons,
    mjy_per_sr_to_electrons_factor,
)
from euclid_polish.skirt.image import (
    block_mean,
    centered_rotation_crop_slices,
    downsample_surface_brightness,
    radius_int_grid,
    stochastic_round_factor,
)
from euclid_polish.sky.generation.redshift_model import (
    band_drift_factors,
    compactness_factor,
    physical_pc_to_arcsec,
    rebin_factor_for_redshift,
)
from euclid_polish.sky.generation.tng_types import (
    TNG_MODEL_BANDS,
    TNG_NATIVE_PC_PER_PIXEL,
    NativePhotometry,
    NominalRadiusGeometry,
    PhysicalRedshiftGeometry,
    RenderedTNG,
    TNGRedshiftTransform,
    TNGRenderTrace,
    TNGRotation,
    TNGView,
    VIS2FWHMNormalization,
)

N_ORIENTATIONS = 5
ARBITRARY_ROTATION_MIN_REBIN = 4
TNG_ROTATION_CROP_ENCLOSED_FRACTION = 0.99
TNG_ROTATION_CROP_PADDING = 1.05
TNG_MAX_REBIN_FACTOR = 64

TNG_RADIUS_RENDERING = "euclid_sersic_shrink_only_v2"
TNG_RADIUS_RENDERER_FINGERPRINT = hashlib.sha256(
    (
        f"{TNG_RADIUS_RENDERING}|"
        "typed_cube_boundary|shrink_only|eligible_orientation|"
        "one_area_resample|no_output_remeasurement|bounded_support|"
        "adjoint_2fwhm"
    ).encode("ascii")
).hexdigest()

_SOURCE_CACHE_MAX_BYTES = 256 * 1024 * 1024
_APERTURE_CACHE_MAX_BYTES = 16 * 1024 * 1024
_CIRCULAR_PSF_CACHE_MAX_BYTES = 16 * 1024 * 1024


def _positive_finite(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return number


def _four_floats(
    values: Iterable[float],
) -> tuple[float, float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 4:
        raise ValueError(f"expected four TNG band values, got {len(result)}")
    return cast(tuple[float, float, float, float], result)


def tng_fits_path(
    galaxy_dir: str | Path,
    subhalo_id: int | str,
    orientation: int,
    fits_band: str,
) -> str:
    """Resolve one atlas filename, including its zero-padded variant."""
    directory = Path(galaxy_dir)
    band = str(fits_band).strip().upper()
    if band not in ("VIS", "Y", "J", "H"):
        raise ValueError(f"unknown TNG FITS band {fits_band!r}")
    unpadded = directory / (
        f"TNG{subhalo_id}_O{int(orientation)}_Euclid_{band}.fits"
    )
    if unpadded.is_file():
        return str(unpadded)
    try:
        padded_id = f"{int(subhalo_id):06d}"
    except (TypeError, ValueError):
        return str(unpadded)
    padded = directory / (
        f"TNG{padded_id}_O{int(orientation)}_Euclid_{band}.fits"
    )
    return str(padded if padded.is_file() else unpadded)


def list_tng_galaxies(tng_dir: str | Path) -> list[tuple[str, str]]:
    """List completed atlas galaxies that have at least their VIS O1 frame."""
    root = Path(tng_dir)
    if not root.is_dir():
        return []
    galaxies = [
        (str(folder), folder.name)
        for folder in root.iterdir()
        if folder.is_dir()
        and (folder / Config.Tng.DONE_MARKER).is_file()
        and Path(tng_fits_path(folder, folder.name, 1, "VIS")).is_file()
    ]
    try:
        return sorted(galaxies, key=lambda item: int(item[1]))
    except ValueError:
        return sorted(galaxies)


def _array_fingerprint(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _circularize_psf_kernel(kernel: np.ndarray) -> np.ndarray:
    values = np.asarray(kernel, dtype=np.float64)
    if (
        values.ndim != 2
        or min(values.shape) < 3
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("aperture PSF kernel must be a finite 2-D array")
    yy, xx = np.indices(values.shape, dtype=np.float64)
    cy = 0.5 * (values.shape[0] - 1)
    cx = 0.5 * (values.shape[1] - 1)
    radius = np.floor(np.hypot(yy - cy, xx - cx) + 0.5).astype(np.int64)
    radial_sum = np.bincount(radius.ravel(), weights=values.ravel())
    radial_count = np.bincount(radius.ravel())
    profile = np.divide(
        radial_sum,
        radial_count,
        out=np.zeros_like(radial_sum),
        where=radial_count > 0,
    )
    circular = profile[radius]
    total = float(np.sum(circular, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("aperture PSF kernel has non-positive total flux")
    return np.asarray(circular / total, dtype=np.float32)


def _bounded_crop_slices(
    shape: tuple[int, int],
    max_side: int,
) -> tuple[slice, slice]:
    """Centre-crop without moving an integer or half-integer image centre."""

    def bounded_side(current: int, limit: int) -> int:
        side = min(current, max(1, limit))
        if side < current and side % 2 != current % 2:
            side = max(1, side - 1)
        return side

    height, width = shape
    out_height = bounded_side(height, max_side)
    out_width = bounded_side(width, max_side)
    row0 = (height - out_height) // 2
    col0 = (width - out_width) // 2
    return (
        slice(row0, row0 + out_height),
        slice(col0, col0 + out_width),
    )


class TNGRenderer:
    """Render TNG views and own all process-local image and PSF caches."""

    def __init__(
        self,
        *,
        pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
        max_output_side: int | None = None,
        source_cache_max_bytes: int = _SOURCE_CACHE_MAX_BYTES,
        aperture_cache_max_bytes: int = _APERTURE_CACHE_MAX_BYTES,
        circular_psf_cache_max_bytes: int = _CIRCULAR_PSF_CACHE_MAX_BYTES,
    ) -> None:
        self.pixel_scale_arcsec = _positive_finite(
            pixel_scale_arcsec, "pixel_scale_arcsec"
        )
        if max_output_side is not None and int(max_output_side) < 1:
            raise ValueError("max_output_side must be positive")
        self.max_output_side = (
            None if max_output_side is None else int(max_output_side)
        )
        self._source_cache_max_bytes = max(0, int(source_cache_max_bytes))
        self._aperture_cache_max_bytes = max(0, int(aperture_cache_max_bytes))
        self._circular_psf_cache_max_bytes = max(
            0, int(circular_psf_cache_max_bytes)
        )
        self._source_cache: OrderedDict[tuple, ImageCube] = OrderedDict()
        self._source_cache_bytes = 0
        self._native_photometry_cache: dict[tuple, NativePhotometry] = {}
        self._aperture_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._aperture_cache_bytes = 0
        self._circular_psf_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._circular_psf_cache_bytes = 0

    def clear_caches(self) -> None:
        """Release every donor and PSF cache owned by this renderer."""
        self._source_cache.clear()
        self._source_cache_bytes = 0
        self._native_photometry_cache.clear()
        self._aperture_cache.clear()
        self._aperture_cache_bytes = 0
        self._circular_psf_cache.clear()
        self._circular_psf_cache_bytes = 0

    def cache_info(self) -> dict[str, int]:
        """Return compact cache counts and resident-byte totals."""
        return {
            "source_entries": len(self._source_cache),
            "source_bytes": self._source_cache_bytes,
            "native_photometry_entries": len(self._native_photometry_cache),
            "aperture_entries": len(self._aperture_cache),
            "aperture_bytes": self._aperture_cache_bytes,
            "circular_psf_entries": len(self._circular_psf_cache),
            "circular_psf_bytes": self._circular_psf_cache_bytes,
        }

    def choose_view(
        self,
        galaxies: Sequence[tuple[str, str]],
        radius_lookup: Mapping[tuple[str, int], float],
        target_re_arcsec: float,
        rng: np.random.Generator,
        *,
        radius_manifest_fingerprint: str = "",
    ) -> TNGView:
        """Choose a donor orientation that reaches a radius without enlargement."""
        target = _positive_finite(target_re_arcsec, "target_re_arcsec")
        eligible: list[TNGView] = []
        for galaxy_dir, subhalo_id in galaxies:
            for orientation in range(1, N_ORIENTATIONS + 1):
                native_re_px = radius_lookup.get((str(subhalo_id), orientation))
                if native_re_px is None:
                    continue
                try:
                    view = TNGView(
                        galaxy_dir=Path(galaxy_dir),
                        subhalo_id=str(subhalo_id),
                        orientation=orientation,
                        native_re_px=float(native_re_px),
                        radius_manifest_fingerprint=(
                            radius_manifest_fingerprint
                        ),
                    )
                except (TypeError, ValueError):
                    continue
                if view.can_render(target, self.pixel_scale_arcsec):
                    eligible.append(view)
        if not eligible:
            raise ValueError(
                f"no supplied TNG donor orientation can render R_e={target:g} "
                "arcsec without enlargement"
            )
        return eligible[int(rng.integers(0, len(eligible)))]

    def render_for_radius(
        self,
        view: TNGView,
        target_re_arcsec: float,
        *,
        rng: np.random.Generator | None = None,
        target_vis_flux_e: float | None = None,
    ) -> RenderedTNG:
        """Shrink one donor to a nominal Euclid radius on the angular grid."""
        rendered = self._render_nominal_geometry(view, target_re_arcsec, rng)
        if target_vis_flux_e is not None:
            rendered = rendered.normalised_to_total_vis(target_vis_flux_e)
        return rendered

    def render_for_radius_at_redshift(
        self,
        view: TNGView,
        target_re_arcsec: float,
        redshift: float,
        *,
        rng: np.random.Generator | None = None,
        target_vis_flux_e: float | None = None,
    ) -> RenderedTNG:
        """Render nominal radius geometry and then apply redshift photometry."""
        rendered = self._render_nominal_geometry(view, target_re_arcsec, rng)
        sed_fnu = self._native_sed_from_render(rendered)
        factors, metadata = band_drift_factors(sed_fnu, redshift, rng)
        transform = TNGRedshiftTransform(
            redshift=float(redshift),
            band_factors=_four_floats(factors),
            drift_mode=metadata["drift_mode"],
            drift_epsilon=float(metadata["drift_eps"]),
            dimming_factor=float(metadata["dimming"]),
        )
        rendered = rendered.transformed_at_redshift(transform)
        if target_vis_flux_e is not None:
            rendered = rendered.normalised_to_total_vis(target_vis_flux_e)
        return rendered

    def render_for_physical_redshift(
        self,
        view: TNGView,
        redshift: float,
        *,
        rng: np.random.Generator | None = None,
        surface_brightness_cut_mag_arcsec2: float = (
            Config.TNG_SB_TRUNCATE_MAG_ARCSEC2
        ),
    ) -> RenderedTNG:
        """Place the native 100 pc grid at redshift using integer rebinning."""
        redshift_value = float(redshift)
        compactness = compactness_factor(redshift_value)
        geometric_rebin = rebin_factor_for_redshift(
            redshift_value,
            pixel_scale_arcsec=self.pixel_scale_arcsec,
        )
        continuous_rebin = min(
            float(TNG_MAX_REBIN_FACTOR), geometric_rebin * compactness
        )
        rebin = stochastic_round_factor(continuous_rebin, rng)
        rotation = self._draw_physical_rotation(rng, rebin)

        native = self._native_source(view)
        source = native
        if rotation.is_arbitrary:
            rows, columns = centered_rotation_crop_slices(
                native,
                rebin,
                band="VIS",
                enclosed_fraction=TNG_ROTATION_CROP_ENCLOSED_FRACTION,
                padding=TNG_ROTATION_CROP_PADDING,
            )
            source = native.cropped(rows, columns)
            source = rotation.apply(source)
            surface_brightness = block_mean(source, rebin)
        else:
            surface_brightness = block_mean(source, rebin)
            surface_brightness = rotation.apply(surface_brightness)

        electron_cube = self._surface_brightness_to_electrons(surface_brightness)
        sed_fnu = _four_floats(
            float(np.sum(electron_cube.plane(band), dtype=np.float64))
            / mjy_per_sr_to_electrons_factor(
                Config.get_band(band), self.pixel_scale_arcsec
            )
            for band in electron_cube.bands
        )
        factors, metadata = band_drift_factors(
            sed_fnu, redshift_value, rng
        )
        transform = TNGRedshiftTransform(
            redshift=redshift_value,
            band_factors=_four_floats(factors),
            drift_mode=metadata["drift_mode"],
            drift_epsilon=float(metadata["drift_eps"]),
            dimming_factor=float(metadata["dimming"]),
        )
        geometry = PhysicalRedshiftGeometry(
            rebin_factor=rebin,
            continuous_rebin_factor=continuous_rebin,
            compactness=compactness,
            apparent_re_arcsec=(
                physical_pc_to_arcsec(
                    view.native_re_px * TNG_NATIVE_PC_PER_PIXEL,
                    redshift_value,
                )
                / compactness
            ),
            surface_brightness_cut_mag_arcsec2=max(
                0.0, float(surface_brightness_cut_mag_arcsec2)
            ),
        )
        trace = TNGRenderTrace(
            view=view,
            rotation=rotation,
            geometry=geometry,
            redshift=transform,
        )
        combined = np.asarray(factors, dtype=np.float32)
        combined *= np.float32((rebin / geometric_rebin) ** 2)
        rendered = RenderedTNG(
            cube=electron_cube.with_data(
                electron_cube.data * combined[None, None, :]
            ),
            trace=trace,
        )
        rendered = self._truncate_surface_brightness(
            rendered, float(surface_brightness_cut_mag_arcsec2)
        )
        return rendered

    def normalize_vis_2fwhm(
        self,
        rendered: RenderedTNG,
        *,
        target_flux_e: float,
        psf_kernel: np.ndarray,
        psf_fwhm_arcsec: float,
        psf_identity: str = "",
    ) -> RenderedTNG:
        """Apply one shared scalar fixing the centred VIS two-FWHM flux."""
        if not np.isclose(
            rendered.pixel_scale_arcsec,
            self.pixel_scale_arcsec,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("rendered stamp and renderer use different grids")
        target = _positive_finite(target_flux_e, "target_flux_e")
        fwhm = _positive_finite(psf_fwhm_arcsec, "psf_fwhm_arcsec")
        circular_psf, fingerprint = self._cached_circularized_psf(
            psf_kernel, psf_identity=psf_identity
        )
        measured = self._measure_vis_2fwhm_aperture_flux(
            rendered.plane("VIS"),
            circular_psf=circular_psf,
            psf_fwhm_arcsec=fwhm,
            psf_identity=psf_identity or fingerprint,
        )
        if not np.isfinite(measured) or measured <= 0.0:
            raise ValueError("resized TNG has no positive VIS 2FWHM flux")
        brightness_scale = target / measured
        normalization = VIS2FWHMNormalization(
            target_flux_e=target,
            achieved_flux_e=measured * brightness_scale,
            brightness_scale=brightness_scale,
            psf_fwhm_arcsec=fwhm,
            psf_source=str(psf_identity),
            psf_fingerprint=fingerprint,
        )
        return rendered.normalised(normalization)

    def native_photometry(self, view: TNGView) -> NativePhotometry:
        """Return cached native VIS radial profile and four-band sums."""
        key = self._source_key(view)
        cached = self._native_photometry_cache.get(key)
        if cached is not None:
            return cached
        source = self._native_source(view)
        vis = np.asarray(source.plane("VIS"), dtype=np.float64)
        radii = radius_int_grid(vis.shape)
        flux = np.bincount(radii.ravel(), weights=vis.ravel())
        count = np.bincount(radii.ravel()).astype(np.float64)
        profile = flux / np.maximum(count, 1.0)
        photometry = NativePhotometry(
            vis_mean_profile_mjy_sr=profile,
            band_sums_mjy_sr=_four_floats(tuple(
                float(np.sum(source.plane(band), dtype=np.float64))
                for band in source.bands
            )),
        )
        self._native_photometry_cache[key] = photometry
        return photometry

    def predict_visible_radius_arcsec(
        self,
        view: TNGView,
        redshift: float,
        *,
        surface_brightness_cut_mag_arcsec2: float = (
            Config.TNG_SB_TRUNCATE_MAG_ARCSEC2
        ),
    ) -> float:
        """Predict the detectable VIS radius without rendering a stamp."""
        photometry = self.native_photometry(view)
        factors, _ = band_drift_factors(
            photometry.band_sums_mjy_sr, redshift, None
        )
        compactness = compactness_factor(redshift)
        vis_band = Config.get_band("VIS")
        conversion = mjy_per_sr_to_electrons_factor(
            vis_band, self.pixel_scale_arcsec
        )
        surface_brightness = (
            photometry.vis_mean_profile_mjy_sr
            * conversion
            * factors[0]
            * compactness**2
        )
        threshold = (
            ab_mag_to_electrons(
                surface_brightness_cut_mag_arcsec2, vis_band
            )
            * self.pixel_scale_arcsec**2
        )
        above = np.nonzero(surface_brightness >= threshold)[0]
        if above.size == 0:
            return 0.0
        return (
            physical_pc_to_arcsec(
                float(above.max()) * TNG_NATIVE_PC_PER_PIXEL,
                redshift,
            )
            / compactness
        )

    def predict_vis_flux_e(self, view: TNGView, redshift: float) -> float:
        """Predict total VIS electrons at redshift, ignoring truncation."""
        photometry = self.native_photometry(view)
        factors, _ = band_drift_factors(
            photometry.band_sums_mjy_sr, redshift, None
        )
        geometric_rebin = rebin_factor_for_redshift(
            redshift, pixel_scale_arcsec=self.pixel_scale_arcsec
        )
        conversion = mjy_per_sr_to_electrons_factor(
            Config.get_band("VIS"), self.pixel_scale_arcsec
        )
        return float(
            photometry.band_sum("VIS")
            * conversion
            * factors[0]
            / geometric_rebin**2
        )

    def _render_nominal_geometry(
        self,
        view: TNGView,
        target_re_arcsec: float,
        rng: np.random.Generator | None,
    ) -> RenderedTNG:
        target = _positive_finite(target_re_arcsec, "target_re_arcsec")
        if not view.can_render(target, self.pixel_scale_arcsec):
            scale = target / view.native_re_arcsec(self.pixel_scale_arcsec)
            raise ValueError(
                f"TNG donors cannot be enlarged (scale={scale!r}); choose an "
                "orientation with a larger native half-light radius"
            )
        scale = target / view.native_re_arcsec(self.pixel_scale_arcsec)
        rotation = self._draw_nominal_rotation(rng)
        native = self._native_source(view)
        rows, columns = centered_rotation_crop_slices(
            native,
            1,
            band="VIS",
            enclosed_fraction=0.999,
            padding=TNG_ROTATION_CROP_PADDING,
        )
        source = native.cropped(rows, columns)
        source = rotation.apply(source)
        surface_brightness = downsample_surface_brightness(source, scale)
        support_clipped = False
        if self.max_output_side is not None and (
            surface_brightness.shape[0] > self.max_output_side
            or surface_brightness.shape[1] > self.max_output_side
        ):
            crop_rows, crop_columns = _bounded_crop_slices(
                surface_brightness.spatial_shape, self.max_output_side
            )
            surface_brightness = surface_brightness.cropped(
                crop_rows, crop_columns
            )
            support_clipped = True
        geometry = NominalRadiusGeometry(
            target_re_arcsec=target,
            scale_factor=scale,
            radius_rendering=TNG_RADIUS_RENDERING,
            radius_renderer_fingerprint=TNG_RADIUS_RENDERER_FINGERPRINT,
        )
        trace = TNGRenderTrace(
            view=view,
            rotation=rotation,
            geometry=geometry,
            render_support_clipped=support_clipped,
            max_output_side=self.max_output_side,
        )
        return RenderedTNG(
            cube=self._surface_brightness_to_electrons(surface_brightness),
            trace=trace,
        )

    def _native_source(self, view: TNGView) -> ImageCube:
        key = self._source_key(view)
        cached = self._source_cache.pop(key, None)
        if cached is not None:
            self._source_cache[key] = cached
            return cached
        source = view.load_surface_brightness()
        if source.data.nbytes <= self._source_cache_max_bytes:
            logical_key = key[:3]
            for old_key in list(self._source_cache):
                if old_key[:3] == logical_key and old_key != key:
                    evicted = self._source_cache.pop(old_key)
                    self._source_cache_bytes -= int(evicted.data.nbytes)
                    self._native_photometry_cache.pop(old_key, None)
            while (
                self._source_cache
                and self._source_cache_bytes + source.data.nbytes
                > self._source_cache_max_bytes
            ):
                evicted_key, evicted = self._source_cache.popitem(last=False)
                self._source_cache_bytes -= int(evicted.data.nbytes)
                self._native_photometry_cache.pop(evicted_key, None)
            self._source_cache[key] = source
            self._source_cache_bytes += int(source.data.nbytes)
        return source

    @staticmethod
    def _source_key(view: TNGView) -> tuple:
        return (
            str(view.galaxy_dir.resolve()),
            view.subhalo_id,
            view.orientation,
            view.radius_manifest_fingerprint,
            view.file_identity(),
        )

    def _surface_brightness_to_electrons(
        self, source: ImageCube
    ) -> ImageCube:
        if source.unit is not PixelUnit.MJY_PER_SR:
            raise ValueError("TNG source must be in MJy/sr")
        if not isinstance(source.grid, PhysicalGrid):
            raise ValueError("TNG source must use a physical parsec grid")
        if source.bands != TNG_MODEL_BANDS:
            raise ValueError(
                f"TNG source bands must be {TNG_MODEL_BANDS!r}, "
                f"got {source.bands!r}"
            )
        factors = np.asarray(
            [
                mjy_per_sr_to_electrons_factor(
                    Config.get_band(band), self.pixel_scale_arcsec
                )
                for band in source.bands
            ],
            dtype=np.float32,
        )
        return ImageCube(
            data=source.data * factors[None, None, :],
            bands=source.bands,
            unit=PixelUnit.ELECTRONS_PER_PIXEL,
            grid=AngularGrid(self.pixel_scale_arcsec),
        )

    def _native_sed_from_render(
        self, rendered: RenderedTNG
    ) -> tuple[float, float, float, float]:
        values = _four_floats(tuple(
            rendered.flux_e(band)
            / mjy_per_sr_to_electrons_factor(
                Config.get_band(band), self.pixel_scale_arcsec
            )
            for band in rendered.bands
        ))
        return values

    @staticmethod
    def _draw_nominal_rotation(
        rng: np.random.Generator | None,
    ) -> TNGRotation:
        if rng is None:
            return TNGRotation()
        return TNGRotation(angle_deg=float(rng.uniform(0.0, 360.0)))

    @staticmethod
    def _draw_physical_rotation(
        rng: np.random.Generator | None,
        rebin: int,
    ) -> TNGRotation:
        if rng is None:
            return TNGRotation()
        if rebin >= ARBITRARY_ROTATION_MIN_REBIN:
            return TNGRotation(angle_deg=float(rng.uniform(0.0, 360.0)))
        return TNGRotation(quarter_turns=int(rng.integers(0, 4)))

    def _truncate_surface_brightness(
        self,
        rendered: RenderedTNG,
        cutoff_mag_arcsec2: float,
    ) -> RenderedTNG:
        if cutoff_mag_arcsec2 <= 0.0:
            return rendered
        data = rendered.as_array(copy=True)
        keep = np.zeros(data.shape[:2], dtype=bool)
        pixel_area = self.pixel_scale_arcsec**2
        for index, band_name in enumerate(rendered.bands):
            threshold = (
                ab_mag_to_electrons(
                    cutoff_mag_arcsec2, Config.get_band(band_name)
                )
                * pixel_area
            )
            channel = data[..., index]
            channel[channel < threshold] = 0.0
            keep |= channel > 0.0
        if not keep.any():
            return RenderedTNG(rendered.cube.with_data(data), rendered.trace)
        total = data.sum(axis=2, dtype=np.float64)
        radii = radius_int_grid(total.shape)
        profile = np.bincount(radii.ravel(), weights=total.ravel())
        cumulative = np.cumsum(profile)
        radius = int(np.searchsorted(cumulative, 0.995 * cumulative[-1])) + 4
        height, width = total.shape
        centre_y = int(round((height - 1) / 2.0))
        centre_x = int(round((width - 1) / 2.0))
        row0, row1 = max(0, centre_y - radius), min(
            height, centre_y + radius + 1
        )
        col0, col1 = max(0, centre_x - radius), min(
            width, centre_x + radius + 1
        )
        cube = rendered.cube.with_data(data[row0:row1, col0:col1, :])
        return RenderedTNG(cube=cube, trace=rendered.trace)

    def _cached_circularized_psf(
        self,
        kernel: np.ndarray,
        *,
        psf_identity: str,
    ) -> tuple[np.ndarray, str]:
        values = np.ascontiguousarray(kernel, dtype=np.float32)
        input_fingerprint = _array_fingerprint(values)
        key = (str(psf_identity), input_fingerprint)
        cached = self._circular_psf_cache.pop(key, None)
        if cached is not None:
            self._circular_psf_cache[key] = cached
            return cached, _array_fingerprint(cached)
        circular = _circularize_psf_kernel(values)
        circular.setflags(write=False)
        if circular.nbytes <= self._circular_psf_cache_max_bytes:
            while (
                self._circular_psf_cache
                and self._circular_psf_cache_bytes + circular.nbytes
                > self._circular_psf_cache_max_bytes
            ):
                _, evicted = self._circular_psf_cache.popitem(last=False)
                self._circular_psf_cache_bytes -= int(evicted.nbytes)
            self._circular_psf_cache[key] = circular
            self._circular_psf_cache_bytes += int(circular.nbytes)
        return circular, _array_fingerprint(circular)

    def _aperture_response(
        self,
        aperture: np.ndarray,
        circular_psf: np.ndarray,
        *,
        psf_identity: str,
        psf_fwhm_arcsec: float,
        centre_parity: tuple[int, int],
    ) -> np.ndarray:
        aperture_values = np.ascontiguousarray(aperture, dtype=np.uint8)
        psf_values = np.ascontiguousarray(circular_psf, dtype=np.float32)
        key = (
            str(psf_identity),
            _array_fingerprint(psf_values),
            float(psf_fwhm_arcsec),
            self.pixel_scale_arcsec,
            tuple(int(value) for value in centre_parity),
            aperture_values.shape,
            aperture_values.tobytes(),
        )
        cached = self._aperture_cache.pop(key, None)
        if cached is not None:
            self._aperture_cache[key] = cached
            return cached
        response = fftconvolve(
            aperture_values.astype(np.float64),
            np.asarray(psf_values[::-1, ::-1], dtype=np.float64),
            mode="full",
        )
        response = np.asarray(response, dtype=np.float64)
        response.setflags(write=False)
        if response.nbytes <= self._aperture_cache_max_bytes:
            while (
                self._aperture_cache
                and self._aperture_cache_bytes + response.nbytes
                > self._aperture_cache_max_bytes
            ):
                _, evicted = self._aperture_cache.popitem(last=False)
                self._aperture_cache_bytes -= int(evicted.nbytes)
            self._aperture_cache[key] = response
            self._aperture_cache_bytes += int(response.nbytes)
        return response

    def _measure_vis_2fwhm_aperture_flux(
        self,
        vis: np.ndarray,
        *,
        circular_psf: np.ndarray,
        psf_fwhm_arcsec: float,
        psf_identity: str,
    ) -> float:
        image = np.asarray(vis, dtype=np.float64)
        kernel = np.asarray(circular_psf, dtype=np.float32)
        if image.ndim != 2 or min(image.shape) < 1:
            raise ValueError("VIS aperture measurement requires a 2-D image")
        if kernel.ndim != 2 or min(kernel.shape) < 1:
            raise ValueError("VIS aperture measurement requires a 2-D PSF")
        radius = psf_fwhm_arcsec / self.pixel_scale_arcsec
        height, width = image.shape
        centre_y = 0.5 * (height - 1)
        centre_x = 0.5 * (width - 1)
        row0 = max(0, int(math.ceil(centre_y - radius)))
        row1 = min(height - 1, int(math.floor(centre_y + radius)))
        col0 = max(0, int(math.ceil(centre_x - radius)))
        col1 = min(width - 1, int(math.floor(centre_x + radius)))
        yy, xx = np.indices(
            (row1 - row0 + 1, col1 - col0 + 1), dtype=np.float64
        )
        aperture = np.hypot(
            yy + row0 - centre_y, xx + col0 - centre_x
        ) <= radius
        if not np.any(aperture):
            return 0.0
        response = self._aperture_response(
            aperture,
            kernel,
            psf_identity=psf_identity,
            psf_fwhm_arcsec=psf_fwhm_arcsec,
            centre_parity=(height % 2, width % 2),
        )
        kernel_height, kernel_width = kernel.shape
        same_row0 = (kernel_height - 1) // 2
        same_col0 = (kernel_width - 1) // 2
        input_row0 = row0 + same_row0 - (kernel_height - 1)
        input_col0 = col0 + same_col0 - (kernel_width - 1)
        response_row0 = max(0, -input_row0)
        response_col0 = max(0, -input_col0)
        response_row1 = min(response.shape[0], height - input_row0)
        response_col1 = min(response.shape[1], width - input_col0)
        if response_row1 <= response_row0 or response_col1 <= response_col0:
            return 0.0
        image_row0 = input_row0 + response_row0
        image_col0 = input_col0 + response_col0
        image_row1 = input_row0 + response_row1
        image_col1 = input_col0 + response_col1
        return float(
            np.sum(
                image[image_row0:image_row1, image_col0:image_col1]
                * response[
                    response_row0:response_row1,
                    response_col0:response_col1,
                ],
                dtype=np.float64,
            )
        )
