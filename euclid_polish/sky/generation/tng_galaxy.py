"""Turn an IllustrisTNG TNG50-1 SKIRT-atlas mock into an injectable,
electron-calibrated galaxy stamp on the simulation's HR clean-sky grid.

The atlas (downloaded by ``scripts/fasrc_download_tng_skirt_atlas.py``) renders
each subhalo as dusty Euclid frames ``TNG<id>_O<k>_Euclid_<band>.fits`` for
band ∈ {VIS, Y, J, H} and orientation k ∈ {1..5}. Each frame is a 1600×1600
**surface-brightness** image (``BUNIT = MJy/sr``) on a *physical* grid
(``CDELT = 100 pc/pixel``) — there is no instrument PSF, noise, or angular WCS;
these are idealized, dust-attenuated intrinsic images.

For Euclid-conditioned field galaxies, the validated native VIS half-light
radius comes from the offline radius manifest.  Generation applies the direct
similarity scale ``R_e,draw / (R_e,native * pixel_scale)``: rotate once, resize
the registered four-band cube once, and never remeasure the pixelized output.
The sampled Euclid Sersic radius is therefore explicitly a nominal
continuous-space value.  The native cube is cached per process and rendering
is bounded to pixels that can affect the field or central normalization
aperture.

After redshift-dependent colour drift, a shared four-band scalar sets the
drawn VIS 2FWHM flux.  Its aperture flux is evaluated with the compact adjoint
aperture response rather than a full-stamp convolution.  The result remains a
clean, pre-PSF source; the existing forward model independently supplies the
dirty-image Euclid PSF, noise, and LR rebin.  The legacy physical-redshift and
integer-rebin helpers remain available for strong-lens rendering.

The returned stamp is ``(H, W, 4)`` in ``Config.LR_INPUT_BAND_NAMES`` order
(VIS, Y_E, J_E, H_E), float32 electrons.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import OrderedDict

import numpy as np
from scipy.signal import fftconvolve

from euclid_polish.config import BandConfig, Config
from euclid_polish.photometry import (
    ab_mag_to_electrons,
    electrons_to_ab_mag,
    mjy_per_sr_to_electrons,
    mjy_per_sr_to_electrons_factor,
)
from euclid_polish.skirt.image import (
    block_mean,
    centered_rotation_crop_slices,
    load_skirt_frame,
    measure_halflight_radius_px,
    radius_int_grid,
    resample_surface_brightness,
    rotate_arbitrary,
    rotate_quarter,
    stochastic_round_factor,
)
from euclid_polish.skirt.image import (
    composite_stamp as _composite_stamp,
)
from euclid_polish.sky.generation.redshift_model import (
    TNG_NATIVE_PC_PER_PIXEL,
    band_drift_factors,
    compactness_factor,
    physical_pc_to_arcsec,
    rebin_factor_for_redshift,
)

# FITS band token (in the filename) → simulation BandConfig. The atlas frames
# are named with the short Euclid band labels; the model's bands carry the
# ``_E`` suffix on the NISP filters.
TNG_FITS_BANDS: tuple[str, ...] = ("VIS", "Y", "J", "H")
_FITS_BAND_TO_CONFIG: dict[str, str] = {
    "VIS": "VIS", "Y": "Y_E", "J": "J_E", "H": "H_E",
}

# Backward-compatible import for callers that historically obtained this
# generic operation from ``tng_galaxy``.
composite_stamp = _composite_stamp


def tng_fits_path(galaxy_dir: str, subhalo_id: int | str,
                  orientation: int, fits_band: str) -> str:
    """Path of one atlas frame, e.g. ``…/167396/TNG167396_O4_Euclid_VIS.fits``."""
    name = f"TNG{subhalo_id}_O{orientation}_Euclid_{fits_band}.fits"
    path = os.path.join(galaxy_dir, name)
    if os.path.isfile(path):
        return path
    try:
        padded_name = (
            f"TNG{int(subhalo_id):06d}_O{orientation}_Euclid_{fits_band}.fits"
        )
    except (TypeError, ValueError):
        return path
    padded_path = os.path.join(galaxy_dir, padded_name)
    return padded_path if os.path.isfile(padded_path) else path


def load_tng_frame(path: str) -> np.ndarray:
    """Read a TNG-SKIRT FITS as a native-endian float32 MJy/sr array.

    This compatibility wrapper keeps the established TNG API while the generic
    FITS mechanics live in :mod:`euclid_polish.skirt.image`.
    """
    return load_skirt_frame(path)


#: TNG-policy threshold at/above which an arbitrary-angle rotation may precede
#: the block mean. The generic rotation primitive itself lives in
#: :mod:`euclid_polish.skirt.image`; this threshold is specific to our validation
#: on the TNG atlas and Euclid HR grid.
ARBITRARY_ROTATION_MIN_REBIN = 4
TNG_ROTATION_CROP_ENCLOSED_FRACTION = 0.99
TNG_ROTATION_CROP_PADDING = 1.05
TNG_MAX_REBIN_FACTOR = 64

# The native-radius manifest remains an offline measurement artifact.  This
# separate renderer identity records how that measurement is transported to a
# sampled Euclid Sersic R_e at generation time.
TNG_RADIUS_RENDERER_VERSION = 1
TNG_RADIUS_RENDERING = "euclid_sersic_nominal_similarity_v1"
TNG_RADIUS_RENDERER_FINGERPRINT = hashlib.sha256(
    (
        f"{TNG_RADIUS_RENDERING}|version={TNG_RADIUS_RENDERER_VERSION}|"
        "one_resize|no_output_remeasurement|bounded_support|adjoint_2fwhm"
    ).encode("ascii")
).hexdigest()

# A worker repeatedly draws from a small atlas.  Cache the registered,
# unrotated four-band source rather than reopening four 1600-square FITS files
# for every galaxy.  The byte cap is per process, matching generation's
# process-level parallelism.
_TNG_SOURCE_CACHE_MAX_BYTES = 256 * 1024 * 1024
_TNG_SOURCE_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_TNG_SOURCE_CACHE_BYTES = 0

# Aperture-response kernels are tiny, but keep their cache bounded as well so
# a long job drawing many empirical PSFs cannot grow without limit.
_APERTURE_RESPONSE_CACHE_MAX_BYTES = 16 * 1024 * 1024
_APERTURE_RESPONSE_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_APERTURE_RESPONSE_CACHE_BYTES = 0
_CIRCULAR_PSF_CACHE_MAX_BYTES = 16 * 1024 * 1024
_CIRCULAR_PSF_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_CIRCULAR_PSF_CACHE_BYTES = 0


def surface_brightness_to_electrons(arr_mjy_sr: np.ndarray, band: BandConfig,
                                    pixel_scale_arcsec: float) -> np.ndarray:
    """MJy/sr → electrons-per-pixel over ``band``'s stack at the HR pixel
    scale. Thin wrapper over the photometry primitive, kept here so the
    three-step recipe reads top-to-bottom."""
    return mjy_per_sr_to_electrons(arr_mjy_sr, band, pixel_scale_arcsec)


def prepare_tng_galaxy(
    galaxy_dir: str,
    subhalo_id: int | str,
    orientation: int,
    *,
    rebin_factor: int = 2,
    rot_k: int = 0,
    rot_angle: float | None = None,
    min_rebin_for_angle: int = ARBITRARY_ROTATION_MIN_REBIN,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
    fits_bands: tuple[str, ...] = TNG_FITS_BANDS,
) -> tuple[np.ndarray, dict]:
    """Build a 4-band, electron-calibrated TNG stamp ready to inject.

    Loads the four Euclid frames for ``(subhalo_id, orientation)`` and converts
    MJy/sr → electrons at ``pixel_scale_arcsec`` (the HR clean-sky grid, 0.05″ by
    default). Rotation depends on the downsample:

    * ``rot_angle`` given **and** ``rebin_factor >= min_rebin_for_angle`` →
      arbitrary-angle cubic-spline rotation at native resolution *before* the
      block-mean (the ≥K× averaging washes out the interpolation blur — see
      :func:`rotate_arbitrary`). This is the orientation-augmentation path.
    * otherwise → block-mean first, then an exact ``rot_k`` quarter-turn (the
      lossless fallback for low-downsample stamps / when no angle is requested).

    Returns
    -------
    stamp : ndarray
        ``(H, W, 4)`` float32 electrons, channels in
        ``Config.LR_INPUT_BAND_NAMES`` order (VIS, Y_E, J_E, H_E).
    meta : dict
        Provenance + per-band integrated electron counts.
    """
    if rebin_factor < 1:
        raise ValueError(f"rebin_factor must be ≥ 1, got {rebin_factor}")
    # Arbitrary-angle rotation only when requested AND enough downsample follows
    # to wash out the spline blur; else the exact quarter-turn.
    use_angle = (rot_angle is not None
                 and rebin_factor >= int(min_rebin_for_angle))
    # Assemble in canonical model-band order so channel c is LR band c.
    config_to_fits = {v: k for k, v in _FITS_BAND_TO_CONFIG.items()}
    channels: list[np.ndarray] = []
    flux_e: dict[str, float] = {}
    rot_crop = None     # centred crop slices, computed once (from VIS) for all bands
    for cfg_name in Config.LR_INPUT_BAND_NAMES:
        fband = config_to_fits[cfg_name]
        if fband not in fits_bands:
            raise ValueError(f"band {fband} not in requested fits_bands={fits_bands}")
        band = Config.get_band(cfg_name)
        path = tng_fits_path(galaxy_dir, subhalo_id, orientation, fband)
        sb = load_tng_frame(path)                       # MJy/sr, 1600²
        if use_angle:
            if rot_crop is None:                        # size from the first (VIS) frame
                rot_crop = centered_rotation_crop_slices(
                    sb,
                    rebin_factor,
                    enclosed_fraction=TNG_ROTATION_CROP_ENCLOSED_FRACTION,
                    padding=TNG_ROTATION_CROP_PADDING,
                )
            # Crop to the galaxy core, then spline-rotate (cheap) at native res.
            sb = rotate_arbitrary(sb[rot_crop], rot_angle)
        sb = block_mean(sb, rebin_factor)               # still MJy/sr
        if not use_angle:
            sb = rotate_quarter(sb, rot_k)              # exact 90° fallback
        e = surface_brightness_to_electrons(sb, band, pixel_scale_arcsec)
        channels.append(e)
        flux_e[cfg_name] = float(e.sum())
    stamp = np.stack(channels, axis=-1).astype(np.float32)
    meta = {
        "subhalo_id": str(subhalo_id),
        "orientation": int(orientation),
        "rebin_factor": int(rebin_factor),
        "rot_k": int(rot_k) % 4,
        "rot_angle": float(rot_angle) % 360.0 if use_angle else None,
        "arbitrary_rotation": bool(use_angle),
        "pixel_scale_arcsec": float(pixel_scale_arcsec),
        "shape": tuple(stamp.shape),
        "flux_e_per_band": flux_e,
        "bands": tuple(Config.LR_INPUT_BAND_NAMES),
    }
    return stamp, meta


def _registered_source_key(
    galaxy_dir: str,
    subhalo_id: int | str,
    orientation: int,
    *,
    fits_bands: tuple[str, ...],
    radius_manifest_fingerprint: str,
) -> tuple:
    """Return a cache key that changes when any contributing FITS changes."""
    config_to_fits = {v: k for k, v in _FITS_BAND_TO_CONFIG.items()}
    identities: list[tuple[str, int, int]] = []
    for cfg_name in Config.LR_INPUT_BAND_NAMES:
        fband = config_to_fits[cfg_name]
        if fband not in fits_bands:
            raise ValueError(
                f"band {fband} not in requested fits_bands={fits_bands}"
            )
        path = os.path.realpath(tng_fits_path(
            galaxy_dir, subhalo_id, orientation, fband,
        ))
        status = os.stat(path)
        identities.append((path, int(status.st_size), int(status.st_mtime_ns)))
    return (
        os.path.realpath(galaxy_dir), str(subhalo_id), int(orientation),
        tuple(fits_bands), str(radius_manifest_fingerprint), tuple(identities),
    )


def _clear_tng_source_cache() -> None:
    """Clear the process-local registered-source cache (mainly for tests)."""
    global _TNG_SOURCE_CACHE_BYTES
    _TNG_SOURCE_CACHE.clear()
    _TNG_SOURCE_CACHE_BYTES = 0


def _cache_registered_source(key: tuple, source: np.ndarray) -> np.ndarray:
    global _TNG_SOURCE_CACHE_BYTES
    values = np.ascontiguousarray(source, dtype=np.float32)
    values.setflags(write=False)
    if values.nbytes > _TNG_SOURCE_CACHE_MAX_BYTES:
        return values

    # A changed file identity or manifest fingerprint should replace, rather
    # than coexist with, the stale copy of the same donor/orientation.
    logical_key = key[:3]
    for old_key in list(_TNG_SOURCE_CACHE):
        if old_key[:3] == logical_key and old_key != key:
            evicted = _TNG_SOURCE_CACHE.pop(old_key)
            _TNG_SOURCE_CACHE_BYTES -= int(evicted.nbytes)
    cached = _TNG_SOURCE_CACHE.pop(key, None)
    if cached is not None:
        _TNG_SOURCE_CACHE_BYTES -= int(cached.nbytes)
    while (
        _TNG_SOURCE_CACHE
        and _TNG_SOURCE_CACHE_BYTES + values.nbytes
        > _TNG_SOURCE_CACHE_MAX_BYTES
    ):
        _, evicted = _TNG_SOURCE_CACHE.popitem(last=False)
        _TNG_SOURCE_CACHE_BYTES -= int(evicted.nbytes)
    _TNG_SOURCE_CACHE[key] = values
    _TNG_SOURCE_CACHE_BYTES += int(values.nbytes)
    return values


def _prepare_tng_continuous_source(
    galaxy_dir: str,
    subhalo_id: int | str,
    orientation: int,
    *,
    rot_angle: float | None = None,
    fits_bands: tuple[str, ...] = TNG_FITS_BANDS,
    radius_manifest_fingerprint: str = "",
) -> tuple[np.ndarray, bool]:
    """Return one cached, cropped, unrotated registered four-band source."""
    key = _registered_source_key(
        galaxy_dir, subhalo_id, orientation, fits_bands=fits_bands,
        radius_manifest_fingerprint=radius_manifest_fingerprint,
    )
    cached = _TNG_SOURCE_CACHE.pop(key, None)
    if cached is not None:
        _TNG_SOURCE_CACHE[key] = cached
        return cached, rot_angle is not None

    config_to_fits = {v: k for k, v in _FITS_BAND_TO_CONFIG.items()}
    native_channels: list[np.ndarray] = []
    for cfg_name in Config.LR_INPUT_BAND_NAMES:
        fband = config_to_fits[cfg_name]
        if fband not in fits_bands:
            raise ValueError(f"band {fband} not in requested fits_bands={fits_bands}")
        native_channels.append(load_tng_frame(tng_fits_path(
            galaxy_dir, subhalo_id, orientation, fband)))
    native_cube = np.stack(native_channels, axis=-1).astype(np.float32)
    crop = centered_rotation_crop_slices(
        native_cube[..., 0], 1, enclosed_fraction=0.999,
        padding=TNG_ROTATION_CROP_PADDING,
    )
    source = _cache_registered_source(key, native_cube[crop])
    return source, rot_angle is not None


def _center_crop_for_limit(
    values: np.ndarray,
    max_height: int,
    max_width: int,
) -> np.ndarray:
    """Centre-crop without moving an integer or half-integer image centre."""
    height, width = values.shape[:2]

    def _bounded_side(current: int, limit: int) -> int:
        side = min(current, max(1, int(limit)))
        if side < current and side % 2 != current % 2:
            side = max(1, side - 1)
        return side

    out_height = _bounded_side(height, max_height)
    out_width = _bounded_side(width, max_width)
    y0 = (height - out_height) // 2
    x0 = (width - out_width) // 2
    return values[y0:y0 + out_height, x0:x0 + out_width]


def _source_for_bounded_render(
    source: np.ndarray,
    *,
    scale: float,
    use_angle: bool,
    max_output_side: int | None,
) -> tuple[np.ndarray, bool]:
    """Pre-crop only when an enlarged output would exceed field support."""
    if max_output_side is None:
        return source, False
    limit = max(1, int(max_output_side))
    output_height = max(1, int(round(source.shape[0] * scale)))
    output_width = max(1, int(round(source.shape[1] * scale)))
    if output_height <= limit and output_width <= limit:
        return source, False

    # Shrinkage never allocates more than the native source.  Preserve its
    # validated rotate-then-area-resample path exactly and crop only afterward.
    if scale < 1.0:
        return source, True

    # For enlargement, the inverse of a rotated square is bounded by sqrt(2)
    # times its side.  Cubic interpolation needs two pixels of support; retain
    # four to make the bound conservative under centre/parity rounding.
    inverse_side = math.ceil(limit / scale)
    if use_angle:
        inverse_side = math.ceil(math.sqrt(2.0) * inverse_side)
    inverse_side += 8
    return _center_crop_for_limit(
        source, inverse_side, inverse_side,
    ), True


def _render_tng_continuous_source(
    source: np.ndarray,
    *,
    scale: float,
    rot_k: int,
    use_angle: bool,
    subhalo_id: int | str,
    orientation: int,
    rot_angle: float | None,
    pixel_scale_arcsec: float,
    max_output_side: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Apply one shared spatial scale to an already prepared TNG source."""
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale!r}")
    render_source, support_clipped = _source_for_bounded_render(
        source, scale=scale, use_angle=use_angle,
        max_output_side=max_output_side,
    )
    if use_angle:
        render_source = rotate_arbitrary(render_source, float(rot_angle))
    else:
        render_source = rotate_quarter(render_source, rot_k)
    cube = resample_surface_brightness(render_source, scale)
    if max_output_side is not None:
        bounded = _center_crop_for_limit(
            cube, int(max_output_side), int(max_output_side),
        )
        support_clipped = support_clipped or bounded.shape != cube.shape
        cube = bounded

    channels: list[np.ndarray] = []
    flux_e: dict[str, float] = {}
    for index, cfg_name in enumerate(Config.LR_INPUT_BAND_NAMES):
        electrons = surface_brightness_to_electrons(
            cube[..., index], Config.get_band(cfg_name), pixel_scale_arcsec,
        )
        channels.append(electrons)
        flux_e[cfg_name] = float(electrons.sum())
    stamp = np.stack(channels, axis=-1).astype(np.float32)
    return stamp, {
        "subhalo_id": str(subhalo_id),
        "orientation": int(orientation),
        "rebin_factor": 1,
        "rebin_factor_continuous": float(1.0 / scale),
        "scale_factor": float(scale),
        "rot_k": int(rot_k) % 4,
        "rot_angle": float(rot_angle) % 360.0 if use_angle else None,
        "arbitrary_rotation": bool(use_angle),
        "render_support_clipped": bool(support_clipped),
        "max_output_side": (
            int(max_output_side) if max_output_side is not None else None
        ),
        "pixel_scale_arcsec": float(pixel_scale_arcsec),
        "shape": tuple(stamp.shape),
        "flux_e_per_band": flux_e,
        "bands": tuple(Config.LR_INPUT_BAND_NAMES),
    }


def prepare_tng_galaxy_continuous(
    galaxy_dir: str,
    subhalo_id: int | str,
    orientation: int,
    *,
    scale: float,
    rot_k: int = 0,
    rot_angle: float | None = None,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
    fits_bands: tuple[str, ...] = TNG_FITS_BANDS,
    radius_manifest_fingerprint: str = "",
    max_output_side: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Render the registered four-band cube with one linear scale.

    This is the size-matching path.  The source is cropped from the native
    atlas footprint using the VIS curve of growth, then the complete cube is
    rotated once and resampled with one shared linear scale.  This guarantees
    identical geometry in every band and preserves the native TNG relative
    SED.  The stamp side is a consequence of the light footprint, never the
    matched quantity.
    """
    source, use_angle = _prepare_tng_continuous_source(
        galaxy_dir, subhalo_id, orientation,
        rot_angle=rot_angle, fits_bands=fits_bands,
        radius_manifest_fingerprint=radius_manifest_fingerprint,
    )
    return _render_tng_continuous_source(
        source,
        scale=scale,
        rot_k=rot_k,
        use_angle=use_angle,
        subhalo_id=subhalo_id,
        orientation=orientation,
        rot_angle=rot_angle,
        pixel_scale_arcsec=pixel_scale_arcsec,
        max_output_side=max_output_side,
    )


# ---------------------------------------------------------------------------
# Enumeration + random sampling (for injection into synthetic scenes)
# ---------------------------------------------------------------------------

N_ORIENTATIONS = 5                              # SKIRT viewpoints O1..O5
DOWNSAMPLE_CHOICES: tuple[int, ...] = (1, 2, 3, 4)


def list_tng_galaxies(tng_dir: str) -> list[tuple[str, str]]:
    """List ``(galaxy_dir, subhalo_id)`` for downloaded galaxies ready to inject.

    A galaxy qualifies if its folder holds a ``.done`` marker AND the VIS O1
    frame exists, so :func:`prepare_tng_galaxy` won't choke on a partial dir."""
    if not os.path.isdir(tng_dir):
        return []
    out: list[tuple[str, str]] = []
    for gid in os.listdir(tng_dir):
        gdir = os.path.join(tng_dir, gid)
        if (os.path.isfile(os.path.join(gdir, Config.Tng.DONE_MARKER))
                and os.path.isfile(tng_fits_path(gdir, gid, 1, "VIS"))):
            out.append((gdir, gid))
    try:
        return sorted(out, key=lambda t: int(t[1]))
    except ValueError:
        return sorted(out)


#: Per-(dir, galaxy, orientation) native VIS half-light radius cache (pixels).
#: Filled lazily the first time a galaxy is sized; one entry per orientation, so
#: repeat draws skip the FITS read + curve-of-growth entirely. Keyed on the
#: directory too, so distinct galaxy sets (e.g. test fixtures reusing ids) never
#: alias.
_HALFLIGHT_PX_CACHE: dict[tuple[str, str, int], float] = {}


def native_halflight_px(
    galaxy_dir: str, subhalo_id: int | str, orientation: int,
    *, fits_band: str = "VIS",
) -> float:
    """Cached native half-light radius (px) of one galaxy/orientation's VIS
    frame. NaN if the frame can't be read or carries no flux."""
    key = (str(galaxy_dir), str(subhalo_id), int(orientation))
    cached = _HALFLIGHT_PX_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        frame = load_tng_frame(
            tng_fits_path(galaxy_dir, subhalo_id, orientation, fits_band))
        re_px = measure_halflight_radius_px(frame)
    except Exception:
        re_px = float("nan")
    _HALFLIGHT_PX_CACHE[key] = re_px
    return re_px


def truncate_below_sb(
    stamp: np.ndarray,
    pixel_scale_arcsec: float,
    sb_cut_mag_arcsec2: float,
) -> np.ndarray:
    """Zero pixels fainter than ``sb_cut_mag_arcsec2`` (AB, per band) and
    crop to the surviving footprint, kept square and centred so the galaxy
    stays at the stamp centre.

    The SKIRT box is 160 kpc with nonzero light everywhere, with outskirts
    far below Euclid's detection limit. Modifies per-band channels in place
    and returns the (possibly smaller) cropped view.
    """
    if not (sb_cut_mag_arcsec2 > 0.0):
        return stamp
    pix2 = pixel_scale_arcsec ** 2
    keep = np.zeros(stamp.shape[:2], dtype=bool)
    for k, name in enumerate(Config.LR_INPUT_BAND_NAMES):
        # SB cut (mag/arcsec²) → e⁻/arcsec² → e⁻/pixel via the pixel area.
        thr = ab_mag_to_electrons(sb_cut_mag_arcsec2,
                                  Config.get_band(name)) * pix2
        ch = stamp[..., k]
        ch[ch < thr] = 0.0
        keep |= ch > 0.0
    if not keep.any():
        return stamp
    # Crop at the radius enclosing 99.5% of the surviving (band-summed)
    # flux, so a faint outlying satellite blob doesn't hold the whole
    # 160 kpc box open.
    total = stamp.sum(axis=2, dtype=np.float64)
    H, W = total.shape
    rint = radius_int_grid((H, W))
    prof = np.bincount(rint.ravel(), weights=total.ravel())
    cum = np.cumsum(prof)
    r = int(np.searchsorted(cum, 0.995 * cum[-1])) + 4
    cy, cx = int(round((H - 1) / 2.0)), int(round((W - 1) / 2.0))
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    return stamp[y0:y1, x0:x1]


def _normalise_target_vis(
    stamp: np.ndarray,
    target_vis_flux_e: float | None,
) -> float:
    if target_vis_flux_e is None:
        return 1.0
    current = float(stamp[..., 0].sum(dtype=np.float64))
    if current <= 0.0 or not np.isfinite(current) or target_vis_flux_e < 0.0:
        raise ValueError("target VIS flux cannot be applied to an empty stamp")
    factor = float(target_vis_flux_e) / current
    stamp *= np.float32(factor)
    return factor


def _record_target_vis_normalisation(
    stamp: np.ndarray,
    meta: dict,
    target_vis_flux_e: float | None,
) -> float:
    """Apply and record one cube-wide VIS anchor after geometry is final."""
    brightness_scale = _normalise_target_vis(stamp, target_vis_flux_e)
    meta["brightness_scale"] = float(brightness_scale)
    meta["shared_photometric_scale"] = float(brightness_scale)
    meta["photometric_scaling"] = "single_shared_vis_anchor_after_nominal_scale"
    meta["flux_e_per_band"] = {
        band: float(stamp[..., index].sum(dtype=np.float64))
        for index, band in enumerate(Config.LR_INPUT_BAND_NAMES)
    }
    return brightness_scale


def circularize_psf_kernel(kernel: np.ndarray) -> np.ndarray:
    """Return an azimuthally averaged, unit-sum copy of an empirical PSF."""
    values = np.asarray(kernel, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 3 or not np.all(np.isfinite(values)):
        raise ValueError("aperture PSF kernel must be a finite 2-D array")
    yy, xx = np.indices(values.shape, dtype=np.float64)
    cy, cx = 0.5 * (values.shape[0] - 1), 0.5 * (values.shape[1] - 1)
    radius = np.floor(np.hypot(yy - cy, xx - cx) + 0.5).astype(np.int64)
    radial_sum = np.bincount(radius.ravel(), weights=values.ravel())
    radial_count = np.bincount(radius.ravel())
    profile = np.divide(
        radial_sum, radial_count,
        out=np.zeros_like(radial_sum), where=radial_count > 0,
    )
    circular = profile[radius]
    total = float(np.sum(circular))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("aperture PSF kernel has non-positive total flux")
    return np.asarray(circular / total, dtype=np.float32)


def _array_fingerprint(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _cached_circularized_psf(
    kernel: np.ndarray,
    *,
    psf_identity: str,
) -> tuple[np.ndarray, str]:
    """Return a byte-bounded cached circularization and its fingerprint."""
    global _CIRCULAR_PSF_CACHE_BYTES
    values = np.ascontiguousarray(kernel, dtype=np.float32)
    input_fingerprint = _array_fingerprint(values)
    key = (str(psf_identity), input_fingerprint)
    cached = _CIRCULAR_PSF_CACHE.pop(key, None)
    if cached is not None:
        _CIRCULAR_PSF_CACHE[key] = cached
        return cached, _array_fingerprint(cached)
    circular = circularize_psf_kernel(values)
    circular.setflags(write=False)
    if circular.nbytes <= _CIRCULAR_PSF_CACHE_MAX_BYTES:
        while (
            _CIRCULAR_PSF_CACHE
            and _CIRCULAR_PSF_CACHE_BYTES + circular.nbytes
            > _CIRCULAR_PSF_CACHE_MAX_BYTES
        ):
            _, evicted = _CIRCULAR_PSF_CACHE.popitem(last=False)
            _CIRCULAR_PSF_CACHE_BYTES -= int(evicted.nbytes)
        _CIRCULAR_PSF_CACHE[key] = circular
        _CIRCULAR_PSF_CACHE_BYTES += int(circular.nbytes)
    return circular, _array_fingerprint(circular)


def _aperture_response(
    aperture: np.ndarray,
    circular_psf: np.ndarray,
    *,
    psf_identity: str,
    psf_fwhm_arcsec: float,
    pixel_scale_arcsec: float,
    centre_parity: tuple[int, int],
) -> np.ndarray:
    """Return cached ``A * K[::-1]`` on its compact, full support."""
    global _APERTURE_RESPONSE_CACHE_BYTES
    aperture_values = np.ascontiguousarray(aperture, dtype=np.uint8)
    psf_values = np.ascontiguousarray(circular_psf, dtype=np.float32)
    key = (
        str(psf_identity), _array_fingerprint(psf_values),
        float(psf_fwhm_arcsec), float(pixel_scale_arcsec),
        tuple(int(value) for value in centre_parity),
        aperture_values.shape, aperture_values.tobytes(),
    )
    cached = _APERTURE_RESPONSE_CACHE.pop(key, None)
    if cached is not None:
        _APERTURE_RESPONSE_CACHE[key] = cached
        return cached
    # Empirical VIS kernels are commonly about 1,000 px across.  Direct
    # convolution makes this one-time response build take many CPU minutes
    # even though both inputs are compact; FFT convolution is algebraically
    # identical here and completes in well under a second.
    response = fftconvolve(
        aperture_values.astype(np.float64),
        np.asarray(psf_values[::-1, ::-1], dtype=np.float64),
        mode="full",
    )
    response = np.asarray(response, dtype=np.float64)
    response.setflags(write=False)
    if response.nbytes <= _APERTURE_RESPONSE_CACHE_MAX_BYTES:
        while (
            _APERTURE_RESPONSE_CACHE
            and _APERTURE_RESPONSE_CACHE_BYTES + response.nbytes
            > _APERTURE_RESPONSE_CACHE_MAX_BYTES
        ):
            _, evicted = _APERTURE_RESPONSE_CACHE.popitem(last=False)
            _APERTURE_RESPONSE_CACHE_BYTES -= int(evicted.nbytes)
        _APERTURE_RESPONSE_CACHE[key] = response
        _APERTURE_RESPONSE_CACHE_BYTES += int(response.nbytes)
    return response


def measure_vis_2fwhm_aperture_flux(
    vis: np.ndarray,
    *,
    circular_psf: np.ndarray,
    psf_fwhm_arcsec: float,
    pixel_scale_arcsec: float,
    psf_identity: str = "",
) -> float:
    """Measure the centred 2FWHM flux through a compact adjoint response.

    This is algebraically identical to summing a circular aperture after
    ``fftconvolve(vis, circular_psf, mode='same')``.  The aperture has diameter
    ``2 * FWHM`` and therefore radius ``FWHM``.  Only the input pixels within
    the aperture-plus-PSF support enter the dot product.
    """
    image = np.asarray(vis, dtype=np.float64)
    kernel = np.asarray(circular_psf, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 1:
        raise ValueError("VIS aperture measurement requires a non-empty 2-D image")
    if kernel.ndim != 2 or min(kernel.shape) < 1:
        raise ValueError("VIS aperture measurement requires a 2-D PSF")
    fwhm = float(psf_fwhm_arcsec)
    scale = float(pixel_scale_arcsec)
    if not np.isfinite(fwhm) or fwhm <= 0.0 or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("VIS aperture FWHM and pixel scale must be positive")

    height, width = image.shape
    cy, cx = 0.5 * (height - 1), 0.5 * (width - 1)
    radius = fwhm / scale
    y0 = max(0, int(math.ceil(cy - radius)))
    y1 = min(height - 1, int(math.floor(cy + radius)))
    x0 = max(0, int(math.ceil(cx - radius)))
    x1 = min(width - 1, int(math.floor(cx + radius)))
    yy, xx = np.indices((y1 - y0 + 1, x1 - x0 + 1), dtype=np.float64)
    aperture = np.hypot(yy + y0 - cy, xx + x0 - cx) <= radius
    if not np.any(aperture):
        return 0.0
    response = _aperture_response(
        aperture, kernel,
        psf_identity=psf_identity,
        psf_fwhm_arcsec=fwhm,
        pixel_scale_arcsec=scale,
        centre_parity=(height % 2, width % 2),
    )

    kernel_height, kernel_width = kernel.shape
    same_y0 = (kernel_height - 1) // 2
    same_x0 = (kernel_width - 1) // 2
    input_y0 = y0 + same_y0 - (kernel_height - 1)
    input_x0 = x0 + same_x0 - (kernel_width - 1)
    response_y0 = max(0, -input_y0)
    response_x0 = max(0, -input_x0)
    response_y1 = min(response.shape[0], height - input_y0)
    response_x1 = min(response.shape[1], width - input_x0)
    if response_y1 <= response_y0 or response_x1 <= response_x0:
        return 0.0
    image_y0 = input_y0 + response_y0
    image_x0 = input_x0 + response_x0
    image_y1 = input_y0 + response_y1
    image_x1 = input_x0 + response_x1
    return float(np.sum(
        image[image_y0:image_y1, image_x0:image_x1]
        * response[response_y0:response_y1, response_x0:response_x1],
        dtype=np.float64,
    ))


def normalise_tng_to_vis_2fwhm(
    stamp: np.ndarray,
    meta: dict,
    *,
    target_flux_e: float,
    psf_kernel: np.ndarray,
    psf_fwhm_arcsec: float,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
    psf_identity: str = "",
) -> tuple[np.ndarray, dict]:
    """Scale a geometry-final TNG cube to a goal VIS 2FWHM aperture flux."""
    cube = np.asarray(stamp, dtype=np.float32)
    target = float(target_flux_e)
    fwhm = float(psf_fwhm_arcsec)
    scale = float(pixel_scale_arcsec)
    if cube.ndim != 3 or cube.shape[-1] != len(Config.LR_INPUT_BAND_NAMES):
        raise ValueError("TNG aperture normalization requires an (H,W,4) stamp")
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("goal VIS 2FWHM aperture flux must be positive")
    if not np.isfinite(fwhm) or fwhm <= 0.0 or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("VIS aperture PSF FWHM and pixel scale must be positive")
    circular_psf, psf_fingerprint = _cached_circularized_psf(
        psf_kernel, psf_identity=psf_identity,
    )
    measured = measure_vis_2fwhm_aperture_flux(
        cube[..., 0], circular_psf=circular_psf,
        psf_fwhm_arcsec=fwhm, pixel_scale_arcsec=scale,
        psf_identity=psf_identity or psf_fingerprint,
    )
    if not np.isfinite(measured) or measured <= 0.0:
        raise ValueError("resized TNG has no positive VIS 2FWHM aperture flux")
    brightness_scale = target / measured
    if not np.isfinite(brightness_scale) or brightness_scale <= 0.0:
        raise ValueError("VIS 2FWHM brightness scale is invalid")
    cube *= np.float32(brightness_scale)
    achieved = measured * brightness_scale
    updated = dict(meta)
    updated.update({
        "brightness_scale": float(brightness_scale),
        "shared_photometric_scale": float(brightness_scale),
        "photometric_scaling": "vis_2fwhm_after_redshift_and_nominal_scale",
        "target_vis_2fwhm_flux_e": target,
        "achieved_vis_2fwhm_flux_e": float(achieved),
        "target_vis_2fwhm_mag": float(electrons_to_ab_mag(
            target, Config.get_band("VIS"),
        )),
        "achieved_vis_2fwhm_mag": float(electrons_to_ab_mag(
            achieved, Config.get_band("VIS"),
        )),
        "aperture_psf_fwhm_arcsec": fwhm,
        "aperture_radius_arcsec": fwhm,
        "aperture_diameter_arcsec": 2.0 * fwhm,
        "aperture_psf_model": "circularized_empirical_vis_psf",
        "aperture_response_method": "compact_adjoint_v1",
        "aperture_psf_fingerprint": psf_fingerprint,
        "flux_e_per_band": {
            band: float(cube[..., index].sum(dtype=np.float64))
            for index, band in enumerate(Config.LR_INPUT_BAND_NAMES)
        },
    })
    return cube, updated


def _record_nominal_radius(
    meta: dict,
    *,
    native_re_px: float,
    target_re_arcsec: float,
    scale: float,
    pixel_scale_arcsec: float,
    radius_manifest_fingerprint: str,
) -> None:
    nominal = float(native_re_px) * float(pixel_scale_arcsec) * float(scale)
    meta.update({
        "native_halflight_px": float(native_re_px),
        "target_re_arcsec": float(target_re_arcsec),
        "nominal_re_arcsec": nominal,
        "radius_scale_factor": float(scale),
        "scale_factor": float(scale),
        "radius_rendering": TNG_RADIUS_RENDERING,
        "radius_renderer_version": TNG_RADIUS_RENDERER_VERSION,
        "radius_renderer_fingerprint": TNG_RADIUS_RENDERER_FINGERPRINT,
        "radius_remeasured": False,
        "native_tng_sed_preserved": True,
        "radius_manifest_fingerprint": str(radius_manifest_fingerprint),
    })


def tng_stamp_to_target_re(
    galaxy_dir: str,
    subhalo_id: int | str,
    orientation: int,
    target_re_arcsec: float,
    *,
    rng: np.random.Generator | None = None,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
    target_vis_flux_e: float | None = None,
    native_re_px: float | None = None,
    radius_manifest_fingerprint: str = "",
    max_output_side: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Render one TNG stamp at a nominal continuous-space Euclid Sersic R_e."""
    if not np.isfinite(target_re_arcsec) or target_re_arcsec <= 0.0:
        raise ValueError(f"target_re_arcsec must be positive, got {target_re_arcsec!r}")
    if native_re_px is None:
        raise ValueError(
            "one-pass Euclid radius rendering requires native_re_px from "
            "the validated radius manifest"
        )
    native = float(native_re_px)
    if not np.isfinite(native) or native <= 0.0:
        raise ValueError("TNG VIS frame has no measurable native half-light radius")
    rot_k = int(rng.integers(0, 4)) if rng is not None else 0
    rot_angle = (float(rng.uniform(0.0, 360.0))
                 if rng is not None else None)
    scale = float(target_re_arcsec) / (float(native) * pixel_scale_arcsec)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("computed TNG-to-Euclid scale is invalid")
    stamp, meta = prepare_tng_galaxy_continuous(
        galaxy_dir, subhalo_id, orientation, scale=scale,
        rot_k=rot_k, rot_angle=rot_angle,
        pixel_scale_arcsec=pixel_scale_arcsec,
        radius_manifest_fingerprint=radius_manifest_fingerprint,
        max_output_side=max_output_side,
    )
    _record_target_vis_normalisation(stamp, meta, target_vis_flux_e)
    _record_nominal_radius(
        meta, native_re_px=native, target_re_arcsec=target_re_arcsec,
        scale=scale, pixel_scale_arcsec=pixel_scale_arcsec,
        radius_manifest_fingerprint=radius_manifest_fingerprint,
    )
    return stamp, meta


def tng_stamp_at_redshift(
    galaxy_dir: str,
    subhalo_id: int | str,
    orientation: int,
    z: float,
    rng: np.random.Generator | None = None,
    *,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
    rot_k: int | None = None,
    f_max: int = TNG_MAX_REBIN_FACTOR,
    sb_cut_mag_arcsec2: float = Config.TNG_SB_TRUNCATE_MAG_ARCSEC2,
    mass_scale: float = 1.0,
    target_re_arcsec: float | None = None,
    target_vis_flux_e: float | None = None,
    native_re_px: float | None = None,
    radius_manifest_fingerprint: str = "",
    max_output_side: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Build one TNG stamp **as it would appear at redshift ``z``**.

    ``mass_scale`` < 1 re-uses the stamp as a *smaller galaxy of similar
    morphology*: flux × s (L ∝ M) and an extra size squeeze s^-α along the
    observed mass-size relation R ∝ M^α — so its surface brightness drops
    as s^(1-2α), the observed trend. Distinct from the fixed-mass
    compactness correction, which conserves flux.

    When no explicit target radius/flux is supplied, a single ``z`` drives all
    three observables (see :mod:`euclid_polish.sky.generation.redshift_model`):

    * the block-mean factor comes from the angular size of the 100 pc native
      pixel at D_A(z) (stochastically rounded with ``rng`` so the mean
      apparent size is unbiased), times the flux-conserving compactness
      correction C(z) for the z = 0 atlas morphologies
      (:func:`~euclid_polish.sky.generation.redshift_model.compactness_factor`);
    * Tolman (1+z)⁻³ surface-brightness dimming;
    * a randomized spectral drift across the four bands, anchored on the
      stamp's own 4-point SED (``rng=None`` → deterministic drift only);
    * outskirts fainter than ``sb_cut_mag_arcsec2`` are truncated and the
      stamp cropped (:func:`truncate_below_sb`).

    Raises on unreadable frames, like :func:`prepare_tng_galaxy`.
    """
    compact = compactness_factor(z)
    if not (0.0 < mass_scale <= 1.0):
        raise ValueError(f"mass_scale must be in (0, 1], got {mass_scale}")
    squeeze = compact * mass_scale ** -Config.TNG_MASS_SIZE_ALPHA
    f_geo = rebin_factor_for_redshift(z, pixel_scale_arcsec=pixel_scale_arcsec)
    re_px = (
        float(native_re_px) if native_re_px is not None else float("nan")
    )
    if target_re_arcsec is not None:
        if native_re_px is None:
            raise ValueError(
                "one-pass Euclid radius rendering requires native_re_px from "
                "the validated radius manifest"
            )
        if not np.isfinite(target_re_arcsec) or target_re_arcsec <= 0.0:
            raise ValueError(f"target_re_arcsec must be positive, got {target_re_arcsec!r}")
        if not np.isfinite(re_px) or re_px <= 0.0:
            raise ValueError("TNG VIS frame has no measurable native half-light radius")
        if rot_k is None:
            rot_k = int(rng.integers(0, 4)) if rng is not None else 0
        rot_angle = (float(rng.uniform(0.0, 360.0))
                     if rng is not None else None)
        scale = float(target_re_arcsec) / (float(re_px) * pixel_scale_arcsec)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("computed TNG-to-Euclid scale is invalid")
        stamp, meta = prepare_tng_galaxy_continuous(
            galaxy_dir, subhalo_id, orientation, scale=scale,
            rot_k=int(rot_k), rot_angle=rot_angle,
            pixel_scale_arcsec=pixel_scale_arcsec,
            radius_manifest_fingerprint=radius_manifest_fingerprint,
            max_output_side=max_output_side,
        )
        # Geometry is final before brightness. Apply only the redshift-driven
        # relative band drift here; the later 2FWHM normalization supplies the
        # independent common flux anchor and therefore cancels any common
        # luminosity-distance factor.
        sed_fnu = [
            meta["flux_e_per_band"][band]
            / mjy_per_sr_to_electrons_factor(
                Config.get_band(band), pixel_scale_arcsec,
            )
            for band in Config.LR_INPUT_BAND_NAMES
        ]
        factors, dmeta = band_drift_factors(sed_fnu, z, rng)
        stamp *= np.asarray(factors, dtype=np.float32)[None, None, :]
        # The staged generator deliberately supplies no target here and applies
        # its independent 2FWHM anchor later.  Preserve the legacy total-VIS
        # request, however, by restoring it after the colour drift.
        if target_vis_flux_e is not None:
            _record_target_vis_normalisation(
                stamp, meta, target_vis_flux_e,
            )
        meta["flux_e_per_band"] = {
            band: float(stamp[..., index].sum(dtype=np.float64))
            for index, band in enumerate(Config.LR_INPUT_BAND_NAMES)
        }
        _record_nominal_radius(
            meta, native_re_px=re_px, target_re_arcsec=target_re_arcsec,
            scale=scale, pixel_scale_arcsec=pixel_scale_arcsec,
            radius_manifest_fingerprint=radius_manifest_fingerprint,
        )
        meta.update({
            "z": float(z),
            "physical_redshift_rescaling_applied": True,
            "redshift_band_factors": [float(value) for value in factors],
        })
        meta.update(dmeta)
        return stamp, meta
    f_cont = min(float(f_max), f_geo * squeeze)
    rebin = stochastic_round_factor(f_cont, rng)
    if rot_k is None:
        rot_k = int(rng.integers(0, 4)) if rng is not None else 0
    # Arbitrary orientation for the dataset-multiplying augmentation (applied by
    # prepare_tng_galaxy only when rebin ≥ ARBITRARY_ROTATION_MIN_REBIN); rng=None
    # → no angle → exact quarter-turn, preserving the deterministic path.
    rot_angle = float(rng.uniform(0.0, 360.0)) if rng is not None else None

    stamp, meta = prepare_tng_galaxy(
        galaxy_dir, subhalo_id, orientation,
        rebin_factor=rebin, rot_k=rot_k, rot_angle=rot_angle,
        pixel_scale_arcsec=pixel_scale_arcsec,
    )
    # The stamp's own rest-frame 4-point SED in relative f_ν: undo each
    # band's linear MJy/sr → electrons factor so the zeropoints/integration
    # don't masquerade as colour.
    sed_fnu = [
        meta["flux_e_per_band"][b]
        / mjy_per_sr_to_electrons_factor(Config.get_band(b), pixel_scale_arcsec)
        for b in Config.LR_INPUT_BAND_NAMES
    ]
    factors, dmeta = band_drift_factors(sed_fnu, z, rng)
    # Flux: the block-mean keeps surface brightness while the pixel count
    # shrinks, so (rebin/f_geo)² pins the total to the *continuous
    # geometric* prediction (covering the compactness squeeze and the
    # integer rounding); mass_scale then dims the rescaled galaxy (L ∝ M).
    stamp *= (np.asarray(factors, dtype=np.float32)[None, None, :]
              * np.float32((rebin / f_geo) ** 2 * mass_scale))
    brightness_scale = 1.0
    if target_vis_flux_e is not None:
        current_vis = float(stamp[..., 0].sum(dtype=np.float64))
        if current_vis > 0.0 and target_vis_flux_e >= 0.0:
            brightness_scale = float(target_vis_flux_e) / current_vis
            # One scalar preserves the TNG VIS/NISP proportions.
            stamp *= np.float32(brightness_scale)
    stamp = truncate_below_sb(stamp, pixel_scale_arcsec, sb_cut_mag_arcsec2)
    # Restore the fitted VIS brightness after cropping, again with one shared
    # scalar so no COSMOS proxy colour is injected into the TNG SED.
    if target_vis_flux_e is not None:
        current_vis = float(stamp[..., 0].sum(dtype=np.float64))
        if current_vis > 0.0 and target_vis_flux_e >= 0.0:
            correction = float(target_vis_flux_e) / current_vis
            stamp *= np.float32(correction)
            brightness_scale *= correction
    meta["flux_e_per_band"] = {
        b: float(stamp[..., k].sum())
        for k, b in enumerate(Config.LR_INPUT_BAND_NAMES)
    }
    meta["shape"] = tuple(stamp.shape)
    meta["sb_cut_mag_arcsec2"] = float(sb_cut_mag_arcsec2)
    meta["z"] = float(z)
    meta["rebin_factor_continuous"] = float(f_cont)
    meta["compactness"] = float(compact)
    meta["mass_scale"] = float(mass_scale)
    meta["redshift_band_factors"] = [float(f) for f in factors]
    meta.update(dmeta)
    if np.isfinite(re_px) and re_px > 0.0:
        meta["native_halflight_px"] = float(re_px)
        meta["apparent_re_arcsec"] = float(
            pixel_scale_arcsec * re_px / rebin
            if target_re_arcsec is not None
            else physical_pc_to_arcsec(
                re_px * TNG_NATIVE_PC_PER_PIXEL, z) / squeeze
        )
    if target_vis_flux_e is not None:
        meta["target_vis_flux_e"] = float(target_vis_flux_e)
        meta["brightness_scale"] = float(brightness_scale)
    return stamp, meta


#: Per-(dir, galaxy, orientation) native photometry cache: the VIS frame's
#: mean-SB radial profile (MJy/sr per native-pixel radius) + each band's
#: total MJy/sr sum. One 4-frame read each, then dict lookups — powers the
#: ANALYTIC lens-showability predictors (no rendering).
_NATIVE_PHOTOM_CACHE: dict[tuple[str, str, int],
                           tuple[np.ndarray, np.ndarray]] = {}


def native_photometry(galaxy_dir: str, subhalo_id: int | str,
                      orientation: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached ``(vis_sb_profile, band_sums)`` for one atlas frame set."""
    key = (str(galaxy_dir), str(subhalo_id), int(orientation))
    cached = _NATIVE_PHOTOM_CACHE.get(key)
    if cached is not None:
        return cached
    sums = []
    profile = None
    for fband in TNG_FITS_BANDS:
        frame = load_tng_frame(
            tng_fits_path(galaxy_dir, subhalo_id, orientation, fband))
        sums.append(float(frame.sum()))
        if fband == "VIS":
            rint = radius_int_grid(frame.shape)
            flux = np.bincount(rint.ravel(), weights=frame.ravel())
            cnt = np.bincount(rint.ravel()).astype(np.float64)
            profile = (flux / np.maximum(cnt, 1.0)).astype(np.float64)
    out = (profile, np.asarray(sums, dtype=np.float64))
    _NATIVE_PHOTOM_CACHE[key] = out
    return out


def predict_visible_radius_arcsec(
    galaxy_dir: str, subhalo_id: int | str, orientation: int, z: float,
    *,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
    sb_cut_mag_arcsec2: float = Config.TNG_SB_TRUNCATE_MAG_ARCSEC2,
) -> float:
    """Analytic prediction of a stamp's visible radius at redshift ``z`` —
    where the dimmed/drifted/compactness-boosted mean VIS surface brightness
    crosses the truncation threshold. Approximate (mean profile, VIS only,
    deterministic drift) but cheap: scalars on the cached native profile,
    no stamp load. Used to reject unshowable lens systems before rendering;
    a post-render check remains the backstop.
    """
    profile, sums = native_photometry(galaxy_dir, subhalo_id, orientation)
    band = Config.get_band("VIS")
    factors, _ = band_drift_factors(sums, z, None)        # dimming + drift
    compact = compactness_factor(z)
    fac = mjy_per_sr_to_electrons_factor(band, pixel_scale_arcsec)
    sb_e = profile * fac * factors[0] * compact ** 2      # e⁻/HR-pixel
    thr = (ab_mag_to_electrons(sb_cut_mag_arcsec2, band)
           * pixel_scale_arcsec ** 2)
    above = np.nonzero(sb_e >= thr)[0]
    if above.size == 0:
        return 0.0
    return physical_pc_to_arcsec(
        float(above.max()) * TNG_NATIVE_PC_PER_PIXEL, z) / compact


def predict_vis_flux_e(
    galaxy_dir: str, subhalo_id: int | str, orientation: int, z: float,
    *,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
    mass_scale: float = 1.0,
) -> float:
    """Analytic prediction of a stamp's total VIS flux at redshift ``z``
    (electrons; truncation losses ignored). The flux-conservation boost makes
    the total independent of the integer rebin: native sum × conversion ×
    (dimming · drift) / f_geo² × mass_scale.
    """
    _, sums = native_photometry(galaxy_dir, subhalo_id, orientation)
    factors, _ = band_drift_factors(sums, z, None)
    f_geo = rebin_factor_for_redshift(z, pixel_scale_arcsec=pixel_scale_arcsec)
    fac = mjy_per_sr_to_electrons_factor(
        Config.get_band("VIS"), pixel_scale_arcsec)
    return float(sums[0] * fac * factors[0] / f_geo ** 2 * mass_scale)


def sample_tng_stamp(
    galaxies: list[tuple[str, str]],
    rng: np.random.Generator,
    *,
    pixel_scale_arcsec: float = Config.DEFAULT_PIXEL_SCALE,
    downsample_choices: tuple[int, ...] = DOWNSAMPLE_CHOICES,
    target_re_arcsec: float | None = None,
    z: float | None = None,
    mass_scale: float = 1.0,
    f_max: int = TNG_MAX_REBIN_FACTOR,
    target_vis_flux_e: float | None = None,
    native_re_px: float | None = None,
    radius_manifest_fingerprint: str = "",
    radius_lookup_map: dict[tuple[str, int], float] | None = None,
    max_output_side: int | None = None,
) -> tuple[np.ndarray, dict] | None:
    """Pick a random galaxy / orientation / downsample / quarter-rotation and
    return its injectable ``(H,W,4)`` electron stamp + meta (None if it can't
    load).

    Sizing (first match wins):

    * ``z`` and ``target_re_arcsec`` given → the Euclid-conditioned path uses
      one nominal similarity scale and one cube-wide VIS normalization. TNG
      inter-band ratios are unchanged and the rendered radius is not remeasured.
    * ``z`` alone → the explicit physical-redshift path applies D_A(z),
      dimming, and spectral drift for strong-lens rendering.
    * ``target_re_arcsec`` alone → the same single-scale cube path without the
      physical-redshift metadata.
    * otherwise → uniform draw from ``downsample_choices`` (×1/×2/×3/×4);
      coarser = smaller and fainter, like a more distant galaxy.
    """
    if not galaxies:
        return None
    gdir, gid = galaxies[int(rng.integers(0, len(galaxies)))]
    orientation = int(rng.integers(1, N_ORIENTATIONS + 1))      # O1..O5
    selected_native_re = (
        radius_lookup_map.get((str(gid), orientation))
        if radius_lookup_map is not None else native_re_px
    )
    if radius_lookup_map is not None and selected_native_re is None:
        raise ValueError(
            f"radius manifest has no entry for TNG {gid} orientation {orientation}"
        )

    if z is not None:
        return tng_stamp_at_redshift(
            gdir, gid, orientation, z, rng,
            pixel_scale_arcsec=pixel_scale_arcsec, f_max=f_max,
            mass_scale=mass_scale,
            target_re_arcsec=target_re_arcsec,
            target_vis_flux_e=target_vis_flux_e,
            native_re_px=selected_native_re,
            radius_manifest_fingerprint=radius_manifest_fingerprint,
            max_output_side=max_output_side)

    if target_re_arcsec is not None:
        return tng_stamp_to_target_re(
            gdir, gid, orientation, target_re_arcsec,
            rng=rng, pixel_scale_arcsec=pixel_scale_arcsec,
            target_vis_flux_e=target_vis_flux_e,
            native_re_px=selected_native_re,
            radius_manifest_fingerprint=radius_manifest_fingerprint,
            max_output_side=max_output_side,
        )

    rot_k = int(rng.integers(0, 4))
    rot_angle = float(rng.uniform(0.0, 360.0))   # used when rebin ≥ threshold

    rebin = int(downsample_choices[
        int(rng.integers(0, len(downsample_choices)))])

    stamp, meta = prepare_tng_galaxy(
        gdir, gid, orientation,
        rebin_factor=rebin, rot_k=rot_k, rot_angle=rot_angle,
        pixel_scale_arcsec=pixel_scale_arcsec,
    )
    return stamp, meta
