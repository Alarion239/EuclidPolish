"""Private image mechanics for native TNG atlas products.

SKIRT broadband images are commonly stored as surface brightness in MJy/sr.
The helpers here load those products into validated native images, preserve their
intensive units while rebinning, provide controlled rotation augmentation, and
measure centred curves of growth.  They do not assign a distance or redshift,
or convert into detector-specific units.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from astropy import units as u
from astropy.io import fits
from numba import njit

from euclid_polish.config import Config
from euclid_polish.tng._paths import dust_free_twin
from euclid_polish.tng.image import TNGSurfaceBrightnessImage

# Generation already parallelises over one process per allocated CPU.  OpenCV
# otherwise creates a thread pool in every worker and oversubscribes the outer
# process pool (for example, 10 x 10 runnable threads on a 10-CPU job).
cv2.setNumThreads(1)


def load_star_mask(
    dusty_paths: Sequence[str | PathLike[str]],
    *,
    required: bool,
) -> np.ndarray | None:
    """Where any dust-free twin of one orientation's frames has light.

    SKIRT's dusty frames carry dust-scattered light far beyond the stars: in
    massive galaxies a flat box filling the simulation volume, with straight
    edges, radial streaks and shadows (2026-10-19, subhalo 117266: 15-36 % of
    the VIS light, absent from every dust-free view). Light where no dust-free
    band has any is scattered light only, so a frame keeps its pixels where
    the mask is true; one mask for all bands keeps the edges identical.
    Returns ``None`` when a twin is missing, or raises if ``required``.
    """
    twins = _present_twins(dusty_paths, required=required)
    if twins is None:
        return None
    return _union_mask(twins, [_lit_pixels(twin) for twin in twins])


def _present_twins(
    dusty_paths: Sequence[str | PathLike[str]],
    *,
    required: bool,
) -> list[Path] | None:
    """The dust-free twins of ``dusty_paths``, or ``None`` when one is missing
    (an error if ``required``)."""
    twins = [dust_free_twin(Path(path)) for path in dusty_paths]
    missing = [twin for twin in twins if not twin.is_file()]
    if missing:
        if required:
            raise FileNotFoundError(
                f"dust-free twin missing: {missing[0]} (scripts/"
                "fasrc_download_tng_skirt_atlas.py keeps them; re-fetch the galaxy)"
            )
        return None
    return twins


def _lit_pixels(twin: str | PathLike[str]) -> np.ndarray:
    """Where one dust-free frame is > 0. A direct comparison gives what
    ``nan_to_num(float64(x)) > 0`` gives: NaN and -inf are dark, +inf lit."""
    with fits.open(twin, memmap=False) as hdul:
        return np.asarray(hdul[0].data) > 0.0


def _union_mask(twins: Sequence[Path], lit: Sequence[np.ndarray]) -> np.ndarray:
    mask: np.ndarray | None = None
    for twin, frame in zip(twins, lit, strict=True):
        if mask is not None and frame.shape != mask.shape:
            raise ValueError(f"dust-free twins differ in shape: {twin}")
        mask = frame if mask is None else (mask | frame)
    assert mask is not None
    return mask


@dataclass(frozen=True)
class ViewRead:
    """The eight FITS reads of one donor view, in flight on a thread pool
    (:func:`start_view_read`); :func:`finish_view_read` assembles them."""

    paths: tuple[Path, ...]
    twins: tuple[Path, ...]
    primaries: tuple[Future, ...]
    lit: tuple[Future, ...] | None          # None: a dust-free twin is missing


def start_view_read(
    dusty_paths: Sequence[str | PathLike[str]],
    pool: Executor,
) -> ViewRead:
    """Submit one view's dusty frames and dust-free twins to ``pool``.

    Generation loads a donor view for nearly every galaxy it draws; read
    serially, its ~80 MB left the CPU idle on network storage. The files are
    read concurrently, and a renderer can start them before it needs them.
    The twins are read only for the star mask (``Config.Tng.STAR_MASK``).
    """
    paths = tuple(Path(path) for path in dusty_paths)
    twins = tuple(dust_free_twin(path) for path in paths)
    have_twins = Config.Tng.STAR_MASK and all(twin.is_file() for twin in twins)
    return ViewRead(
        paths=paths,
        twins=twins,
        primaries=tuple(pool.submit(_read_primary, path) for path in paths),
        lit=(tuple(pool.submit(_lit_pixels, twin) for twin in twins)
             if have_twins else None),
    )


def finish_view_read(
    read: ViewRead,
    band_names: Sequence[str],
) -> tuple[TNGSurfaceBrightnessImage, np.ndarray | None]:
    """One orientation's dusty planes, stacked in ``band_names`` order, and
    its star mask, ``None`` when a dust-free twin is missing (the caller
    decides, as :func:`load_star_mask` does) or the mask is off.

    Exactly what loading each plane with :func:`_load_tng_plane`, stacking
    them and calling :func:`load_star_mask` gives, checked in the same order,
    with each pixel converted once.
    """
    planes = [
        _tng_plane_array(*future.result(), path)
        for future, path in zip(read.primaries, read.paths, strict=True)
    ]
    # The planes' grids are checked as TNGSurfaceBrightnessImage.stack does.
    first_shape, scale = planes[0][0].shape, planes[0][1]
    for index, (array, plane_scale) in enumerate(planes):
        if array.shape != first_shape:
            raise ValueError(
                f"image {index} has spatial shape {array.shape}, "
                f"expected {first_shape}"
            )
        if plane_scale != scale:
            raise ValueError(
                f"image {index} has pixel scale {plane_scale!r} pc, "
                f"expected {scale!r} pc"
            )
    stacked = np.empty((*first_shape, len(planes)), dtype=np.float32)
    for index, (array, _) in enumerate(planes):
        stacked[..., index] = array
    _zero_non_finite(stacked)
    image = TNGSurfaceBrightnessImage.adopt(stacked, band_names, scale)
    if read.lit is None:
        return image, None
    return image, _union_mask(read.twins, [future.result() for future in read.lit])


def load_view_frames(
    dusty_paths: Sequence[str | PathLike[str]],
    band_names: Sequence[str],
) -> tuple[TNGSurfaceBrightnessImage, np.ndarray | None]:
    """:func:`finish_view_read` of a :func:`start_view_read` on a pool of its own."""
    with ThreadPoolExecutor(max_workers=2 * len(dusty_paths)) as pool:
        return finish_view_read(start_view_read(dusty_paths, pool), band_names)


def _zero_non_finite(values: np.ndarray) -> None:
    """In place: what ``np.nan_to_num(values, nan=0, posinf=0, neginf=0)``
    returns, without its three full passes when everything is finite."""
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = 0.0


#: The glow's fade reaches exactly zero this many fade lengths from the
#: outline. A plain exp(-d / FADE), computed only that far out (exp(-14) ~
#: 1e-6 of the glow), stopped on the straight edges of the computed region: a
#: faint rectangle around bright galaxies at knee 1e-3 (434357 in validate 18,
#: 2026-10-20).
GLOW_FADE_REACH = 14.0


def _glow_fade(distance_px: np.ndarray, fade_px: float) -> np.ndarray:
    """exp(-d / fade), lowered to reach zero at ``GLOW_FADE_REACH`` fade
    lengths and rescaled to stay one on the outline."""
    floor = np.exp(-GLOW_FADE_REACH)
    fade = (np.exp(-np.asarray(distance_px, dtype=np.float64) / fade_px) - floor) / (1.0 - floor)
    return np.maximum(fade, 0.0).astype(np.float32)


def star_light_with_glow(
    data: np.ndarray,
    stars: np.ndarray,
    *,
    pixel_scale_pc: float,
) -> np.ndarray:
    """Dusty light on the stars, a smooth glow of scattered light around them.

    Where a dust-free twin has light (``stars``) the dusty frame is kept as it
    is: attenuation, dust lanes and the light scattered onto the galaxy. Off
    the stars SKIRT's scattered light is drawn on an octree of dust cubes
    (flat facets, straight edges, shadow rays; Baes et al. 2024), so only a
    smooth version of it is kept, close to the galaxy: the off-star light
    blurred by ``Config.Tng.SCATTER_GLOW_BLUR_PC`` (normalised over the
    off-star pixels, so no starlight leaks into it), times exp(-d / FADE)
    (reaching zero at ``GLOW_FADE_REACH`` fade lengths) with d the distance
    from the galaxy's outline (the star mask smoothed by
    ``STAR_BODY_SMOOTH_PC``, cut at one half). Measured from the outline, not
    from each star, so lone halo stars grow no glow of their own, while gaps
    between the stars inside the outline get the full glow.
    """
    data = np.asarray(data, dtype=np.float32)
    stars = np.asarray(stars, dtype=bool)
    blur_px = Config.Tng.SCATTER_GLOW_BLUR_PC / pixel_scale_pc
    body_px = Config.Tng.STAR_BODY_SMOOTH_PC / pixel_scale_pc
    fade_px = Config.Tng.SCATTER_GLOW_FADE_PC / pixel_scale_pc
    out = data * stars[..., None]
    body = _star_body(stars, body_px)
    rows = np.flatnonzero(body.any(axis=1))
    if rows.size == 0:
        return out
    cols = np.flatnonzero(body.any(axis=0))
    # Only where the fade is not zero plus the blur's reach: generation loads
    # a donor for nearly every galaxy it draws.
    margin = int(np.ceil(GLOW_FADE_REACH * fade_px + 3.0 * blur_px))
    ys = slice(max(rows[0] - margin, 0), min(rows[-1] + margin + 1, data.shape[0]))
    xs = slice(max(cols[0] - margin, 0), min(cols[-1] + margin + 1, data.shape[1]))
    sub_stars, sub_body = stars[ys, xs], body[ys, xs]
    fade = _glow_fade(distance_to_mask(sub_body), fade_px)
    off = (~sub_stars).astype(np.float32)
    # The 3 kpc blur is smooth: computed on a coarser grid, then interpolated.
    factor = max(1, int(blur_px // 10.0))
    coverage = np.maximum(_coarse_blur(off, blur_px, factor), 1e-6)
    if not np.isfinite(data[ys, xs]).all():     # loaded frames never are
        for band in range(data.shape[-1]):
            plane = data[ys, xs, band]
            glow = _coarse_blur(np.nan_to_num(plane) * off, blur_px, factor) / coverage
            out[ys, xs, band] = np.where(sub_stars, plane, glow * fade)
        return out
    # The same float32 products and quotients as the numpy expressions above,
    # fused into two passes over the four bands.
    light = _off_star_light(data, ys.start, xs.start, off)
    glows = np.stack([_coarse_blur(light[band], blur_px, factor)
                      for band in range(data.shape[-1])])
    _place_glow(out, ys.start, xs.start, np.ascontiguousarray(sub_stars), glows, coverage, fade)
    return out


@njit(nogil=True, cache=False)
def _off_star_light(data, row0, col0, off):
    """``data[rows, cols, band] * off`` per band (band-major, contiguous)."""
    height, width = off.shape
    light = np.empty((data.shape[2], height, width), np.float32)
    for y in range(height):
        for x in range(width):
            for band in range(data.shape[2]):
                light[band, y, x] = data[row0 + y, col0 + x, band] * off[y, x]
    return light


@njit(nogil=True, cache=False)
def _place_glow(out, row0, col0, stars, glows, coverage, fade):
    """Off the stars, ``out = glow / coverage * fade`` (on them ``out``
    already holds the dusty light)."""
    height, width = stars.shape
    for y in range(height):
        for x in range(width):
            if not stars[y, x]:
                for band in range(glows.shape[0]):
                    out[row0 + y, col0 + x, band] = (
                        glows[band, y, x] / coverage[y, x]) * fade[y, x]


def compile_glow_kernels() -> None:
    """Run :func:`star_light_with_glow` once on a small read-only frame, as a
    loaded donor is, which compiles its numba kernels for the types every
    donor uses (a generation parent does this before forking its workers)."""
    frame = np.ones((400, 400, 4), np.float32)
    frame.setflags(write=False)
    stars = np.zeros((400, 400), dtype=bool)
    stars[170:230, 170:230] = True     # a body survives the 1 kpc blur
    star_light_with_glow(frame, stars, pixel_scale_pc=100.0)


def distance_to_mask(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (float64 pixels) from every pixel to the nearest
    ``True`` pixel of ``mask``; ``inf`` everywhere when there is none.

    Bit for bit ``scipy.ndimage.distance_transform_edt(~mask)``: both take the
    square root of the exact integer squared distance to the nearest mask
    pixel, which is unique even where the nearest pixel is not. Computed by
    Meijster et al.'s linear-time algorithm in integers, ~5x faster than
    SciPy's feature transform (which also builds the nearest-pixel indices).
    """
    mask = np.ascontiguousarray(mask, dtype=bool)
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return _distance_to_mask(mask)


@njit(nogil=True, cache=False)
def _distance_to_mask(mask):
    height, width = mask.shape
    infinity = height + width          # exceeds every in-image distance
    column = np.empty((height, width), np.int64)
    for x in range(width):              # phase 1: distance down each column
        d = infinity
        for y in range(height):
            d = 0 if mask[y, x] else min(d + 1, infinity)
            column[y, x] = d
        d = infinity
        for y in range(height - 1, -1, -1):
            d = 0 if mask[y, x] else min(d + 1, infinity)
            if d < column[y, x]:
                column[y, x] = d
    out = np.empty((height, width), np.float64)
    s = np.empty(width, np.int64)       # phase 2: lower envelope along rows
    t = np.empty(width, np.int64)
    for y in range(height):
        g = column[y]
        q = 0
        s[0] = 0
        t[0] = 0
        for x in range(1, width):
            while q >= 0 and ((t[q] - s[q]) ** 2 + g[s[q]] ** 2
                              > (t[q] - x) ** 2 + g[x] ** 2):
                q -= 1
            if q < 0:
                q = 0
                s[0] = x
            else:
                w = 1 + (x * x - s[q] * s[q] + g[x] ** 2 - g[s[q]] ** 2) // (2 * (x - s[q]))
                if w < width:
                    q += 1
                    s[q] = x
                    t[q] = w
        for x in range(width - 1, -1, -1):
            # The exact integer square distance; IEEE sqrt is correctly
            # rounded, as SciPy's np.sqrt of the same float64 is.
            out[y, x] = math.sqrt(float((x - s[q]) ** 2 + g[s[q]] ** 2))
            if x == t[q]:
                q -= 1
    return out


def _gaussian_blur(image: np.ndarray, sigma_px: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_px, sigmaY=sigma_px, borderType=cv2.BORDER_CONSTANT)


def _star_body(stars: np.ndarray, sigma_px: float) -> np.ndarray:
    """The galaxy's outline: the star mask blurred by ``sigma_px``, cut at one
    half. The blur runs on the mask's bounding box padded past the kernel's
    reach (OpenCV truncates a float Gaussian at 4 sigma), the same values a
    full-frame blur gives there; everywhere else that blur is exactly zero."""
    body = np.zeros(stars.shape, dtype=bool)
    rows = np.flatnonzero(stars.any(axis=1))
    if rows.size == 0:
        return body
    cols = np.flatnonzero(stars.any(axis=0))
    pad = int(np.ceil(4.0 * sigma_px)) + 2
    ys = slice(max(int(rows[0]) - pad, 0), min(int(rows[-1]) + pad + 1, stars.shape[0]))
    xs = slice(max(int(cols[0]) - pad, 0), min(int(cols[-1]) + pad + 1, stars.shape[1]))
    body[ys, xs] = _gaussian_blur(stars[ys, xs].astype(np.float32), sigma_px) > 0.5
    return body


def _coarse_blur(image: np.ndarray, sigma_px: float, factor: int) -> np.ndarray:
    """Gaussian blur on a grid ``factor`` times coarser (area-averaged), then
    bilinearly back to full size; the image's total is kept."""
    if factor <= 1:
        return _gaussian_blur(image, sigma_px)
    height, width = image.shape
    pad = ((0, -height % factor), (0, -width % factor))
    padded_shape = (height + pad[0][1], width + pad[1][1])
    # numpy sums a block row by row when the blocks do not span the width
    # (it merges the two block axes otherwise) and sums 8 or more terms
    # pairwise: only that case is ours.
    if factor < 8 and padded_shape[1] > factor and image.dtype == np.float32:
        small = _block_mean_f32(np.ascontiguousarray(image), factor)
    else:
        padded = np.pad(image, pad)
        small = padded.reshape(padded.shape[0] // factor, factor,
                               padded.shape[1] // factor, factor).mean(axis=(1, 3))
    small = _gaussian_blur(small.astype(np.float32), sigma_px / factor)
    big = cv2.resize(small, (padded_shape[1], padded_shape[0]), interpolation=cv2.INTER_LINEAR)
    return big[:height, :width]


@njit(nogil=True, cache=False)
def _block_mean_f32(image, factor):
    """``np.pad(image, to a multiple of factor).reshape(..).mean(axis=(1, 3))``
    for float32, ``factor < 8`` and at least two blocks across, summed in
    numpy's order: each block row from zero along the row, the rows into a
    total from zero, then divided by the count."""
    height, width = image.shape
    rows = (height + factor - 1) // factor
    cols = (width + factor - 1) // factor
    count = np.float32(factor * factor)
    out = np.empty((rows, cols), np.float32)
    for a in range(rows):
        for c in range(cols):
            total = np.float32(0.0)
            for b in range(factor):
                y = a * factor + b
                partial = np.float32(0.0)
                for d in range(factor):
                    x = c * factor + d
                    partial += image[y, x] if y < height and x < width else np.float32(0.0)
                total += partial
            out[a, c] = total / count
    return out


def _load_tng_plane(
    path: str | PathLike[str],
    band_name: str,
) -> TNGSurfaceBrightnessImage:
    """Load one SKIRT surface-brightness FITS plane on its physical grid.

    A supported SKIRT product is a two-dimensional primary image whose
    ``BUNIT`` is exactly MJy/sr and whose two FITS axes describe the same
    physical pixel scale through ``CDELT1/2`` and ``CUNIT1/2``.  The returned
    image is always native-endian float32 in ``(height, width, 1)`` layout.
    Non-finite input samples are replaced with zero before construction.
    """
    array, pixel_scale_pc = _tng_plane_array(*_read_primary(path), path)
    values = np.array(array, dtype=np.float32)[..., None]
    _zero_non_finite(values)
    return TNGSurfaceBrightnessImage.adopt(values, (band_name,), pixel_scale_pc)


def _read_primary(path: str | PathLike[str]) -> tuple[np.ndarray | None, fits.Header]:
    """A FITS file's primary image and header, read in one go (no memory map:
    one large read is what network storage serves fastest)."""
    with fits.open(path, memmap=False) as hdul:
        primary = hdul[0]
        if not isinstance(primary, fits.PrimaryHDU):
            raise ValueError(f"first FITS extension is not a primary HDU: {path}")
        return primary.data, primary.header.copy()


def _tng_plane_array(
    data: np.ndarray | None,
    header: fits.Header,
    path: str | PathLike[str],
) -> tuple[np.ndarray, float]:
    """Check one SKIRT plane (see :func:`_load_tng_plane`); returns its pixels
    as stored and its physical pixel scale in pc."""
    if data is None:
        raise ValueError(f"empty primary HDU: {path}")
    if np.ndim(data) != 2:
        raise ValueError(
            f"SKIRT primary image must be two-dimensional, got "
            f"shape {np.shape(data)}: {path}"
        )

    bunit = header.get("BUNIT")
    if bunit is None:
        raise ValueError(f"SKIRT FITS is missing BUNIT: {path}")
    try:
        parsed_bunit = u.Unit(str(bunit).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid SKIRT BUNIT {bunit!r}: {path}") from exc
    expected_bunit = u.MJy / u.sr
    if parsed_bunit != expected_bunit:
        raise ValueError(
            "SKIRT BUNIT must be 'MJy/sr', "
            f"got {bunit!r}: {path}"
        )

    return data, _physical_pixel_scale_pc(header, path)


def _physical_pixel_scale_pc(
    header: fits.Header,
    path: str | PathLike[str],
) -> float:
    scales_pc: list[float] = []
    for axis in (1, 2):
        scale_key = f"CDELT{axis}"
        unit_key = f"CUNIT{axis}"
        if scale_key not in header or unit_key not in header:
            raise ValueError(
                f"SKIRT FITS is missing {scale_key} or {unit_key}: {path}"
            )
        try:
            axis_unit = u.Unit(str(header[unit_key]).strip())
            scale_value = float(str(header[scale_key]))
            unit_to_pc = cast(float, axis_unit.to(u.pc))
            axis_scale_pc = abs(scale_value * unit_to_pc)
        except (TypeError, ValueError, u.UnitConversionError) as exc:
            raise ValueError(
                f"SKIRT {scale_key}/{unit_key} must describe a physical "
                f"pixel scale: {path}"
            ) from exc
        if not np.isfinite(axis_scale_pc) or axis_scale_pc <= 0.0:
            raise ValueError(
                f"SKIRT {scale_key} must be finite and non-zero, "
                f"got {header[scale_key]!r}: {path}"
            )
        scales_pc.append(float(axis_scale_pc))
    if not np.isclose(scales_pc[0], scales_pc[1], rtol=1e-9, atol=0.0):
        raise ValueError(
            "SKIRT physical pixels must be square; "
            f"got {scales_pc[0]:g} and {scales_pc[1]:g} pc: {path}"
        )
    return scales_pc[0]


def _block_mean(
    image: TNGSurfaceBrightnessImage,
    factor: int,
) -> TNGSurfaceBrightnessImage:
    """Average each ``factor x factor`` block without changing surface brightness.

    ``factor == 1`` returns an independent image.  This is appropriate for
    intensive MJy/sr values; it is not a flux-conserving sum rebin.  The output
    physical pixel scale is multiplied by ``factor``.
    """
    _require_tng_image(image)
    rebinned = _block_mean_array(image.as_array(), factor)
    return image.adopting(
        rebinned,
        pixel_scale_pc=image.pixel_scale_pc * int(factor),
    )


def _block_mean_array(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 3:
        raise ValueError(f"expected an HWC channel cube, got shape {a.shape}")
    if factor == 1:
        return a.copy()
    height, width, channels = a.shape
    if factor > height or factor > width:
        raise ValueError(
            f"factor {factor} is larger than spatial shape {(height, width)}"
        )
    if height % factor != 0 or width % factor != 0:
        height = (height // factor) * factor
        width = (width // factor) * factor
        a = a[:height, :width, :]
    new_height, new_width = height // factor, width // factor
    result = a.reshape(
        new_height, factor, new_width, factor, channels,
    ).mean(axis=(1, 3))
    return np.asarray(result, dtype=np.float32)


def _downsample_surface_brightness(
    image: TNGSurfaceBrightnessImage,
    scale: float,
) -> TNGSurfaceBrightnessImage:
    """Area-resample a surface-brightness image without enlarging it.

    ``scale`` is the output-to-input image side ratio.  It must be at most one:
    the SKIRT atlas may be downsampled to a smaller apparent source, but
    interpolation must never invent spatial detail by enlarging a donor.  Area
    resampling makes each output sample the mean surface brightness over its
    input footprint.  The helper deliberately does not apply a pixel-area flux
    correction: callers decide the final integrated flux separately.

    The output remains MJy/sr, while its physical pixel scale is divided by
    ``scale`` so the grid continues to describe the same physical scene.
    """
    _require_tng_image(image)
    downsampled = _downsample_surface_brightness_array(image.as_array(), scale)
    return image.adopting(
        downsampled,
        pixel_scale_pc=image.pixel_scale_pc / float(scale),
    )


def _downsample_surface_brightness_array(
    arr: np.ndarray,
    scale: float,
) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale!r}")
    if scale > 1.0:
        raise ValueError(
            f"surface-brightness stamps cannot be enlarged (scale={scale!r})"
        )
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 3:
        raise ValueError(f"expected an HWC channel cube, got shape {a.shape}")
    if scale == 1.0:
        return a.copy()
    height, width = a.shape[:2]
    out_height = max(1, int(round(height * float(scale))))
    out_width = max(1, int(round(width * float(scale))))
    out = cv2.resize(
        a, (out_width, out_height), interpolation=cv2.INTER_AREA,
    )
    # OpenCV drops a singleton channel axis; restore canonical HWC layout.
    if out.ndim == 2:
        out = out[..., None]
    np.maximum(out, 0.0, out=out)
    return np.asarray(out, dtype=np.float32)


#: The soft circular window every TNG source is multiplied by before it is
#: rotated: weight 1 inside SOFT_WINDOW_INNER_FRACTION of the rim radius,
#: then an exponential taper - a straight line in log surface brightness,
#: SOFT_WINDOW_DECADES down at the rim, lowered to end there at exactly 0 -
#: and 0 beyond. The rim sits a margin inside the frame's inscribed circle —
#: at least SOFT_WINDOW_RIM_MARGIN_PX (the cubic kernel's reach), more when a
#: later shrink averages many native pixels into one. Light that runs into
#: the edge of the SKIRT frame or a crop square otherwise ends on a hard
#: straight step (square, or octagonal once rotated), the one faint truth
#: feature with a preferred axis (2026-10-18 audit). A disc inside the
#: inscribed circle is unchanged by rotation about the frame centre. Until
#: 2026-10-20 the taper was a cos² from 0.7 of the rim: the stellar halos
#: that reach the frame's edge (45 % of the views) kept ~0.1-1 e-/px there and
#: dropped to 0 within half an R_e, a crisp circle at knees <= 0.01 e-.
SOFT_WINDOW_INNER_FRACTION = 0.5
SOFT_WINDOW_DECADES = 6.0
SOFT_WINDOW_RIM_MARGIN_PX = 2.0
#: The margin never takes more than this fraction of the frame's half-width,
#: so a frame shrunk to a few output pixels still keeps a window.
SOFT_WINDOW_MAX_MARGIN_FRACTION = 0.3


def _soft_circular_window(
    image: TNGSurfaceBrightnessImage,
    margin_px: float = SOFT_WINDOW_RIM_MARGIN_PX,
) -> TNGSurfaceBrightnessImage:
    """``image`` times :func:`_soft_circular_window_array`'s weight."""
    _require_tng_image(image)
    return image.adopting(
        _soft_circular_window_array(image.as_array(), margin_px)
    )


def _soft_circular_window_array(
    arr: np.ndarray,
    margin_px: float = SOFT_WINDOW_RIM_MARGIN_PX,
) -> np.ndarray:
    """Every band of an HWC cube times the soft circular window (centred on
    the frame centre the rotation uses, ``((h-1)/2, (w-1)/2)``), its rim
    ``margin_px`` inside the inscribed circle."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 3:
        raise ValueError(f"expected an HWC channel cube, got shape {a.shape}")
    height, width = a.shape[:2]
    half = min(height, width) / 2.0
    margin = min(max(float(margin_px), SOFT_WINDOW_RIM_MARGIN_PX),
                 SOFT_WINDOW_MAX_MARGIN_FRACTION * half)
    if half - margin <= 1.0:
        return a                        # a degenerate frame: nothing to taper
    return a * _soft_window_weight(height, width, round(margin, 3))[..., None]


@lru_cache(maxsize=8)
def _soft_window_weight(height: int, width: int, margin: float) -> np.ndarray:
    """The (read-only, cached) window weight of one frame shape and margin;
    full-frame crops of large donors repeat the same shape."""
    rim = min(height, width) / 2.0 - margin
    inner = SOFT_WINDOW_INNER_FRACTION * rim
    rows, columns = np.ogrid[:height, :width]
    radius = np.hypot(np.float32(rows - (height - 1) / 2.0),
                      np.float32(columns - (width - 1) / 2.0))
    taper = np.clip((radius - inner) / (rim - inner), 0.0, 1.0).astype(np.float64)
    floor = 10.0 ** -SOFT_WINDOW_DECADES
    weight = ((floor ** taper - floor) / (1.0 - floor)).astype(np.float32)
    weight[taper <= 0.0] = 1.0
    weight[taper >= 1.0] = 0.0
    weight.setflags(write=False)
    return weight


def _rotate_surface_brightness(
    image: TNGSurfaceBrightnessImage,
    angle_deg: float,
) -> TNGSurfaceBrightnessImage:
    """Rotate in place-sized coordinates and clip interpolation undershoot.

    The returned image has the input shape.  Values outside the frame are zero;
    negative cubic overshoot is clipped because surface brightness is
    non-negative.  Whether interpolation is scientifically acceptable at the
    requested resolution is a policy decision for the caller.
    """
    _require_tng_image(image)
    return image.adopting(
        _rotate_arbitrary_array(image.as_array(), angle_deg)
    )


def _rotate_arbitrary_array(
    arr: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    if not np.isfinite(angle_deg):
        raise ValueError(f"angle_deg must be finite, got {angle_deg!r}")
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 3:
        raise ValueError(f"expected an HWC channel cube, got shape {a.shape}")
    height, width = a.shape[:2]
    centre = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, float(angle_deg), 1.0)
    out = cv2.warpAffine(
        a,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    if out.ndim == 2:
        out = out[..., None]
    np.maximum(out, 0.0, out=out)
    return out.astype(np.float32, copy=False)


def _require_tng_image(image: TNGSurfaceBrightnessImage) -> None:
    if not isinstance(image, TNGSurfaceBrightnessImage):
        raise TypeError(
            "expected TNGSurfaceBrightnessImage, "
            f"got {type(image).__name__}"
        )


# Radius grids are expensive enough to cache for repeated native atlas shapes,
# but continuous TNG scaling creates nearly one unique output shape per trial.
# An unbounded shape -> grid dictionary therefore retained tens of MB per field
# in every generation worker.  Admit a shape only after its second use and cap
# retained arrays by bytes, not entry count (one large grid can dwarf hundreds
# of small ones).
_RADIUS_INT_GRID_MAX_BYTES = 64 * 1024 * 1024
_RADIUS_INT_GRID_SEEN_MAX_SHAPES = 4096
_RADIUS_INT_GRID: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
_RADIUS_INT_GRID_SEEN: OrderedDict[tuple[int, int], None] = OrderedDict()
_RADIUS_INT_GRID_BYTES = 0


def _radius_int_grid(shape: tuple[int, int]) -> np.ndarray:
    """Integer-radius grid with bounded reuse for recurring image shapes.

    First-use shapes are returned without caching.  A second request admits
    the grid to a byte-limited LRU, which keeps the repeatedly used 1600-square
    native atlas grid hot while one-off continuously scaled stamp dimensions
    are released normally.  Grids larger than the complete cache budget are
    never retained.
    """
    global _RADIUS_INT_GRID_BYTES
    key = (int(shape[0]), int(shape[1]))
    grid = _RADIUS_INT_GRID.pop(key, None)
    if grid is not None:
        _RADIUS_INT_GRID[key] = grid
        return grid

    height, width = key
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    yy = np.arange(height, dtype=np.float64)[:, None] - cy
    xx = np.arange(width, dtype=np.float64)[None, :] - cx
    grid = np.sqrt(yy * yy + xx * xx).astype(np.int64)

    if grid.nbytes > _RADIUS_INT_GRID_MAX_BYTES:
        return grid
    if key not in _RADIUS_INT_GRID_SEEN:
        _RADIUS_INT_GRID_SEEN[key] = None
        if len(_RADIUS_INT_GRID_SEEN) > _RADIUS_INT_GRID_SEEN_MAX_SHAPES:
            _RADIUS_INT_GRID_SEEN.popitem(last=False)
        return grid

    _RADIUS_INT_GRID_SEEN.pop(key, None)
    while (
        _RADIUS_INT_GRID
        and _RADIUS_INT_GRID_BYTES + grid.nbytes > _RADIUS_INT_GRID_MAX_BYTES
    ):
        _, evicted = _RADIUS_INT_GRID.popitem(last=False)
        _RADIUS_INT_GRID_BYTES -= int(evicted.nbytes)
    _RADIUS_INT_GRID[key] = grid
    _RADIUS_INT_GRID_BYTES += int(grid.nbytes)
    return grid


def _measure_halflight_radius_px(
    image: TNGSurfaceBrightnessImage,
    *,
    band: str | None = None,
    frac: float = 0.5,
) -> float:
    """Radius in pixels enclosing ``frac`` of positive flux about the centre.

    SKIRT atlas products centre their target galaxies geometrically.  For other
    products, callers should centre the source before using this helper.
    ``band`` may be omitted only for a single-channel image.  Empty or
    non-positive images return NaN.
    """
    _require_tng_image(image)
    return _measure_halflight_radius_px_array(image.plane(band), frac=frac)


def _measure_halflight_radius_px_array(
    frame: np.ndarray,
    *,
    frac: float,
) -> float:
    fraction = float(frac)
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError(
            f"frac must be finite and in (0, 1], got {frac!r}"
        )
    a = np.asarray(frame, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected one 2-D image plane, got shape {a.shape}")
    a = np.where(np.isfinite(a) & (a > 0.0), a, 0.0)
    total = float(a.sum())
    if total <= 0.0:
        return float("nan")
    radii = _radius_int_grid(a.shape)
    profile = np.bincount(radii.ravel(), weights=a.ravel())
    cumulative = np.cumsum(profile)
    target = fraction * total
    index = min(int(np.searchsorted(cumulative, target)), cumulative.size - 1)
    if index <= 0:
        return 0.5
    lower, upper = cumulative[index - 1], cumulative[index]
    subpixel = (target - lower) / (upper - lower) if upper > lower else 0.0
    return float(index - 1 + subpixel)


def _centered_rotation_crop_slices(
    image: TNGSurfaceBrightnessImage,
    rebin: int,
    *,
    band: str | None = None,
    enclosed_fraction: float,
    padding: float,
) -> tuple[slice, slice]:
    """Centred square crop large enough to rotate the enclosed source light.

    The half-side is the requested curve-of-growth radius times ``sqrt(2)`` and
    ``padding``.  Its side is snapped to a multiple of ``rebin`` for subsequent
    block averaging.
    """
    _require_tng_image(image)
    plane = image.plane(band)
    return _centered_rotation_crop_slices_array(
        plane,
        rebin,
        enclosed_fraction=enclosed_fraction,
        padding=padding,
    )


def _centered_rotation_crop_slices_array(
    frame: np.ndarray,
    rebin: int,
    *,
    enclosed_fraction: float,
    padding: float,
) -> tuple[slice, slice]:
    a = np.asarray(frame)
    if a.ndim != 2:
        raise ValueError(f"expected one 2-D image plane, got shape {a.shape}")
    height, width = a.shape
    step = int(rebin)
    if step < 1:
        raise ValueError(f"rebin must be >= 1, got {rebin}")
    if step > height or step > width:
        raise ValueError(
            f"rebin {step} is larger than spatial shape {(height, width)}"
        )
    pad = float(padding)
    if not np.isfinite(pad) or pad <= 0.0:
        raise ValueError(f"padding must be finite and positive, got {padding!r}")
    radius = _measure_halflight_radius_px_array(
        a,
        frac=enclosed_fraction,
    )
    half = (
        int(np.ceil(radius * np.sqrt(2.0) * pad))
        if np.isfinite(radius) and radius > 0.0
        else min(height, width) // 2
    )
    side = min(2 * half, height, width)
    side = max(step, side - side % step)
    cy, cx = height // 2, width // 2
    half_side = side // 2
    return (
        slice(cy - half_side, cy - half_side + side),
        slice(cx - half_side, cx - half_side + side),
    )


def _stochastic_round_factor(
    factor: float,
    rng: np.random.Generator | None,
) -> int:
    """Round a continuous scale to an integer >= 1, unbiased in expectation."""
    lower = int(np.floor(factor))
    remainder = factor - lower
    if rng is not None and remainder > 0.0:
        return max(1, lower + (1 if rng.random() < remainder else 0))
    return max(1, int(round(factor)))
