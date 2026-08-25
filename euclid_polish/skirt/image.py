"""Instrument-independent image mechanics for SKIRT FITS products.

SKIRT broadband images are commonly stored as surface brightness in MJy/sr.
The helpers here load those products into validated image cubes, preserve their
intensive units while rebinning, provide controlled rotation augmentation,
measure centred curves of growth, and composite prepared stamps onto a canvas.
They do not assign a distance or redshift, or convert into detector-specific
units.
"""

from __future__ import annotations

from collections import OrderedDict
from os import PathLike
from typing import cast

import cv2
import numpy as np
from astropy import units as u
from astropy.io import fits

from euclid_polish.image.cube import ImageCube, PhysicalGrid, PixelUnit

# Generation already parallelises over one process per allocated CPU.  OpenCV
# otherwise creates a thread pool in every worker and oversubscribes the outer
# process pool (for example, 10 x 10 runnable threads on a 10-CPU job).
cv2.setNumThreads(1)


def load_skirt_image(
    path: str | PathLike[str],
    band_name: str,
) -> ImageCube:
    """Load one SKIRT surface-brightness FITS plane on its physical grid.

    A supported SKIRT product is a two-dimensional primary image whose
    ``BUNIT`` is exactly MJy/sr and whose two FITS axes describe the same
    physical pixel scale through ``CDELT1/2`` and ``CUNIT1/2``.  The returned
    cube is always native-endian float32 in ``(height, width, 1)`` layout.
    Non-finite input samples are replaced with zero before construction.
    """
    with fits.open(path) as hdul:
        primary = hdul[0]
        if not isinstance(primary, fits.PrimaryHDU):
            raise ValueError(f"first FITS extension is not a primary HDU: {path}")
        data = primary.data
        header = primary.header.copy()
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
            f"SKIRT BUNIT must be {PixelUnit.MJY_PER_SR.value!r}, "
            f"got {bunit!r}: {path}"
        )

    pixel_scale_pc = _physical_pixel_scale_pc(header, path)
    arr = np.asarray(data, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return ImageCube(
        data=arr[..., None],
        bands=(band_name,),
        unit=PixelUnit.MJY_PER_SR,
        grid=PhysicalGrid(pixel_scale_pc=pixel_scale_pc),
    )


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


def block_mean(
    image: ImageCube,
    factor: int,
) -> ImageCube:
    """Average each ``factor x factor`` block without changing surface brightness.

    ``factor == 1`` returns an independent cube.  This is appropriate for
    intensive MJy/sr values; it is not a flux-conserving sum rebin.  The output
    physical pixel scale is multiplied by ``factor``.
    """
    physical_grid = _require_skirt_image(image)
    rebinned = _block_mean_array(image.as_array(), factor)
    return image.with_data(
        rebinned,
        grid=PhysicalGrid(
            pixel_scale_pc=physical_grid.pixel_scale_pc * int(factor)
        ),
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


def downsample_surface_brightness(
    image: ImageCube,
    scale: float,
) -> ImageCube:
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
    physical_grid = _require_skirt_image(image)
    downsampled = _downsample_surface_brightness_array(image.as_array(), scale)
    return image.with_data(
        downsampled,
        grid=PhysicalGrid(
            pixel_scale_pc=physical_grid.pixel_scale_pc / float(scale)
        ),
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


def rotate_surface_brightness(
    image: ImageCube,
    angle_deg: float,
) -> ImageCube:
    """Rotate in place-sized coordinates and clip interpolation undershoot.

    The returned image has the input shape.  Values outside the frame are zero;
    negative cubic overshoot is clipped because surface brightness is
    non-negative.  Whether interpolation is scientifically acceptable at the
    requested resolution is a policy decision for the caller.
    """
    _require_skirt_image(image)
    return image.with_data(
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
    return out.astype(np.float32)


def _require_skirt_image(image: ImageCube) -> PhysicalGrid:
    if not isinstance(image, ImageCube):
        raise TypeError(
            f"expected ImageCube, got {type(image).__name__}"
        )
    if image.unit is not PixelUnit.MJY_PER_SR:
        raise ValueError(
            "SKIRT image operations require MJy/sr pixels, "
            f"got {image.unit.value!r}"
        )
    if not isinstance(image.grid, PhysicalGrid):
        raise ValueError(
            "SKIRT image operations require a physical parsec grid, "
            f"got {type(image.grid).__name__}"
        )
    return image.grid


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


def radius_int_grid(shape: tuple[int, int]) -> np.ndarray:
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


def measure_halflight_radius_px(
    image: ImageCube,
    *,
    band: str | None = None,
    frac: float = 0.5,
) -> float:
    """Radius in pixels enclosing ``frac`` of positive flux about the centre.

    SKIRT atlas products centre their target galaxies geometrically.  For other
    products, callers should centre the source before using this helper.
    ``band`` may be omitted only for a single-channel cube.  Empty or
    non-positive images return NaN.
    """
    _require_skirt_image(image)
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
    radii = radius_int_grid(a.shape)
    profile = np.bincount(radii.ravel(), weights=a.ravel())
    cumulative = np.cumsum(profile)
    target = fraction * total
    index = min(int(np.searchsorted(cumulative, target)), cumulative.size - 1)
    if index <= 0:
        return 0.5
    lower, upper = cumulative[index - 1], cumulative[index]
    subpixel = (target - lower) / (upper - lower) if upper > lower else 0.0
    return float(index - 1 + subpixel)


def centered_rotation_crop_slices(
    image: ImageCube,
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
    _require_skirt_image(image)
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


def stochastic_round_factor(
    factor: float,
    rng: np.random.Generator | None,
) -> int:
    """Round a continuous scale to an integer >= 1, unbiased in expectation."""
    lower = int(np.floor(factor))
    remainder = factor - lower
    if rng is not None and remainder > 0.0:
        return max(1, lower + (1 if rng.random() < remainder else 0))
    return max(1, int(round(factor)))


def composite_stamp(
    canvas: np.ndarray,
    stamp: np.ndarray,
    x0: float,
    y0: float,
) -> None:
    """Add a centred ``(height, width, channels)`` stamp, clipped to the canvas."""
    height, width = canvas.shape[:2]
    stamp_height, stamp_width = stamp.shape[:2]
    row0 = int(round(y0)) - stamp_height // 2
    col0 = int(round(x0)) - stamp_width // 2
    canvas_row_lo, canvas_row_hi = max(0, row0), min(height, row0 + stamp_height)
    canvas_col_lo, canvas_col_hi = max(0, col0), min(width, col0 + stamp_width)
    if canvas_row_lo >= canvas_row_hi or canvas_col_lo >= canvas_col_hi:
        return
    stamp_row_lo = canvas_row_lo - row0
    stamp_col_lo = canvas_col_lo - col0
    canvas[canvas_row_lo:canvas_row_hi, canvas_col_lo:canvas_col_hi, :] += stamp[
        stamp_row_lo : stamp_row_lo + (canvas_row_hi - canvas_row_lo),
        stamp_col_lo : stamp_col_lo + (canvas_col_hi - canvas_col_lo),
        :,
    ]
