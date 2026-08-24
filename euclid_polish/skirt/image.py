"""Instrument-independent image mechanics for SKIRT FITS products.

SKIRT broadband images are commonly stored as surface brightness in MJy/sr.
The helpers here load those arrays, preserve their intensive units while
rebinning, provide controlled rotation augmentation, measure centred curves of
growth, and composite prepared stamps onto a canvas.  They do not select bands,
assign a distance or redshift, or convert into detector-specific units.
"""

from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np
from astropy.io import fits

# Generation already parallelises over one process per allocated CPU.  OpenCV
# otherwise creates a thread pool in every worker and oversubscribes the outer
# process pool (for example, 10 x 10 runnable threads on a 10-CPU job).
cv2.setNumThreads(1)


def load_skirt_frame(path: str) -> np.ndarray:
    """Read a SKIRT FITS primary image as native-endian float32.

    Non-finite pixels are replaced with zero.  The caller remains responsible
    for checking the FITS ``BUNIT`` and interpreting the image calibration.
    """
    with fits.open(path) as hdul:
        data = hdul[0].data
    if data is None:
        raise ValueError(f"empty primary HDU: {path}")
    arr = np.asarray(data, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def block_mean(
    arr: np.ndarray,
    factor: int,
) -> np.ndarray:
    """Average each ``factor x factor`` block without changing surface brightness.

    ``factor == 1`` returns a copy.  This is appropriate for intensive image
    units such as MJy/sr; it is not a flux-conserving sum rebin.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    a = np.asarray(arr, dtype=np.float32)
    if factor == 1:
        return a.copy()
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {a.shape}")
    height, width = a.shape
    if height % factor != 0 or width % factor != 0:
        height = (height // factor) * factor
        width = (width // factor) * factor
        a = a[:height, :width]
    new_height, new_width = height // factor, width // factor
    return a.reshape(new_height, factor, new_width, factor).mean(axis=(1, 3))


def resample_surface_brightness(
    arr: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Resample a surface-brightness image by an arbitrary linear scale.

    ``scale`` is the linear size of an output pixel footprint relative to the
    input image: values above one enlarge the source and values below one
    shrink it.  Area resampling is used for shrinkage, so each output sample is
    the mean surface brightness over its input footprint.  Cubic interpolation
    is used only for enlargement.  This is both a closer match to intensive
    MJy/sr semantics and much cheaper than Gaussian-filtering a 1600-pixel
    atlas frame followed by a generic 3-D spline zoom.  It deliberately does
    not apply a pixel-area flux correction: callers decide the final integrated
    flux separately.

    The helper accepts either a 2-D image or a ``(H, W, C)`` channel stack and
    always returns a native-endian float32 array.
    """
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale!r}")
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim not in (2, 3):
        raise ValueError(f"expected a 2-D image or 3-D channel stack, got {a.shape}")
    height, width = a.shape[:2]
    out_height = max(1, int(round(height * float(scale))))
    out_width = max(1, int(round(width * float(scale))))
    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_CUBIC
    )
    out = cv2.resize(a, (out_width, out_height), interpolation=interpolation)
    # OpenCV drops a singleton channel axis.  Preserve this function's input
    # dimensionality contract for callers using an H x W x 1 cube.
    if a.ndim == 3 and out.ndim == 2:
        out = out[..., None]
    np.maximum(out, 0.0, out=out)
    return np.asarray(out, dtype=np.float32)


def rotate_quarter(arr: np.ndarray, k: int) -> np.ndarray:
    """Rotate by ``k`` exact quarter-turns counter-clockwise."""
    return np.rot90(np.asarray(arr), k=int(k) % 4)


def rotate_arbitrary(
    arr: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """Rotate in place-sized coordinates and clip interpolation undershoot.

    The returned image has the input shape.  Values outside the frame are zero;
    negative cubic overshoot is clipped because surface brightness is
    non-negative.  Whether interpolation is scientifically acceptable at the
    requested resolution is a policy decision for the caller.
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim not in (2, 3):
        raise ValueError(f"expected a 2-D image or 3-D channel stack, got {a.shape}")
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
    if a.ndim == 3 and out.ndim == 2:
        out = out[..., None]
    np.maximum(out, 0.0, out=out)
    return out.astype(np.float32)


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


def measure_halflight_radius_px(frame: np.ndarray, *, frac: float = 0.5) -> float:
    """Radius in pixels enclosing ``frac`` of positive flux about the centre.

    SKIRT atlas products centre their target galaxies geometrically.  For other
    products, callers should centre the source before using this helper.
    Empty or non-positive images return NaN.
    """
    a = np.asarray(frame, dtype=np.float64)
    a = np.where(np.isfinite(a) & (a > 0.0), a, 0.0)
    total = float(a.sum())
    if total <= 0.0:
        return float("nan")
    radii = radius_int_grid(a.shape)
    profile = np.bincount(radii.ravel(), weights=a.ravel())
    cumulative = np.cumsum(profile)
    target = frac * total
    index = int(np.searchsorted(cumulative, target))
    if index <= 0:
        return 0.5
    lower, upper = cumulative[index - 1], cumulative[index]
    subpixel = (target - lower) / (upper - lower) if upper > lower else 0.0
    return float(index - 1 + subpixel)


def centered_rotation_crop_slices(
    frame: np.ndarray,
    rebin: int,
    *,
    enclosed_fraction: float,
    padding: float,
) -> tuple[slice, slice]:
    """Centred square crop large enough to rotate the enclosed source light.

    The half-side is the requested curve-of-growth radius times ``sqrt(2)`` and
    ``padding``.  Its side is snapped to a multiple of ``rebin`` for subsequent
    block averaging.
    """
    height, width = frame.shape
    radius = measure_halflight_radius_px(frame, frac=enclosed_fraction)
    half = (
        int(np.ceil(radius * np.sqrt(2.0) * padding))
        if np.isfinite(radius) and radius > 0.0
        else min(height, width) // 2
    )
    side = min(2 * half, height, width)
    step = max(1, int(rebin))
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
