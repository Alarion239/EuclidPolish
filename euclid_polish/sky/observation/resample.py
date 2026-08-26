"""Image resampling primitives.

The pixel-centred bilinear upsampler is used by the NISP-to-VIS-LR step of
the multi-band forward model (NISP native 0.30" to the 0.10" MER grid). It
matches the ``BILINEAR`` interpolation configured in the Euclid Q1 CT_SWarp
mosaicing pipeline. A cubic-spline alternative remains available for
experiments through ``kernel="cubic"``.

Both implementations use ``grid_mode=True`` so output pixel centres map to
input coordinate ``(j + 0.5) / factor - 0.5``. Samples beyond the array edge
use the nearest edge value. The observation model pads its private NISP
workspace and crops the delivered science region, so this boundary convention
does not affect the simulated field interior.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
from scipy.ndimage import zoom


def _validated_input(arr_2d: np.ndarray, factor: int) -> np.ndarray:
    """Return a 2-D array after validating the integer scale factor."""
    values = np.asarray(arr_2d)
    if values.ndim != 2:
        raise ValueError(f"arr_2d must be 2-D, got shape {values.shape}")
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    return values


@lru_cache(maxsize=16)
def _bilinear_coordinates(
    n_input: int,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lower/upper indices and upper-pixel weights for one axis."""
    position = (
        (np.arange(n_input * factor, dtype=np.float64) + 0.5) / factor - 0.5
    )
    lower_raw = np.floor(position).astype(np.int64)
    fraction = position - lower_raw
    lower = np.clip(lower_raw, 0, n_input - 1)
    upper = np.clip(lower_raw + 1, 0, n_input - 1)
    return lower, upper, fraction


def bilinear_upsample(arr_2d: np.ndarray, factor: int) -> np.ndarray:
    """Upsample a 2-D array by an integer factor using bilinear sampling.

    Output shape is ``(H * factor, W * factor)`` and output dtype matches the
    input. ``factor=1`` returns an independent copy.
    """
    values = _validated_input(arr_2d, factor)
    if factor == 1:
        return values.copy()

    height, width = values.shape
    x0, x1, weight_x = _bilinear_coordinates(width, factor)
    y0, y1, weight_y = _bilinear_coordinates(height, factor)
    work = values.astype(np.float64, copy=False)
    horizontal = (
        work[:, x0] * (1.0 - weight_x)
        + work[:, x1] * weight_x
    )
    output = (
        horizontal[y0, :] * (1.0 - weight_y[:, None])
        + horizontal[y1, :] * weight_y[:, None]
    )
    return output.astype(values.dtype, copy=False)


def cubic_upsample(arr_2d: np.ndarray, factor: int) -> np.ndarray:
    """Upsample a 2-D array with cubic-spline interpolation."""
    values = _validated_input(arr_2d, factor)
    if factor == 1:
        return values.copy()
    return zoom(
        values,
        zoom=factor,
        order=3,
        mode="nearest",
        grid_mode=True,
    )


def upsample(
    arr_2d: np.ndarray,
    factor: int,
    kernel: Literal["bilinear", "cubic"] = "bilinear",
) -> np.ndarray:
    """Dispatch to the requested resampling kernel."""
    if kernel == "bilinear":
        return bilinear_upsample(arr_2d, factor)
    if kernel == "cubic":
        return cubic_upsample(arr_2d, factor)
    raise ValueError(f"Unknown resample kernel {kernel!r}")
