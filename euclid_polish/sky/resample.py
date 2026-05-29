"""
Image resampling primitives.

Currently exposes a 3× Lanczos-3 upsampler used by the NISP→VIS-LR step of
the multi-band forward model (NISP native 0.30″ → VIS-LR-matched 0.10″).
The Lanczos-3 kernel matches the SWarp/MER pipeline; a faster cubic-spline
alternative is provided behind the ``kernel="cubic"`` option.

The Lanczos-3 weight matrices for each output axis are pre-computed once
per output shape, then reused for both axes — separable 1-D filters yield
the same result as the 2-D kernel for the (separable) Lanczos kernel.

Edge behaviour: input samples outside the input range are treated as zero
(zero-padding). For our use case the NISP image always extends beyond the
HR field it's downsampled from, so the edges contain only sky+noise and
zero-padding does not bias the science region.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Literal

import numpy as np

from scipy.ndimage import zoom


# ---------------------------------------------------------------------------
# 1-D kernels
# ---------------------------------------------------------------------------

def _lanczos3_kernel(t: np.ndarray) -> np.ndarray:
    """1-D Lanczos-3 kernel L(t) = sinc(t) · sinc(t/3), support |t| < 3."""
    a = 3
    t_abs = np.abs(t)
    out = np.zeros_like(t, dtype=np.float64)
    mask = t_abs < a
    tm = t_abs[mask]
    # Stable evaluation; handle t == 0 separately.
    res = np.empty_like(tm)
    zero = tm == 0.0
    res[zero] = 1.0
    nonzero = ~zero
    tn = tm[nonzero]
    res[nonzero] = (
        a * np.sin(np.pi * tn) * np.sin(np.pi * tn / a)
        / (np.pi * np.pi * tn * tn)
    )
    out[mask] = res
    return out


@lru_cache(maxsize=8)
def _build_lanczos3_weights(
    n_in: int, factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute per-output-pixel weights & input-index window.

    Returns ``(weights, lo, hi)``:

      * weights : (n_out, kernel_len) float64
      * lo      : (n_out,) int  — start input index per output (clamped to 0)
      * hi      : (n_out,) int  — end+1 input index per output

    Where ``kernel_len = hi[i] - lo[i]`` (≤ 7 for Lanczos-3 with arbitrary
    factor; for integer ``factor`` it's exactly ``2 * 3 = 6`` interior and
    fewer near boundaries).

    The output sample at output index ``j`` maps to input coordinate
    ``(j + 0.5) / factor - 0.5`` — i.e. pixel-centred conversion (the
    natural choice if input/output pixel grids share the same continuous
    origin).
    """
    A = 3
    n_out = n_in * factor
    # Continuous input position of each output pixel centre.
    out_idx = np.arange(n_out, dtype=np.float64)
    pos_in  = (out_idx + 0.5) / factor - 0.5

    # Kernel support window per output: [pos - 3, pos + 3]
    lo = np.maximum(0,        np.floor(pos_in - A + 1).astype(np.int64))
    hi = np.minimum(n_in,     np.floor(pos_in + A    ).astype(np.int64) + 1)

    kernel_len = int(2 * A)        # full interior support
    weights = np.zeros((n_out, kernel_len), dtype=np.float64)

    for j in range(n_out):
        idx = np.arange(lo[j], hi[j], dtype=np.int64)
        t   = pos_in[j] - idx
        w   = _lanczos3_kernel(t)
        # Re-normalise the kernel so summed weights equal 1 (preserves a
        # constant input). The Lanczos kernel partition-of-unity is exact
        # in the interior but breaks near boundaries where the kernel is
        # truncated; renormalising at the boundary stops dark edges.
        w_sum = w.sum()
        if w_sum != 0.0:
            w = w / w_sum
        weights[j, : idx.size] = w
    return weights, lo, hi


# ---------------------------------------------------------------------------
# 2-D Lanczos-3 upsample
# ---------------------------------------------------------------------------

def _apply_1d_filter(arr_2d: np.ndarray, weights: np.ndarray,
                     lo: np.ndarray, hi: np.ndarray, axis: int) -> np.ndarray:
    """Apply a precomputed 1-D resampling filter along ``axis``.

    ``arr_2d``  : (H, W) float
    Returns     : (H, W')  if axis=1
                  (H', W)  if axis=0
    """
    if axis == 1:
        H, _ = arr_2d.shape
        n_out = weights.shape[0]
        out = np.zeros((H, n_out), dtype=np.float64)
        for j in range(n_out):
            j_lo, j_hi = int(lo[j]), int(hi[j])
            k = j_hi - j_lo
            if k <= 0:
                continue
            out[:, j] = arr_2d[:, j_lo:j_hi] @ weights[j, :k]
        return out
    elif axis == 0:
        _, W = arr_2d.shape
        n_out = weights.shape[0]
        out = np.zeros((n_out, W), dtype=np.float64)
        for j in range(n_out):
            j_lo, j_hi = int(lo[j]), int(hi[j])
            k = j_hi - j_lo
            if k <= 0:
                continue
            out[j, :] = weights[j, :k] @ arr_2d[j_lo:j_hi, :]
        return out
    raise ValueError(f"axis must be 0 or 1, got {axis}")


def lanczos3_upsample(arr_2d: np.ndarray, factor: int) -> np.ndarray:
    """Upsample a 2-D array by an integer factor using separable Lanczos-3.

    Output shape is ``(H * factor, W * factor)``. Output dtype is the same
    as the input.
    """
    if factor < 1:
        raise ValueError(f"factor must be ≥ 1, got {factor}")
    if factor == 1:
        return arr_2d.copy()
    H, W = arr_2d.shape
    # x then y (separable). Cache by axis length × factor.
    w_x, lo_x, hi_x = _build_lanczos3_weights(W, factor)
    w_y, lo_y, hi_y = _build_lanczos3_weights(H, factor)
    inter = _apply_1d_filter(arr_2d.astype(np.float64, copy=False),
                             w_x, lo_x, hi_x, axis=1)   # (H, W·f)
    out   = _apply_1d_filter(inter, w_y, lo_y, hi_y, axis=0)  # (H·f, W·f)
    return out.astype(arr_2d.dtype, copy=False)


def cubic_upsample(arr_2d: np.ndarray, factor: int) -> np.ndarray:
    """Cubic-spline upsampling (alternative to Lanczos-3).

    Uses :func:`scipy.ndimage.zoom` with order=3. Substantially faster than
    Lanczos-3 but with more low-pass smoothing.
    """
    if factor < 1:
        raise ValueError(f"factor must be ≥ 1, got {factor}")
    if factor == 1:
        return arr_2d.copy()
    return zoom(arr_2d, zoom=factor, order=3, mode="nearest", grid_mode=True)


def upsample(
    arr_2d: np.ndarray, factor: int, kernel: Literal["lanczos3", "cubic"] = "lanczos3",
    *, conserve_flux: bool = False,
) -> np.ndarray:
    """Dispatch to the requested resampling kernel.

    The kernels are **value-preserving** (kernel weights sum to 1, so a flat
    field maps to itself); upsampling an image by ``factor`` therefore leaves
    each pixel's value ~unchanged and inflates the *total* by ~``factor²``.
    Pass ``conserve_flux=True`` to instead split each source pixel's flux
    across the finer grid (divide by ``factor²``) — the correct behaviour for
    resampling **electron counts** (e.g. the NISP 0.30″→0.10″ stage, matching
    how MER conserves total flux). Conservation is approximate (~%) because
    Lanczos has negative side lobes and edge renormalisation.
    """
    if kernel == "lanczos3":
        out = lanczos3_upsample(arr_2d, factor)
    elif kernel == "cubic":
        out = cubic_upsample(arr_2d, factor)
    else:
        raise ValueError(f"Unknown resample kernel {kernel!r}")
    if conserve_flux and factor > 1:
        out = (out / float(factor * factor)).astype(arr_2d.dtype, copy=False)
    return out
