"""Fast numerical kernels for the PSF forward model.

``same_convolve``: ``fftconvolve(mode="same")`` computed on the shortest
circular FFT that reproduces it exactly. A kernel wider than the image needs
only one period at least as long as itself, not the full linear length. For
the training geometry (510² field, 1023² kernel) that is 1024² instead of
1536².

``bilinear_warp_reflect``: ``scipy.ndimage.map_coordinates(order=1,
mode="reflect")`` on the displaced pixel grid, as a numba kernel that runs in
the kernel's own dtype and releases the GIL. It replaces scipy's generic
spline machinery and the float64 copy of a 1023² PSF on every elastic warp.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy import fft as sp_fft


def circular_same_length(n_image: int, n_kernel: int) -> int:
    """Shortest fast FFT length ``M`` for which a circular convolution of an
    ``n_image`` signal with an ``n_kernel`` kernel equals the linear one on
    the ``same`` window.

    The window reads kernel indices ``j - m`` in
    ``[c - (n_image - 1), c + n_image - 1]`` with ``c = (n_kernel - 1) // 2``.
    The result is exact when the kernel fits (``M >= n_kernel``), the window's
    last index fits (``M >= c + n_image``), and negative indices wrap onto the
    zero padding (``M >= n_kernel + n_image - 1 - c``)."""
    c = (n_kernel - 1) // 2
    need = max(n_kernel, c + n_image, n_kernel + n_image - 1 - c)
    return int(sp_fft.next_fast_len(need, real=True))


def same_convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """``scipy.signal.fftconvolve(image, kernel, mode="same")`` for 2-D
    arrays, in their common floating dtype (at least float32)."""
    ny, nx = image.shape
    ky, kx = kernel.shape
    my, mx = circular_same_length(ny, ky), circular_same_length(nx, kx)
    dtype = np.result_type(image.dtype, kernel.dtype, np.float32)
    spec = (sp_fft.rfft2(np.asarray(image, dtype), s=(my, mx))
            * sp_fft.rfft2(np.asarray(kernel, dtype), s=(my, mx)))
    full = sp_fft.irfft2(spec, s=(my, mx))
    cy, cx = (ky - 1) // 2, (kx - 1) // 2
    return full[cy:cy + ny, cx:cx + nx]


@njit(nogil=True)
def _reflect_index(i, n):
    """scipy.ndimage ``reflect`` (half-sample symmetric) index for any
    integer ``i``: ``d c b a | a b c d | d c b a``."""
    if n == 1:
        return 0
    period = 2 * n
    i = i % period
    if i < 0:
        i += period
    if i >= n:
        i = period - 1 - i
    return i


@njit(nogil=True)
def _bilinear_reflect_kernel(data, dy, dx, out):
    h, w = data.shape
    for i in range(h):
        for j in range(w):
            y = i + float(dy[i, j])
            x = j + float(dx[i, j])
            y0 = math.floor(y)
            x0 = math.floor(x)
            fy = y - y0
            fx = x - x0
            ya = _reflect_index(int(y0), h)
            yb = _reflect_index(int(y0) + 1, h)
            xa = _reflect_index(int(x0), w)
            xb = _reflect_index(int(x0) + 1, w)
            out[i, j] = ((1.0 - fy) * ((1.0 - fx) * data[ya, xa] + fx * data[ya, xb])
                         + fy * ((1.0 - fx) * data[yb, xa] + fx * data[yb, xb]))


def bilinear_warp_reflect(data: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """``map_coordinates(data, (yy + dy, xx + dx), order=1, mode="reflect")``
    (``yy, xx`` the pixel grid) in ``data``'s dtype, without holding the GIL."""
    data = np.ascontiguousarray(data)
    out = np.empty_like(data)
    _bilinear_reflect_kernel(data, np.ascontiguousarray(dy), np.ascontiguousarray(dx), out)
    return out
