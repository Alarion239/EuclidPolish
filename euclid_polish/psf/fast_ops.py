"""Fast numerical kernels for the PSF forward model.

``same_convolve``: ``fftconvolve(mode="same")`` computed on the shortest
circular FFT that reproduces it exactly. A kernel wider than the image needs
only one period at least as long as itself, not the full linear length. For
the training geometry (510² field, 1023² kernel) that is 1024² instead of
1536².
"""

from __future__ import annotations

import numpy as np
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
