"""
Differential PSF kernel A satisfying ``A ⊛ H ≈ E``.

Given two PSFs — a sharper Hubble PSF ``H`` and a broader Euclid PSF
``E`` on a common pixel grid — this module computes the "differential"
real-space kernel ``A`` that, when convolved with an HST image, produces
an image consistent with what Euclid would see of the same scene
(modulo noise; see below).

Why this exists
---------------

We use the COSMOS HLSP F814W mosaic as a real-morphology HR target.
The naive forward model would convolve those HST cutouts with the
Euclid PSF directly, but that double-convolves through HST's PSF — the
effective PSF in the synthesised LR ends up ``E ⊛ H`` rather than ``E``,
~17 % broader than real Euclid. The differential kernel removes that
bias: applying ``A`` instead of ``E`` to an HST observation gives an LR
with the right effective PSF.

The math is straightforward Wiener deconvolution in Fourier space:

    Â(k) = Ê(k) · conj(Ĥ(k)) / (|Ĥ(k)|² + reg²)

The regularisation term ``reg`` prevents noise blow-up at high spatial
frequencies where ``Ĥ`` falls off. We keep it small enough that the
in-band response is essentially ``Ê/Ĥ`` but large enough that the kernel
remains a low-pass filter (which it should be when ``E`` is broader than
``H``).

Validity
--------

``A`` is well-defined and physically sensible iff ``E`` is *broader* than
``H`` at every spatial frequency where ``H`` carries appreciable power.
For Euclid VIS (0.16″ FWHM) vs HST F814W (~0.10″ FWHM) this holds; for
NISP bands (0.40-0.48″ FWHM) it's even more comfortable. Reverse the
roles (asking for an HST-like kernel from Euclid data) and the same code
would amplify noise — that's the ill-posed deconvolution regime and is
intentionally not supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from astropy.io import fits
from scipy import signal as scipy_signal


# ---------------------------------------------------------------------------
# Pure-numpy core
# ---------------------------------------------------------------------------

def _fourier_shift(arr: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Shift a 2-D array by sub-pixel ``(dy, dx)`` via an FFT phase ramp.

    Sinc-interpolation for band-limited inputs, exact to round-off.
    Total flux is preserved exactly (a phase ramp doesn't touch the DC
    component). Wraparound at the boundaries is irrelevant in our use
    case — PSFs are centred and decay to ~0 well before the edges.
    """
    H, W = arr.shape
    ky = np.fft.fftfreq(H)
    kx = np.fft.fftfreq(W)
    phase = np.exp(
        -2j * np.pi * (ky[:, None] * dy + kx[None, :] * dx)
    )
    return np.fft.ifft2(np.fft.fft2(arr) * phase).real


def _recenter_to_geometric(arr: np.ndarray) -> np.ndarray:
    """Sub-pixel-shift ``arr`` so its flux centroid lands on the
    geometric centre ``((H-1)/2, (W-1)/2)``.

    The centroid is computed on the *positive* part of the array
    (``max(arr, 0)``) to avoid the Wiener-deconvolution-style
    negative wings of an ePSF biasing the moment. For typical empirical
    PSFs the positive half carries essentially all the flux anyway.

    The shift itself uses :func:`scipy.ndimage.shift` (cubic spline,
    ``mode='constant'`` so out-of-bounds pixels are zero-extrapolated).
    Earlier versions used :func:`_fourier_shift`, which is exact for
    band-limited inputs but **periodic at the array boundary**: a tiny
    amount of flux at one edge wraps around to the opposite edge,
    which after the Wiener inverse showed up as bright ringing along
    each border centred on the central row/column (border / interior
    amplitude ratio ~30× in the convolved result). The spatial-spline
    shift has a small interpolation error (~10 % bump in rel.RMS for
    sub-pixel offsets) but no wraparound — strictly the better trade
    for an empirical-PSF deconvolution workflow.

    No-ops when the centroid is already at the geometric centre to
    within 1e-3 of a pixel (cheap guard against introducing
    interpolation round-off when there's nothing to fix).
    """
    from scipy.ndimage import shift as _ndshift

    pos = np.maximum(arr, 0.0)
    total = float(pos.sum())
    if not np.isfinite(total) or total <= 0:
        return arr     # degenerate — leave it alone
    yy, xx = np.indices(arr.shape)
    cy = float((yy * pos).sum() / total)
    cx = float((xx * pos).sum() / total)
    target_y = (arr.shape[0] - 1) / 2.0
    target_x = (arr.shape[1] - 1) / 2.0
    dy = target_y - cy
    dx = target_x - cx
    if abs(dy) < 1e-3 and abs(dx) < 1e-3:
        return arr
    shifted = _ndshift(
        arr.astype(np.float64), shift=(dy, dx),
        order=3, mode="constant", cval=0.0,
    )
    return shifted.astype(arr.dtype)


def _pad_to(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Zero-pad ``arr`` to ``shape``, placing the input's centre pixel
    at the target array's ``shape // 2`` index in each axis.

    The naive formula ``(target - source) // 2`` agrees with this for
    (even, even) and (odd, odd) but is **off by one** for (even target,
    odd source) — which is exactly our hot case (pad an odd-N input
    PSF into an even-N (power-of-2) FFT grid). The off-by-one matters
    because ``ifftshift`` uses ``shape // 2`` as the centre convention;
    a wrong pad introduces a 1-pixel shift that the FFT then turns
    into a linear phase term, and the resulting kernel comes out
    centred 1 pixel off the geometric centre.
    """
    H, W = arr.shape
    sH, sW = shape
    if sH < H or sW < W:
        raise ValueError(f"target shape {shape} smaller than input {arr.shape}")
    out = np.zeros(shape, dtype=arr.dtype)
    i0 = sH // 2 - H // 2
    j0 = sW // 2 - W // 2
    out[i0:i0 + H, j0:j0 + W] = arr
    return out


def compute_differential_kernel(
    psf_euclid: np.ndarray,
    psf_hubble: np.ndarray,
    *,
    regularisation: float = 1e-3,
    target_shape: Tuple[int, int] | None = None,
    recenter: bool = True,
) -> np.ndarray:
    """Solve ``A ⊛ H ≈ E`` and return ``A`` on the same grid as the inputs.

    Both PSFs MUST be sampled on the same pixel grid (same pixel scale
    and same — odd, square — kernel side length). The caller is
    responsible for any prior resampling.

    Parameters
    ----------
    psf_euclid, psf_hubble
        2-D float arrays, normalised to ``sum=1`` (we don't re-normalise
        — feeding un-normalised PSFs gives a kernel with non-unity DC
        gain which silently breaks photometry downstream).
    regularisation
        Wiener regulariser as a fraction of ``max|Ĥ|``. Larger values
        suppress high-frequency content more aggressively (= smoother A,
        less noise amplification). The default ``1e-3`` is conservative
        and matches the regime where ``H`` falls off well before the
        Nyquist of the grid.
    target_shape
        Optional ``(H, W)`` to zero-pad both PSFs to before the FFT — use
        a larger grid to reduce FFT boundary artefacts. Defaults to the
        nearest power of two ≥ 2× the input side.
    recenter
        When True (default), sub-pixel-shift each input PSF so its flux
        centroid lands on the geometric centre via a Fourier phase ramp.
        Sub-pixel centroid mismatch between ``E`` and ``H`` (e.g. the
        HST ePSF whose centroid sits at (256.6, 256.7) on a 513² grid
        with geometric centre (256, 256)) puts a ~half-pixel phase
        ramp on ``Ê/Ĥ``; ``exp(iπk/N) = ±1`` at Nyquist, so the spatial-
        domain ``A`` comes out as a pixel-by-pixel checkerboard of
        ±values. The recentering is exact (FFT shift preserves total
        flux to round-off) and disables itself when the centroid is
        already inside 1e-3 px of the geometric centre.

    Returns
    -------
    np.ndarray
        Real-space kernel ``A`` on the same grid as the inputs (i.e.,
        cropped back from ``target_shape`` to the input shape). Sums to
        ~1 by construction (DC gain of ``Â`` at k=0 is ``Ê(0)/Ĥ(0) = 1``
        when both inputs are unit-flux normalised).
    """
    if psf_euclid.shape != psf_hubble.shape:
        raise ValueError(
            f"PSF shapes must match; got Euclid {psf_euclid.shape} vs "
            f"Hubble {psf_hubble.shape}"
        )
    if psf_euclid.ndim != 2:
        raise ValueError(f"PSFs must be 2-D, got {psf_euclid.ndim}-D")
    H_in, W_in = psf_euclid.shape

    e_in = psf_euclid.astype(np.float64)
    h_in = psf_hubble.astype(np.float64)
    if recenter:
        e_in = _recenter_to_geometric(e_in)
        h_in = _recenter_to_geometric(h_in)

    # Pad for FFT — larger grid means less wraparound contamination of the
    # kernel's outer wings. Round up to the next power of 2 ≥ 2× the input.
    if target_shape is None:
        side = 1 << int(np.ceil(np.log2(2 * max(H_in, W_in))))
        target_shape = (side, side)
    H_pad, W_pad = target_shape

    e_pad = _pad_to(e_in, target_shape)
    h_pad = _pad_to(h_in, target_shape)

    # ifftshift moves the PSF centre to (0, 0) so the FFT phase reference
    # is the kernel centre rather than the array corner — keeps Â real.
    e_hat = np.fft.fft2(np.fft.ifftshift(e_pad))
    h_hat = np.fft.fft2(np.fft.ifftshift(h_pad))

    reg = float(regularisation) * float(np.abs(h_hat).max())
    # Wiener-regularised inverse: stable even where |H_hat| is near zero.
    a_hat = e_hat * np.conjugate(h_hat) / (np.abs(h_hat) ** 2 + reg ** 2)
    a_pad = np.fft.fftshift(np.fft.ifft2(a_hat).real)

    # Crop back to the input size — the wings of A live mostly inside
    # this support for reasonable regularisation.
    #
    # Same centring convention as _pad_to: ``shape // 2`` is the centre
    # in both source and destination. Using ``(H_pad - H_in) // 2``
    # silently shifts A by 1 pixel for (even pad, odd in), which then
    # shows up as a dipole at the dead centre of ``A ⊛ H − E``.
    i0 = H_pad // 2 - H_in // 2
    j0 = W_pad // 2 - W_in // 2
    a = a_pad[i0:i0 + H_in, j0:j0 + W_in]
    return a.astype(np.float32)


def apply_kernel(image: np.ndarray, kernel: np.ndarray,
                 *, mode: str = "same") -> np.ndarray:
    """Convolve ``image`` with ``kernel`` (FFT-based, same shape out).

    Wraps :func:`scipy.signal.fftconvolve` so callers don't have to
    import it. Kernel is applied with ``mode='same'`` and centred on the
    image's pixel grid — i.e., a delta-function kernel is the identity.
    """
    return scipy_signal.fftconvolve(image, kernel, mode=mode).astype(image.dtype)


# ---------------------------------------------------------------------------
# Provenance dataclass — what goes in/comes out of the FITS file
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DifferentialKernel:
    """In-memory representation of a saved differential kernel.

    Attributes
    ----------
    data
        2-D float32 kernel on the project HR grid (0.05″/pix by default).
    pixel_scale_arcsec
        Pixel scale of the kernel grid.
    euclid_band
        Euclid band name the kernel maps *into* (e.g., ``"VIS"``).
    hst_filter
        HST filter the kernel maps *from* (e.g., ``"F814W"``).
    regularisation
        Wiener regulariser used at construction time.
    """

    data:               np.ndarray
    pixel_scale_arcsec: float
    euclid_band:        str
    hst_filter:         str
    regularisation:     float

    def save(self, path: str) -> None:
        """Persist as a FITS file with provenance header keys."""
        hdu = fits.PrimaryHDU(np.ascontiguousarray(self.data, dtype=np.float32))
        h = hdu.header
        h["OBJECT"]  = ("Differential PSF kernel A", "A satisfies A * H = E")
        h["EUCLBAND"] = (self.euclid_band, "Euclid band (target PSF E)")
        h["HSTFILT"]  = (self.hst_filter,  "HST filter (source PSF H)")
        h["PIXSCALE"] = (self.pixel_scale_arcsec, "arcsec / pix")
        h["WIENERR"]  = (self.regularisation, "Wiener regularisation fraction")
        h["BUNIT"]    = ("", "dimensionless (sums to ~1)")
        hdu.writeto(path, overwrite=True)

    @classmethod
    def from_fits(cls, path: str) -> "DifferentialKernel":
        with fits.open(path, memmap=False) as hdul:
            sci = next(
                (e for e in hdul if e.is_image and e.data is not None), None,
            )
            if sci is None:
                raise IOError(f"no image HDU in {path}")
            h = sci.header
            return cls(
                data               = np.asarray(sci.data, dtype=np.float32),
                pixel_scale_arcsec = float(h.get("PIXSCALE", 0.05)),
                euclid_band        = str(h.get("EUCLBAND", "VIS")),
                hst_filter         = str(h.get("HSTFILT", "F814W")),
                regularisation     = float(h.get("WIENERR", 1e-3)),
            )

    @property
    def dc_gain(self) -> float:
        """Sum of all kernel pixels — should be close to 1 by construction."""
        return float(self.data.sum())
