"""
Differential PSF kernel A satisfying ``A ⊛ H ≈ E``.

Given two PSFs — a sharper Hubble PSF ``H`` and a broader Euclid PSF
``E`` on a common pixel grid — this module computes the "differential"
real-space kernel ``A`` that, when convolved with an HST image, produces
an image consistent with what Euclid would see of the same scene
(modulo noise; see below).

Purpose
-------

The COSMOS HLSP F814W mosaic is the real-morphology HR target.
Convolving those HST cutouts with the Euclid PSF directly would
double-convolve through HST's own PSF — the effective PSF in the
synthesised LR would be ``E ⊛ H`` rather than ``E``, ~17 % broader than
real Euclid. The differential kernel removes that bias: applying ``A``
instead of ``E`` to an HST observation gives an LR with the right
effective PSF.

The math is straightforward Wiener deconvolution in Fourier space:

    Â(k) = Ê(k) · conj(Ĥ(k)) / (|Ĥ(k)|² + reg²)

The regularisation term ``reg`` prevents noise blow-up at high spatial
frequencies where ``Ĥ`` falls off. It is small enough that the in-band
response is essentially ``Ê/Ĥ`` but large enough that the kernel remains
a low-pass filter (which it is when ``E`` is broader than ``H``).

Validity
--------

``A`` is well-defined and physically sensible iff ``E`` is *broader* than
``H`` at every spatial frequency where ``H`` carries appreciable power.
For Euclid VIS (0.16″ FWHM) vs HST F814W (~0.10″ FWHM) this holds; for
NISP bands (0.40-0.48″ FWHM) the margin is larger. Reversing the roles
(asking for an HST-like kernel from Euclid data) is the ill-posed
deconvolution regime that amplifies noise, and is not supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from astropy.io import fits
from scipy import signal as scipy_signal
from scipy.ndimage import shift as _ndshift

# ---------------------------------------------------------------------------
# Pure-numpy core
# ---------------------------------------------------------------------------

def _fourier_shift(arr: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Shift a 2-D array by sub-pixel ``(dy, dx)`` via an FFT phase ramp.

    Sinc-interpolation for band-limited inputs, exact to round-off.
    Total flux is preserved exactly (a phase ramp doesn't touch the DC
    component). Wraparound at the boundaries is harmless here: PSFs are
    centred and decay to ~0 well before the edges.
    """
    H, W = arr.shape
    ky = np.fft.fftfreq(H)
    kx = np.fft.fftfreq(W)
    phase = np.exp(
        -2j * np.pi * (ky[:, None] * dy + kx[None, :] * dx)
    )
    return np.fft.ifft2(np.fft.fft2(arr) * phase).real


def _positive_centroid(arr: np.ndarray) -> tuple[float, float]:
    """Flux-weighted centroid of the positive part of ``arr``.

    Uses the same convention as :func:`_recenter_to_geometric`, kept in one
    helper so callers needing both the centroid and the recentered array
    compute it consistently.
    """
    pos = np.maximum(arr, 0.0)
    total = float(pos.sum())
    if not np.isfinite(total) or total <= 0:
        # Degenerate input: return the geometric centre so a caller
        # computing "shift the centroid to target" sees 0 shift.
        return (arr.shape[0] - 1) / 2.0, (arr.shape[1] - 1) / 2.0
    yy, xx = np.indices(arr.shape)
    return (
        float((yy * pos).sum() / total),
        float((xx * pos).sum() / total),
    )


def _recenter_to_geometric(arr: np.ndarray) -> np.ndarray:
    """Sub-pixel-shift ``arr`` so its flux centroid lands on the
    geometric centre ``((H-1)/2, (W-1)/2)``.

    The centroid is computed on the *positive* part of the array
    (``max(arr, 0)``) so the Wiener-deconvolution-style negative wings of
    an ePSF do not bias the moment. For typical empirical PSFs the
    positive half carries essentially all the flux.

    The shift uses :func:`scipy.ndimage.shift` (cubic spline,
    ``mode='constant'`` so out-of-bounds pixels are zero-extrapolated).
    This has a small interpolation error (~10 % bump in rel.RMS for
    sub-pixel offsets) but no boundary wraparound, which matters for an
    empirical-PSF deconvolution workflow.

    No-ops when the centroid is already at the geometric centre to
    within 1e-3 of a pixel, avoiding interpolation round-off when there
    is nothing to fix.
    """

    cy, cx = _positive_centroid(arr)
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


def _pad_to(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Zero-pad ``arr`` to ``shape``, placing the input's centre pixel
    at the target array's ``shape // 2`` index in each axis.

    The centre is placed at ``shape // 2`` to match the ``ifftshift``
    convention. ``(target - source) // 2`` would be off by one for
    (even target, odd source) — the case of padding an odd-N input PSF
    into an even-N (power-of-2) FFT grid — introducing a 1-pixel shift
    that the FFT turns into a linear phase term, leaving the kernel
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
    target_shape: tuple[int, int] | None = None,
    recenter: bool = True,
) -> np.ndarray:
    """Solve ``A ⊛ H ≈ E`` and return ``A`` on the same grid as the inputs.

    Both PSFs MUST be sampled on the same pixel grid (same pixel scale
    and same — odd, square — kernel side length). The caller is
    responsible for any prior resampling.

    Parameters
    ----------
    psf_euclid, psf_hubble
        2-D float arrays, normalised to ``sum=1``. They are not
        re-normalised here; un-normalised PSFs give a kernel with
        non-unity DC gain that breaks photometry downstream.
    regularisation
        Wiener regulariser as a fraction of ``max|Ĥ|``. Larger values
        suppress high-frequency content more aggressively (= smoother A,
        less noise amplification). The default ``1e-3`` is conservative,
        for the regime where ``H`` falls off well before the Nyquist of
        the grid.
    target_shape
        Optional ``(H, W)`` to zero-pad both PSFs to before the FFT — use
        a larger grid to reduce FFT boundary artefacts. Defaults to the
        nearest power of two ≥ 2× the input side.
    recenter
        When True (default), sub-pixel-shift each input PSF so its
        flux centroid lands on the geometric centre, then **undo the
        net shift on the output kernel** so the returned ``A`` is
        physically correct against the *original* un-recentered E and
        H — what downstream callers convolve real cutouts with.

        A centroid mismatch between E and H puts a sub-pixel phase ramp
        on ``Ê/Ĥ``; at Nyquist this is ``≈±1`` and the spatial-domain
        ``A`` comes out as a pixel-by-pixel checkerboard. Recentering
        both inputs onto the geometric centre removes the phase ramp at
        the cost of baking a *shift* into ``A_solved`` equal to
        ``(dy_e − dy_h, dx_e − dx_h)``. The inverse spatial shift (scipy
        spline, mode='constant', no wraparound) is then applied to
        ``A_solved`` so the final kernel satisfies
        ``A ⊛ H_original ≈ E_original`` — the contract any downstream
        forward model relies on.

        Disables itself when the centroids of both inputs already sit
        inside 1e-3 px of the geometric centre (no recentering, no
        correction needed).

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
    cy_e = cx_e = cy_h = cx_h = 0.0
    if recenter:
        # Centroids of the originals, used to undo the net shift on the
        # kernel below. Same flux-weighted positive centroid
        # `_recenter_to_geometric` uses.
        cy_e, cx_e = _positive_centroid(e_in)
        cy_h, cx_h = _positive_centroid(h_in)
        e_in = _recenter_to_geometric(e_in)
        h_in = _recenter_to_geometric(h_in)

    # Pad for FFT: a larger grid means less wraparound contamination of the
    # kernel's outer wings. Round up to the next power of 2 ≥ 2× the input.
    if target_shape is None:
        side = 1 << int(np.ceil(np.log2(2 * max(H_in, W_in))))
        target_shape = (side, side)
    H_pad, W_pad = target_shape

    e_pad = _pad_to(e_in, target_shape)
    h_pad = _pad_to(h_in, target_shape)

    # ifftshift moves the PSF centre to (0, 0) so the FFT phase reference
    # is the kernel centre rather than the array corner, keeping Â real.
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
    # in both source and destination. ``(H_pad - H_in) // 2`` would shift
    # A by 1 pixel for (even pad, odd in), showing up as a dipole at the
    # dead centre of ``A ⊛ H − E``.
    i0 = H_pad // 2 - H_in // 2
    j0 = W_pad // 2 - W_in // 2
    a = a_pad[i0:i0 + H_in, j0:j0 + W_in]

    if recenter:
        # Undo the (cy_e − cy_h, cx_e − cx_h) shift the recentering
        # baked into A, so the returned kernel satisfies
        # ``A ⊛ H_original ≈ E_original``. Downstream code convolves real
        # cutouts with the *original* H; without this undo the
        # forward-modelled "Euclid-like" LR would be sub-pixel-shifted
        # from what real Euclid observes. See the docstring's `recenter`
        # section for the derivation.
        a = _ndshift(
            a.astype(np.float64),
            shift=(cy_e - cy_h, cx_e - cx_h),
            order=3, mode="constant", cval=0.0,
        )

    return a.astype(np.float32)


def apply_kernel(image: np.ndarray, kernel: np.ndarray,
                 *, mode: Literal["full", "same", "valid"] = "same") -> np.ndarray:
    """Convolve ``image`` with ``kernel`` (FFT-based, same shape out).

    Wraps :func:`scipy.signal.fftconvolve` so callers don't have to
    import it. Kernel is applied with ``mode='same'`` and centred on the
    image's pixel grid — i.e., a delta-function kernel is the identity.
    """
    return scipy_signal.fftconvolve(image, kernel, mode=mode).astype(image.dtype)


# ---------------------------------------------------------------------------
# Provenance dataclass — fields stored in / read from the FITS file
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
    def from_fits(cls, path: str) -> DifferentialKernel:
        with fits.open(path, memmap=False) as hdul:
            sci: fits.PrimaryHDU | fits.ImageHDU | None = None
            for raw_hdu in hdul:
                candidate = cast(fits.PrimaryHDU | fits.ImageHDU, raw_hdu)
                if candidate.is_image and candidate.data is not None:
                    sci = candidate
                    break
            if sci is None:
                raise OSError(f"no image HDU in {path}")
            h = sci.header
            return cls(
                data               = np.asarray(sci.data, dtype=np.float32),
                pixel_scale_arcsec = float(cast(Any, h.get("PIXSCALE", 0.05))),
                euclid_band        = str(h.get("EUCLBAND", "VIS")),
                hst_filter         = str(h.get("HSTFILT", "F814W")),
                regularisation     = float(cast(Any, h.get("WIENERR", 1e-3))),
            )

    @property
    def dc_gain(self) -> float:
        """Sum of all kernel pixels — should be close to 1 by construction."""
        return float(self.data.sum())
