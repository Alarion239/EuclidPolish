"""Physical resolution of a reconstruction against an independent sharper
image: the Fourier ring correlation (FRC) of microscopy and cryo-EM, here a
model's Euclid band against the nearest JWST/NIRCam filter on the NEXUS
field (VIS–F090W, Y–F115W, J–F150W, H–F200W: :data:`JWST_FILTER`).

For two images of the same sky, ``S`` (the model) and ``J`` (JWST) on one
grid, ring by ring in spatial frequency over 128-px round-Hann windows:

* **coherence** ``r(k) = Re<S J*> / sqrt(<|S|²> <|J|²>)`` — 1 where the
  model's structure at that period is JWST's (up to a positive transfer: a
  common blur, or two well-centred PSFs of different widths, leave it at 1;
  a sub-pixel misregistration or invented structure lowers it);
* **gain** ``g(k) = Re<S J*> / <|J|²>`` — how much of JWST's structure at
  that period the model reproduces (1 = all of it, < 1 smoothed away).

The **resolution** is the period where the curve falls to ½ (the FRC
criterion), in pixels of the 0.05″ grid and in arcseconds. JWST's own noise
scales every model's coherence by the same factor at each ring, so model
rankings and differences need no noise correction; an absolute value is
given raw (a lower bound) and corrected with JWST's noise power estimated
from its highest rings (:func:`summary`).

Inputs are matched as far as they can be (:func:`jwst_on_grid`,
:func:`registration_shift`): JWST is anti-aliased and resampled onto the SR
grid from the WCS and converted to the Euclid band's electrons per SR pixel
(:func:`mjysr_to_e`, from the band's electron zeropoint: the colour term
between the two bandpasses is taken as flat); a sub-pixel shift measured
between the Euclid INPUT (so it never depends on a model) and JWST at
Euclid's resolution registers them.
A window needs ``min_coverage`` of its pixels covered by both images
(:class:`RingCorrelation`; the early-release mosaics have thin gaps between
NIRCam detectors) and no saturated star; both images are multiplied by the
SAME apodized coverage mask (zero in the gaps, rising smoothly into the
covered area), so the correlation is that of the covered sky alone — a fully
covered window is unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation

from euclid_polish.config import Config

WINDOW = 128
BORDER = 32
SR_PIXEL_ARCSEC = 0.05
#: The NIRCam filter compared with each Euclid band (the nearest bandpass).
JWST_FILTER = {"VIS": "F090W", "Y_E": "F115W", "J_E": "F150W", "H_E": "F200W"}
#: Euclid PSF FWHM (″) per band, roughly — JWST is blurred to it to register
#: against the input.
EUCLID_FWHM_ARCSEC = {"VIS": 0.18, "Y_E": 0.45, "J_E": 0.45, "H_E": 0.45}
EUCLID_H_FWHM_ARCSEC = EUCLID_FWHM_ARCSEC["H_E"]
#: AB magnitude of 1 MJy/sr over one 0.05″ SR pixel (1e12 µJy/sr × pixel sr).
_MJYSR_PIXEL_AB = Config.AB_ZP_UJY - 2.5 * math.log10(1e12 * (SR_PIXEL_ARCSEC / 206264.806) ** 2)


def mjysr_to_e(band: str) -> float:
    """1 MJy/sr → electrons per 0.05″ SR pixel in Euclid ``band`` (its
    stack's electron zeropoint; 68.4 for H)."""
    return 10.0 ** (0.4 * (Config.get_band(band).sim_zeropoint_e - _MJYSR_PIXEL_AB))


def sb_of_knee(band: str, knee_e: float) -> float:
    """The surface brightness (AB mag arcsec⁻²) of ``knee_e`` electrons per
    0.05″ SR pixel in Euclid ``band``."""
    return (Config.get_band(band).sim_zeropoint_e - 2.5 * math.log10(float(knee_e))
            + 2.5 * math.log10(SR_PIXEL_ARCSEC ** 2))


#: 1 MJy/sr of F200W → electrons per 0.05″ pixel in Euclid H (the colour
#: term between F200W and H taken as flat).
JWST_TO_H_E = mjysr_to_e("H_E")
#: A window holding a JWST pixel above this (e⁻ per SR pixel) has a
#: saturated or near-saturated star and is left out.
STAR_CUT_E = 3.0e4
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _round_hann(window: int) -> np.ndarray:
    yy, xx = np.mgrid[0:window, 0:window]
    rr = np.hypot(yy - (window - 1) / 2.0, xx - (window - 1) / 2.0)
    return np.where(rr < window / 2.0, 0.5 * (1.0 + np.cos(np.pi * rr / (window / 2.0))), 0.0)


_TAPER = _round_hann(WINDOW)
_F = np.fft.fftfreq(WINDOW) * WINDOW
_RING = np.rint(np.hypot(*np.meshgrid(_F, _F, indexing="ij"))).astype(int).ravel()
_RINGS = np.arange(1, WINDOW // 2)
_MEMBER = (_RING[None, :] == _RINGS[:, None]).astype(np.float64).T        # (w², rings)
_RING_PIXELS = _MEMBER.sum(axis=0)                                         # pixels per ring


def period_px() -> np.ndarray:
    """The period (SR px) of each ring, longest first."""
    return WINDOW / _RINGS.astype(np.float64)


def jwst_on_grid(data: np.ndarray, jwst_wcs, sr_wcs, shape: tuple[int, int],
                 jwst_pixel_arcsec: float, to_e: float = JWST_TO_H_E) -> np.ndarray:
    """JWST (MJy/sr) on the SR grid in Euclid e⁻ per SR pixel (``to_e`` per
    MJy/sr; default H): blurred by a Gaussian of half the coarser pixel
    (anti-aliasing), then sampled bilinearly at the SR pixel centres through
    the two WCSs; NaN outside its coverage."""
    sigma = 0.5 * SR_PIXEL_ARCSEC / float(jwst_pixel_arcsec)
    finite = np.isfinite(data)
    smooth = gaussian_filter(np.where(finite, data, 0.0), sigma)
    weight = gaussian_filter(finite.astype(np.float64), sigma)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    jx, jy = jwst_wcs.world_to_pixel_values(*sr_wcs.pixel_to_world_values(xx, yy))
    value = map_coordinates(smooth, [jy, jx], order=1, cval=np.nan)
    cover = map_coordinates(weight, [jy, jx], order=1, cval=0.0)
    out = np.where(cover > 0.99, value / np.maximum(cover, 1e-9), np.nan)
    return out * float(to_e)


def registration_shift(lr_band: np.ndarray, jwst_sr: np.ndarray,
                       fwhm_arcsec: float = EUCLID_H_FWHM_ARCSEC) -> tuple[float, float] | None:
    """``(dy, dx)`` in SR px to apply to the JWST image so it lines up with
    the Euclid input band (upsampled ×2), both at Euclid's resolution
    (``fwhm_arcsec``) and asinh-stretched (so a few bright stars do not
    decide); ``None`` without enough overlap."""
    if not np.isfinite(jwst_sr).mean() > 0.95:
        return None
    up = np.kron(np.nan_to_num(np.asarray(lr_band, np.float64)), np.ones((2, 2))) / 4.0
    up = up[:jwst_sr.shape[0], :jwst_sr.shape[1]]
    sigma = float(fwhm_arcsec) / SR_PIXEL_ARCSEC * _FWHM_TO_SIGMA
    reference = gaussian_filter(np.nan_to_num(jwst_sr), sigma)
    scale = max(float(np.nanstd(up)), 1e-6)
    shift, _error, _phase = phase_cross_correlation(
        np.arcsinh(up / scale), np.arcsinh(reference / scale), upsample_factor=20,
        normalization=None)
    return float(shift[0]), float(shift[1])


def register(jwst_sr: np.ndarray, shift_yx: tuple[float, float]) -> np.ndarray:
    """JWST on the SR grid moved by ``shift_yx`` (cubic spline; NaN kept)."""
    finite = np.isfinite(jwst_sr)
    moved = nd_shift(np.where(finite, jwst_sr, 0.0), shift_yx, order=3, mode="constant", cval=0.0)
    cover = nd_shift(finite.astype(np.float64), shift_yx, order=1, mode="constant", cval=0.0)
    return np.where(cover > 0.99, moved, np.nan)


def _windows(img: np.ndarray) -> np.ndarray:
    img = img[BORDER:-BORDER, BORDER:-BORDER]
    n_y, n_x = img.shape[0] // WINDOW, img.shape[1] // WINDOW
    cut = img[:n_y * WINDOW, :n_x * WINDOW]
    return cut.reshape(n_y, WINDOW, n_x, WINDOW).transpose(0, 2, 1, 3).reshape(-1, WINDOW, WINDOW)


def _fft(win: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """FFT of each window: the (taper × mask)-weighted mean removed, then
    the taper × mask applied (``mask`` 1 = covered; ``None`` = all covered)."""
    weight = _TAPER if mask is None else _TAPER * mask
    win = np.where(weight > 0, win, 0.0)
    win = win - (win * weight).sum(axis=(-2, -1), keepdims=True) / weight.sum(axis=(-2, -1), keepdims=True)
    return np.fft.fft2(win * weight).reshape(win.shape[0], -1)


def _apodized(valid: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """A coverage mask (``(N, w, w)`` booleans) apodized inside the covered
    area: 0 in the gaps, rising over ~``sigma`` px to 1; all ones stay ones."""
    smooth = gaussian_filter(valid.astype(np.float64), (0, sigma, sigma), mode="nearest")
    return np.where(valid, np.clip(2.0 * smooth - 1.0, 0.0, 1.0), 0.0)


class RingCorrelation:
    """Ring sums of ``|S|²``, ``|J|²`` and ``Re S J*`` over every usable
    window of every tile, for one stretch ``fn`` (applied to both); a window
    is usable with ``min_coverage`` of its pixels covered by both images and
    no saturated star."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], min_coverage: float = 1.0) -> None:
        self.fn = fn
        self.min_coverage = float(min_coverage)
        self.ss = np.zeros(_RINGS.size)
        self.jj = np.zeros(_RINGS.size)
        self.sj = np.zeros(_RINGS.size)
        self.windows = 0

    def add(self, sr: np.ndarray, jwst: np.ndarray) -> int:
        """Add one tile (``sr``, ``jwst``: one band, SR grid, e⁻); returns the
        windows used."""
        ws, wj = _windows(np.asarray(sr, np.float64)), _windows(np.asarray(jwst, np.float64))
        valid = np.isfinite(wj) & np.isfinite(ws)
        keep = valid.mean(axis=(1, 2)) >= self.min_coverage
        keep &= np.max(np.where(valid, wj, -np.inf), axis=(1, 2)) < STAR_CUT_E
        if not keep.any():
            return 0
        full = valid[keep].all()
        mask = None if full else _apodized(valid[keep])
        fs = _fft(self.fn(np.where(valid[keep], ws[keep], 0.0)), mask)
        fj = _fft(self.fn(np.where(valid[keep], wj[keep], 0.0)), mask)
        self.ss += (np.abs(fs) ** 2).sum(axis=0) @ _MEMBER
        self.jj += (np.abs(fj) ** 2).sum(axis=0) @ _MEMBER
        self.sj += np.real(fs * np.conj(fj)).sum(axis=0) @ _MEMBER
        self.windows += int(keep.sum())
        return int(keep.sum())


def half_crossing(period: np.ndarray, curve: np.ndarray, level: float = 0.5) -> float | None:
    """The period (going from long to short) where ``curve`` first falls
    below ``level``, interpolated in log period; ``None`` if it never does
    (or starts below it)."""
    order = np.argsort(-period)
    p, c = period[order], curve[order]
    if not np.isfinite(c[0]) or c[0] < level:
        return None
    for i in range(1, len(c)):
        if np.isfinite(c[i]) and c[i] < level:
            t = (c[i - 1] - level) / (c[i - 1] - c[i])
            return float(np.exp(np.log(p[i - 1]) + t * (np.log(p[i]) - np.log(p[i - 1]))))
    return None


def summary(acc: RingCorrelation, *, noise_rings: int = 4) -> dict:
    """Curves and the ½-crossing resolutions of one accumulation. The noise-
    corrected coherence divides by ``sqrt(1 - N/<|J|²>)``, with JWST's noise
    power ``N`` taken as the mean of its ``noise_rings`` highest rings (an
    over-estimate where JWST still holds structure there: conservative)."""
    period = period_px()
    with np.errstate(invalid="ignore", divide="ignore"):
        coherence = acc.sj / np.sqrt(acc.ss * acc.jj)
        gain = acc.sj / acc.jj
        # The rings are sums over ~2πk pixels: estimate N per pixel, scale it per ring.
        noise = float(np.mean(acc.jj[-noise_rings:] / _RING_PIXELS[-noise_rings:]))
        signal = np.clip(acc.jj - noise * _RING_PIXELS, 1e-30, None)
        coherence_c = acc.sj / np.sqrt(acc.ss * signal)
        gain_c = acc.sj / signal

    def rows(a):
        return [round(float(v), 4) if np.isfinite(v) else None for v in a]

    def res(curve):
        p = half_crossing(period, curve)
        return None if p is None else {"px": round(p, 3), "arcsec": round(p * SR_PIXEL_ARCSEC, 4)}

    return {"windows": acc.windows, "period_px": [round(float(v), 3) for v in period],
            "coherence": rows(coherence), "gain": rows(gain),
            "coherence_noise_corrected": rows(np.clip(coherence_c, -1.5, 1.5)),
            "resolution_coherence": res(coherence), "resolution_gain": res(gain),
            "resolution_coherence_noise_corrected": res(coherence_c),
            "resolution_gain_noise_corrected": res(gain_c)}


__all__ = ["EUCLID_FWHM_ARCSEC", "JWST_FILTER", "JWST_TO_H_E", "RingCorrelation", "half_crossing",
           "jwst_on_grid", "mjysr_to_e", "period_px", "register", "registration_shift", "sb_of_knee",
           "summary"]
