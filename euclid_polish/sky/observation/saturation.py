"""Bright-star detector saturation for the synthetic LR (dirty) image.

Bright stars in Euclid frames have *saturated* cores: once a star's peak
detector pixel exceeds the well depth, the charge clips and blooms into a small
blocky region. This module applies that to the forward-modelled LR image.

Physics (derived from the project zeropoints):

* A star of VIS magnitude ``m`` deposits ``E = 10^(-0.4·(m + band_offset −
  ZP_e))`` electrons over the stack in a given band (``ZP_e`` = that band's
  stack zeropoint). Its **peak detector pixel** holds ``E · f_peak``, where
  ``f_peak = erf(p / (2√2·σ))²`` is the central-pixel fraction of an effective
  Gaussian of FWHM ``STAR_SATURATION_FWHM_ARCSEC`` at the band's native pixel
  ``p`` (VIS 0.10″, NISP 0.30″).
* The per-band **well depth** ``S`` is calibrated so the peak reaches ``S`` at
  the 50%-saturation magnitudes (``STAR_SATURATION_CALIB_MAG``: VIS≈14,
  NISP≈17). This gives a VIS well of ~197k e⁻ (CCD full well) and effective
  stack-referred NISP clip levels of ~4.6–9.5k e⁻ (scaling with the
  4×87.2 s MACC integration; the physical H2RG well is larger).
* A star **saturates** a band when ``Poisson(peak) ≥ S``. A per-star log-normal
  jitter on ``peak`` (sub-pixel position + PSF variation,
  ``STAR_SATURATION_JITTER_DEX``) spreads the onset into a smooth ~1-mag
  transition, so ``P(saturate) = ½`` at the calibration magnitude is a genuine
  probability. **Drawn independently per band.**

Shape: the saturated region is the union of 1–3 overlapping rectangles (sides
``STAR_SATURATION_RECT_{MIN,MAX}_PX`` px), the first containing the peak pixel
and the rest offset uniformly but overlapping, clipped to ``S`` (flat-topped
core + blocky bloom).

The value-preserving NISP→VIS-LR resample makes the native well depth ``S`` the
correct clip level on the shared 0.10″ LR grid, where saturation is applied.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from euclid_polish.config import BandConfig, Config


def _f_peak(pixel_arcsec: float, fwhm_arcsec: float) -> float:
    """Central-pixel flux fraction of a unit-flux 2-D Gaussian PSF."""
    sigma = fwhm_arcsec / 2.3548200450309493        # FWHM → σ
    return math.erf(pixel_arcsec / (2.0 * math.sqrt(2.0) * sigma)) ** 2


def _native_pixel(band: BandConfig) -> float:
    """The band's native detector pixel scale (NISP 0.30″, VIS 0.10″)."""
    return getattr(band, "native_detector_scale_arcsec", None) or band.pixel_scale_lr_arcsec


class StarSaturationModel:
    """Per-(star, band) saturation onset + rectangular saturated-region shape."""

    def __init__(
        self,
        *,
        bands: Optional[Sequence[BandConfig]] = None,
        band_offsets: Optional[Dict[str, float]] = None,
        calib_mag: Optional[Dict[str, float]] = None,
        fwhm_arcsec: float = Config.STAR_SATURATION_FWHM_ARCSEC,
        jitter_dex: float = Config.STAR_SATURATION_JITTER_DEX,
        rect_min_px: int = Config.STAR_SATURATION_RECT_MIN_PX,
        rect_max_px: int = Config.STAR_SATURATION_RECT_MAX_PX,
        max_rects: int = Config.STAR_SATURATION_MAX_RECTS,
    ):
        self.bands = list(bands) if bands is not None else list(Config.BANDS)
        self.band_offsets = dict(band_offsets if band_offsets is not None
                                 else Config.STAR_BAND_OFFSETS_MAG)
        self.calib_mag = dict(calib_mag if calib_mag is not None
                              else Config.STAR_SATURATION_CALIB_MAG)
        self.fwhm_arcsec = float(fwhm_arcsec)
        self.jitter_dex = float(jitter_dex)
        self.rect_min_px = int(rect_min_px)
        self.rect_max_px = int(rect_max_px)
        self.max_rects = int(max_rects)
        if not (0 < self.rect_min_px <= self.rect_max_px):
            raise ValueError("require 0 < rect_min_px ≤ rect_max_px")
        if self.max_rects < 1:
            raise ValueError("max_rects must be ≥ 1")

        # Precompute per-band peak fraction + calibrated well depth (electrons).
        self._f_peak: Dict[str, float] = {}
        self._well_e: Dict[str, float] = {}
        for b in self.bands:
            self._f_peak[b.name] = _f_peak(_native_pixel(b), self.fwhm_arcsec)
            self._well_e[b.name] = self._peak_mean_e(self.calib_mag[b.name], b)

    # ------------------------------------------------------------------ #
    def _peak_mean_e(self, m_vis: float, band: BandConfig) -> float:
        """Mean peak-pixel electrons for a VIS-mag-``m_vis`` star in ``band``."""
        m_band = m_vis + self.band_offsets.get(band.name, 0.0)
        e_total = 10.0 ** (-0.4 * (m_band - band.sim_zeropoint_e))
        return e_total * self._f_peak[band.name]

    def well_depth_e(self, band: BandConfig) -> float:
        """Calibrated saturation/clip level (electrons) for ``band``."""
        return self._well_e[band.name]

    def saturation_probability(self, m_vis: float, band: BandConfig) -> float:
        """``P(saturate)`` (jitter-dominated; Poisson width negligible here)."""
        mean = self._peak_mean_e(m_vis, band)
        if mean <= 0.0:
            return 0.0
        if self.jitter_dex <= 0.0:
            return 1.0 if mean >= self._well_e[band.name] else 0.0
        # P(mean·10^N(0,σ) ≥ S) = Φ(−log10(S/mean)/σ).
        z = math.log10(self._well_e[band.name] / mean) / self.jitter_dex
        return 0.5 * math.erfc(z / math.sqrt(2.0))

    def saturates(self, m_vis: float, band: BandConfig,
                  rng: np.random.Generator) -> bool:
        """Draw whether a star saturates ``band`` (independent per band)."""
        mean = self._peak_mean_e(m_vis, band)
        if mean <= 0.0:
            return False
        if self.jitter_dex > 0.0:
            mean *= 10.0 ** float(rng.normal(0.0, self.jitter_dex))
        well = self._well_e[band.name]
        # Poisson realisation of the peak count (skip for huge means — then it is
        # deterministically saturated and numpy.poisson would be slow/overflow).
        peak = float(rng.poisson(mean)) if mean < 1e9 else mean
        return peak >= well

    def rectangles(self, rng: np.random.Generator
                   ) -> List[Tuple[int, int, int, int]]:
        """1–3 overlapping rectangles ``(x0, y0, w, h)`` (pixels, relative to the
        peak pixel at the origin). The first contains the origin; the rest are
        placed to overlap it."""
        lo, hi = self.rect_min_px, self.rect_max_px + 1
        w = int(rng.integers(lo, hi))
        h = int(rng.integers(lo, hi))
        x0 = int(rng.integers(-(w - 1), 1))          # origin ∈ [x0, x0+w-1]
        y0 = int(rng.integers(-(h - 1), 1))
        rects = [(x0, y0, w, h)]
        bx0, by0, bw, bh = rects[0]
        for _ in range(int(rng.integers(1, self.max_rects + 1)) - 1):
            rw = int(rng.integers(lo, hi))
            rh = int(rng.integers(lo, hi))
            # x-range [nx0, nx0+rw-1] must intersect [bx0, bx0+bw-1] → overlap.
            nx0 = int(rng.integers(bx0 - (rw - 1), bx0 + bw))
            ny0 = int(rng.integers(by0 - (rh - 1), by0 + bh))
            rects.append((nx0, ny0, rw, rh))
        return rects


def apply_star_saturation(
    lr_4ch: np.ndarray,
    stars: Sequence[dict],
    model: StarSaturationModel,
    rng: np.random.Generator,
    *,
    hr_to_lr_scale: float,
    band_names: Sequence[str],
) -> None:
    """Stamp saturated rectangles onto the LR dirty image in place.

    For every star, each band independently draws ``model.saturates``; if it
    does, the rectangle-union mask (centred on the star's LR pixel) is clipped to
    that band's well depth.
    ``hr_to_lr_scale`` maps the star's HR pixel position to the LR grid."""
    H, W = lr_4ch.shape[:2]
    for star in stars:
        try:
            cx = int(round(float(star["x_pix"]) * hr_to_lr_scale))
            cy = int(round(float(star["y_pix"]) * hr_to_lr_scale))
            m_vis = float(star["mag_vis"])
        except (KeyError, TypeError, ValueError):
            continue
        for k, bn in enumerate(band_names):
            band = Config.get_band(bn)
            if not model.saturates(m_vis, band, rng):
                continue
            well = np.float32(model.well_depth_e(band))
            for (x0, y0, w, h) in model.rectangles(rng):
                i0, j0 = cy + y0, cx + x0
                ii0, ii1 = max(0, i0), min(H, i0 + h)
                jj0, jj1 = max(0, j0), min(W, j0 + w)
                if ii0 < ii1 and jj0 < jj1:
                    lr_4ch[ii0:ii1, jj0:jj1, k] = well
