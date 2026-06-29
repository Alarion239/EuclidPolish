"""Detector saturation masking for the synthetic LR (dirty) image.

Any source bright enough to drive a detector pixel past the full well —
bright stars AND bright galaxy nuclei — saturates. In the Euclid (MER)
pipeline the saturated pixels are *masked*: a rectangular patch around the
source is set to a fill value (≈0), NOT recorded at the clipped well level.
This module reproduces that on the forward-modelled LR image: it finds the
pixels that exceed each band's well depth and zeros a blocky rectangular
patch around them, so nothing in the dirty image sits above the physical
well and the masked region reads ~0.

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
from collections.abc import Sequence

import numpy as np
from scipy.ndimage import find_objects, label

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
        bands: Sequence[BandConfig] | None = None,
        band_offsets: dict[str, float] | None = None,
        calib_mag: dict[str, float] | None = None,
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
        self._f_peak: dict[str, float] = {}
        self._well_e: dict[str, float] = {}
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
                   ) -> list[tuple[int, int, int, int]]:
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


def apply_saturation_masking(
    lr_4ch: np.ndarray,
    model: StarSaturationModel,
    rng: np.random.Generator,
    *,
    band_names: Sequence[str],
) -> None:
    """Zero a blocky rectangular patch over every saturated region (in place).

    The trigger is the rendered pixel value, not the source type: any pixel at
    or above a band's well depth is saturated, so **bright stars and bright
    galaxy nuclei alike** are masked. For each band:

    1. label the connected regions of pixels ``>= well_depth``;
    2. zero each region's bounding box (so nothing stays above the well), then
    3. zero a union of 1–3 small overlapping rectangles
       (:meth:`StarSaturationModel.rectangles`, the current 3–6 px scale) at
       the region's peak, for the characteristic blocky mask edge.

    The masked patch reads ~0 (the MER fill value on the sky-subtracted LR
    grid), so the network must inpaint the source from its surroundings. The
    clean HR target is untouched."""
    H, W = lr_4ch.shape[:2]
    for k, bn in enumerate(band_names):
        well = np.float32(model.well_depth_e(Config.get_band(bn)))
        ch = lr_4ch[..., k]                      # view → writes propagate
        sat = ch >= well
        if not sat.any():
            continue
        labelled, n_regions = label(sat)
        if n_regions == 0:
            continue
        for sl in find_objects(labelled):
            if sl is None:
                continue
            ys, xs = sl                          # bounding-box slices
            sub = ch[ys, xs]
            pj, pi = np.unravel_index(int(np.argmax(sub)), sub.shape)
            cy, cx = ys.start + int(pj), xs.start + int(pi)
            # Zero the whole bounding box → no pixel survives above the well.
            ch[ys, xs] = 0.0
            # Blocky border: 1–3 small rectangles around the saturated peak.
            for (x0, y0, w, h) in model.rectangles(rng):
                i0, j0 = cy + y0, cx + x0
                ii0, ii1 = max(0, i0), min(H, i0 + h)
                jj0, jj1 = max(0, j0), min(W, j0 + w)
                if ii0 < ii1 and jj0 < jj1:
                    ch[ii0:ii1, jj0:jj1] = 0.0
