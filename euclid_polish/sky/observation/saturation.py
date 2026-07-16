"""Detector saturation masking for the synthetic LR (dirty) image.

Any source bright enough to drive a detector pixel past the full well —
bright stars AND bright galaxy nuclei — saturates. In the Euclid (MER)
pipeline the saturated pixels are *masked*: a rectangular patch around the
source is set to a fill value (≈0), NOT recorded at the clipped well level.
This module reproduces that on the forward-modelled LR image: it finds the
pixels that exceed each band's well depth and zeros a blocky rectangular
patch around them, so nothing in the dirty image sits above the physical
well and the masked region reads ~0.

Well depth (the saturation level, in electrons on the shared 0.10″ LR stack):

* **VIS** — McCracken et al. (2025), Euclid Q1 VIS processing (OU-VIS):
  blooming sets in at 40–61 kADU (mean 51 kADU) PER READOUT at a Q1 gain of
  3.48 e⁻/ADU → 177 480 e⁻. We work in electrons (McCracken delivers ADU) and
  refer the well to the co-added 4-exposure stack our forward model produces:
  ``4 × 177 480 ≈ 709 920 e⁻`` (``Config.STAR_SATURATION_WELL_E["VIS"]``). That
  reproduces the known VIS saturation magnitude m_AB ≈ 17.8.
* **NISP** (Y_E/J_E/H_E) — effective stack-referred clip levels for the
  4×87.2 s MACC integration (the physical H2RG well is larger), calibrated via
  ``scripts/measure_star_saturation.py``.

Shape: the masked region is each saturated component's bounding box (so no
pixel survives above the well) unioned with 1–3 overlapping rectangles (sides
``STAR_SATURATION_RECT_{MIN,MAX}_PX`` px) at its peak, zeroed (flat blocky
mask, ≈0 fill on the sky-subtracted grid).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import find_objects, label

from euclid_polish.config import BandConfig, Config


class StarSaturationModel:
    """Per-band saturation well depth + rectangular mask shape.

    Despite the legacy name the model is source-agnostic:
    :func:`apply_saturation_masking` masks any pixel — star core or galaxy
    nucleus — that reaches the band well depth.
    """

    def __init__(
        self,
        *,
        well_e: dict[str, float] | None = None,
        rect_min_px: int = Config.STAR_SATURATION_RECT_MIN_PX,
        rect_max_px: int = Config.STAR_SATURATION_RECT_MAX_PX,
        max_rects: int = Config.STAR_SATURATION_MAX_RECTS,
    ):
        self._well_e = dict(well_e if well_e is not None
                            else Config.STAR_SATURATION_WELL_E)
        self.rect_min_px = int(rect_min_px)
        self.rect_max_px = int(rect_max_px)
        self.max_rects = int(max_rects)
        if not (0 < self.rect_min_px <= self.rect_max_px):
            raise ValueError("require 0 < rect_min_px ≤ rect_max_px")
        if self.max_rects < 1:
            raise ValueError("max_rects must be ≥ 1")

    # ------------------------------------------------------------------ #
    def well_depth_e(self, band: BandConfig) -> float:
        """Saturation level (electrons) for ``band`` — a pixel at/above this
        saturates and is masked."""
        return self._well_e[band.name]

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
    trigger_4ch: np.ndarray | None = None,
) -> None:
    """Zero a blocky rectangular patch over every saturated scene region.

    ``trigger_4ch`` is the pre-noise, pre-artifact optical scene when supplied;
    the blackout is still written into the final dirty ``lr_4ch``.  Separating
    those arrays prevents hot pixels and cosmic rays from being expanded into
    stellar-sized blackout rectangles merely because their injected charge is
    above a low NISP effective well.  The trigger remains source-agnostic:
    bright stars and bright galaxy nuclei alike are masked.  Omitting
    ``trigger_4ch`` retains the direct-array behaviour for standalone callers.

    For each band:

    1. label connected regions in the trigger at ``>= well_depth``;
    2. zero each region's bounding box (so nothing stays above the well), then
    3. zero a union of 1–3 small overlapping rectangles
       (:meth:`StarSaturationModel.rectangles`, the current 3–6 px scale) at
       the region's peak, for the characteristic blocky mask edge.

    The masked patch reads ~0 (the MER fill value on the sky-subtracted LR
    grid), so the network must inpaint the source from its surroundings. The
    clean HR target is untouched."""
    trigger = lr_4ch if trigger_4ch is None else np.asarray(trigger_4ch)
    if trigger.shape != lr_4ch.shape:
        raise ValueError(
            f"trigger_4ch shape {trigger.shape} must match lr_4ch shape "
            f"{lr_4ch.shape}"
        )
    H, W = lr_4ch.shape[:2]
    for k, bn in enumerate(band_names):
        well = np.float32(model.well_depth_e(Config.get_band(bn)))
        ch = lr_4ch[..., k]                      # view → writes propagate
        trigger_ch = trigger[..., k]
        sat = trigger_ch >= well
        if not sat.any():
            continue
        labelled, n_regions = label(sat)
        if n_regions == 0:
            continue
        for sl in find_objects(labelled):
            if sl is None:
                continue
            ys, xs = sl                          # bounding-box slices
            sub = trigger_ch[ys, xs]
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
