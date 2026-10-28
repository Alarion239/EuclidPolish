"""Flux conservation of a super-resolved image against the Euclid MER
catalogue: the catalogue's own photometry, redone on the SR image.

Two measurements, both checked on the Euclid input first (they must give
back the catalogue there, or the comparison means nothing):

* **VIS Kron total** (``FLUX_DETECTION_TOTAL``): MER's elliptical Kron
  aperture on the native VIS image — semi-major axis ``KRON_RADIUS`` (LR
  pixels), axis ratio ``1 − ELLIPTICITY``, angle ``POSITION_ANGLE + 90°`` from
  the pixel x axis. Redone on the input it returns the catalogue flux to
  ~1e-4, so ``(SR − catalogue) / FLUXERR_DETECTION_TOTAL`` is a direct pull.
* **Aperture fluxes** (``FLUX_<band>_<n>FWHM_APER``): circles of diameter
  ``n × FWHM`` on images PSF-matched to the catalogue's ``FWHM`` column — one
  external FWHM (~1.2″, the same for every band), not the band's own PSF.
  The matching kernel here is a Gaussian (the real one has broader wings), so
  the input reads ~4% (n = 4) to ~17% (n = 1) high; :func:`aperture_correction`
  takes that per band and ``n`` from the input and applies it to every model.
  The SR is smoothed by the same kernel as the input, so a sharper model can
  put more light in a small aperture without breaking conservation — ``n = 4``
  is the conservation test, small ``n`` shows concentration.

Every pull uses the catalogue's own error; a model whose pulls sit near 0
with the input's spread keeps the photometry Euclid reports.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from photutils.aperture import CircularAperture, EllipticalAperture, aperture_photometry
from scipy.ndimage import gaussian_filter

#: Catalogue band tokens in the model's channel order (VIS, Y, J, H).
BANDS = ("vis", "y", "j", "h")
#: Measured MER-mosaic PSF FWHM (arcsec, median of Q1 stars — the paper's
#: imaging table), taken out of the matching kernel.
NATIVE_FWHM_ARCSEC = {"vis": 0.147, "y": 0.363, "j": 0.379, "h": 0.395}
LR_PIXEL_ARCSEC = 0.1
APERTURE_MULTIPLES = (1, 2, 3, 4)
#: Kron ellipse angle = catalogue POSITION_ANGLE + this (degrees, from pixel x).
KRON_ANGLE_OFFSET_DEG = 90.0
#: VIS magnitude bins of the summaries.
MAG_EDGES = (16.0, 20.0, 22.0, 23.5, 25.0)


def match_psf(image: np.ndarray, band: str, target_fwhm_arcsec: float,
              pixel_arcsec: float) -> np.ndarray:
    """``image`` smoothed by a Gaussian taking the band's native PSF to the
    catalogue's matched FWHM."""
    width = np.sqrt(max(target_fwhm_arcsec ** 2 - NATIVE_FWHM_ARCSEC[band] ** 2, 0.0))
    sigma = width / (2.0 * np.sqrt(2.0 * np.log(2.0))) / pixel_arcsec
    return gaussian_filter(np.nan_to_num(np.asarray(image, np.float64)), sigma, mode="nearest")


def kron_flux(image: np.ndarray, x: float, y: float, kron_lr_px: float, ellipticity: float,
              position_angle_deg: float, scale: int = 1) -> float:
    """MER's Kron total in ``image`` (``scale`` image pixels per LR pixel)."""
    a = kron_lr_px * scale
    aperture = EllipticalAperture((x, y), a, a * (1.0 - ellipticity),
                                  np.deg2rad(position_angle_deg + KRON_ANGLE_OFFSET_DEG))
    return float(aperture_photometry(np.nan_to_num(image), aperture, method="exact")["aperture_sum"][0])


def circle_flux(image: np.ndarray, x: float, y: float, diameter_px: float) -> float:
    aperture = CircularAperture((x, y), r=diameter_px / 2.0)
    return float(aperture_photometry(image, aperture, method="exact")["aperture_sum"][0])


def measure(cube: np.ndarray, xy: np.ndarray, rows: Sequence[Mapping[str, float]],
            target_fwhm_arcsec: float, scale: int = 1) -> list[dict[str, float]]:
    """Kron VIS totals and matched-aperture fluxes (electrons) of ``rows`` at
    pixel positions ``xy`` in a ``(H, W, 4)`` cube with ``scale`` pixels per
    LR pixel (1 for the input, 2 for an SR)."""
    pixel = LR_PIXEL_ARCSEC / scale
    out: list[dict[str, float]] = [{} for _ in rows]
    for b, band in enumerate(BANDS):
        plane = np.asarray(cube[..., b], np.float64)
        matched = match_psf(plane, band, target_fwhm_arcsec, pixel)
        for k, (row, (x, y)) in enumerate(zip(rows, xy, strict=True)):
            if band == "vis":
                out[k]["kron_vis"] = kron_flux(plane, x, y, row["kron_radius"], row["ellipticity"],
                                               row["position_angle"], scale)
            for n in APERTURE_MULTIPLES:
                out[k][f"ap{n}_{band}"] = circle_flux(matched, x, y, n * row["fwhm"] / pixel)
    return out


def aperture_correction(lr: np.ndarray, catalogue: np.ndarray) -> float:
    """Median input/catalogue ratio of one aperture (the Gaussian kernel's
    shortfall), from sources measured at S/N ≥ 20."""
    ratio = lr / catalogue
    ratio = ratio[np.isfinite(ratio)]
    return float(np.median(ratio)) if ratio.size else float("nan")


def _mad(values: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def summarize(model: np.ndarray, lr: np.ndarray, catalogue: np.ndarray, error: np.ndarray,
              mag: np.ndarray, correction: float = 1.0) -> dict:
    """Pulls ``(model/correction − catalogue)/error`` and flux ratios against
    the catalogue and the input, overall and per VIS magnitude bin."""
    def block(sel):
        m, l, c, e = model[sel] / correction, lr[sel] / correction, catalogue[sel], error[sel]
        if not m.size:
            return {"n": 0}
        pull = (m - c) / e
        return {"n": int(m.size),
                "pull_median": round(float(np.median(pull)), 3),
                "pull_mad": round(_mad(pull), 3),
                "within_1sigma": round(float(np.mean(np.abs(pull) <= 1)), 3),
                "within_3sigma": round(float(np.mean(np.abs(pull) <= 3)), 3),
                "ratio_catalogue_median": round(float(np.median(m / c)), 4),
                "ratio_input_median": round(float(np.median(m / l)), 4),
                "ratio_input_p16_p84": [round(float(v), 4) for v in np.percentile(m / l, [16, 84])],
                "input_pull_mad": round(_mad((l - c) / e), 3)}

    ok = np.isfinite(model) & np.isfinite(lr) & np.isfinite(catalogue) & (error > 0) & (catalogue > 0)
    bins = []
    for lo, hi in zip(MAG_EDGES[:-1], MAG_EDGES[1:], strict=True):
        bins.append({"mag": [lo, hi], **block(ok & (mag >= lo) & (mag < hi))})
    return {**block(ok), "by_mag": bins}
