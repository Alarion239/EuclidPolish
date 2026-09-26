"""Truth-free SR metrics for REAL tiles (Sky → Experiments, spec §7.3).

Definitions follow the 2026-09-23/25 real-galaxy analysis (memory
``project_combiner_real_galaxy_holes``). All fluxes are electrons; ``lr`` is
``(H, W, C)`` on the LR grid, ``sr`` is ``(f·H, f·W, C)`` on the SR grid
(``f`` = 2 for STARFULL members), bands in ``Config.LR_INPUT_BAND_NAMES``
order.

Per band:

* **hole %** — the brightest 1 % of LR pixels (``LR ≥`` the 99th percentile
  of the finite LR values, ``LR > 0``); each covers ``f²`` SR pixels. A SR
  pixel is a *hole* when ``SR < 0.5 × LR / f²`` (half the LR flux per SR
  pixel). ``hole_pct = 100 × holes / (bright LR pixels × f²)``.
* **enclosed-flux R** — around bright, locally dominant LR peaks: after
  subtracting the robust background (median ``b``; the SR grid subtracts
  ``b / f²``), a peak is an LR pixel ``> 100 σ`` (``σ = 1.4826 · MAD``) that is
  the brightest pixel within ±1.5″. Detector artifacts (cosmic rays, hot
  pixels, which the truth and the members remove) are dropped first by the
  central-pixel fraction ``F(1 px) / F(3×3)``: sources have ≤ 0.25 in VIS and
  ≤ 0.14 in NISP (artifacts ≥ 0.26 / ≥ 0.19). For square boxes of odd side
  0.3–1.7″ (3–17 LR pixels) centred on the peak,
  ``R(box) = F_SR(box) / F_LR(box)`` over the same sky box, and the peak's
  ``R`` is the minimum over the boxes (a core hole shows in the small ones).
  Reported: ``n_peaks``, ``pct_R_lt_0p8``, ``pct_R_lt_0p5``, ``median_R`` (the
  0.066″ target PSF is narrower than Euclid's, so truth has ``R ≥ 1`` for stars
  and galaxies alike — synthetic truth min R = 1.04 over 266 peaks).
* ``hole_pct_100sigma`` — the same over the bright-1 % pixels that are also
  ``> 100 σ`` above the background (on faint tiles the brightest 1 % is
  noise, where "holes" are noise; ``None`` when no pixel qualifies).
* **flux ratio** — ``Σ SR / Σ LR`` over finite pixels.

Undefined values (no peaks, no finite pixels) are ``None`` — the payloads are
strict JSON.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter

from euclid_polish.config import Config

METRICS_VERSION = 1
BRIGHT_PERCENTILE = 99.0
HOLE_FRACTION = 0.5
PEAK_SIGMA = 100.0
DOMINANCE_ARCSEC = 1.5
BOX_ARCSEC = (0.3, 1.7)
#: Central-pixel-fraction ceilings that keep a peak as a real source.
CPF_MAX = {"VIS": 0.25, "NISP": 0.14}
R_THRESHOLDS = (0.8, 0.5)
MAX_REPORTED_PEAKS = 200


def _cpf_limit(band: str) -> float:
    return CPF_MAX["VIS"] if band == "VIS" else CPF_MAX["NISP"]


def _factor(lr: np.ndarray, sr: np.ndarray) -> int:
    if lr.ndim != 3 or sr.ndim != 3 or lr.shape[-1] != sr.shape[-1]:
        raise ValueError(f"expected (H,W,C) LR and SR, got {lr.shape} and {sr.shape}")
    h, w = lr.shape[:2]
    factor = sr.shape[0] // h if h else 0
    if factor < 2 or sr.shape[:2] != (factor * h, factor * w):
        raise ValueError(
            f"SR {sr.shape[:2]} is not an integer (>=2) magnification of LR {lr.shape[:2]}")
    return factor


def _robust_background(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 0.0
    median = float(np.median(finite))
    sigma = 1.4826 * float(np.median(np.abs(finite - median)))
    if sigma <= 0:
        sigma = float(np.std(finite))
    return median, sigma


def _block_sum(sr: np.ndarray, factor: int) -> np.ndarray:
    h, w = sr.shape[0] // factor, sr.shape[1] // factor
    return sr.reshape(h, factor, w, factor).sum(axis=(1, 3))


def _holes(lr: np.ndarray, sr: np.ndarray, factor: int,
           floor: float | None = None) -> tuple[float | None, int]:
    """``(hole %, bright LR pixel count)``; ``floor`` additionally requires
    ``LR > floor`` (the ``> 100 σ`` variant)."""
    finite = np.isfinite(lr)
    if not finite.any():
        return None, 0
    threshold = np.percentile(lr[finite], BRIGHT_PERCENTILE)
    bright = finite & (lr >= threshold) & (lr > 0)
    if floor is not None:
        bright &= lr > floor
    count = int(bright.sum())
    if count == 0:
        return None, 0
    block = np.ones((factor, factor))
    mask = np.kron(bright, block) > 0
    per_sr_pixel = np.kron(np.where(bright, lr, 0.0), block)[mask] / float(factor * factor)
    # A non-finite SR pixel is a hole too (``not >=``).
    holes = ~(sr[mask] >= HOLE_FRACTION * per_sr_pixel)
    return 100.0 * float(holes.sum()) / float(mask.sum()), count


def _peaks(lr_s: np.ndarray, sigma: float, radius: int) -> list[tuple[int, int, float]]:
    if sigma <= 0:
        return []
    field = np.where(np.isfinite(lr_s), lr_s, -np.inf)
    local_max = maximum_filter(field, size=2 * radius + 1, mode="constant", cval=-np.inf)
    ys, xs = np.nonzero((field > PEAK_SIGMA * sigma) & (field >= local_max))
    order = np.argsort(-field[ys, xs], kind="stable")
    accepted: list[tuple[int, int, float]] = []
    for index in order:
        y, x = int(ys[index]), int(xs[index])
        if any(max(abs(y - ay), abs(x - ax)) <= radius for ay, ax, _v in accepted):
            continue                                  # plateau twin
        accepted.append((y, x, float(field[y, x])))
    return accepted


def band_metrics(lr: np.ndarray, sr: np.ndarray, band: str, *,
                 factor: int, lr_pixel_arcsec: float = Config.VIS_PIXEL_SCALE_ARCSEC
                 ) -> tuple[dict[str, Any], list[list[float]]]:
    """Metrics of one band (2-D ``lr``/``sr``) + the per-peak rows
    ``[x, y, peak_e, cpf, R]`` (LR pixels, 0-based)."""
    lr = np.asarray(lr, np.float64)
    sr = np.asarray(sr, np.float64)
    hole_pct, n_bright = _holes(lr, sr, factor)
    lr_sum = float(np.nansum(lr[np.isfinite(lr)]))
    sr_sum = float(np.nansum(sr[np.isfinite(sr)]))
    flux_ratio = sr_sum / lr_sum if lr_sum != 0 else None

    background, sigma = _robust_background(lr)
    # On faint tiles the brightest 1 % is noise, where a hole is just noise;
    # this variant keeps only those pixels that are also > 100 σ.
    hole_bright, n_bright_sigma = _holes(lr, sr, factor,
                                         floor=background + PEAK_SIGMA * sigma)
    lr_s = lr - background
    sr_s = sr - background / float(factor * factor)
    radius = max(1, int(round(DOMINANCE_ARCSEC / lr_pixel_arcsec)))
    sides = [s for s in range(3, 64, 2)
             if BOX_ARCSEC[0] - 1e-9 <= s * lr_pixel_arcsec <= BOX_ARCSEC[1] + 1e-9]
    half = max(sides) // 2 if sides else 1
    sr_blocks = _block_sum(sr_s, factor)
    limit = _cpf_limit(band)
    rows: list[list[float]] = []
    n_artifacts = n_edge = 0
    h, w = lr.shape
    for y, x, value in _peaks(lr_s, sigma, radius):
        if y - half < 0 or x - half < 0 or y + half >= h or x + half >= w:
            n_edge += 1
            continue
        core = lr_s[y - 1:y + 2, x - 1:x + 2]
        core_sum = float(np.sum(core))
        cpf = value / core_sum if core_sum > 0 else np.inf
        if not np.isfinite(cpf) or cpf > limit:
            n_artifacts += 1
            continue
        ratios = []
        for side in sides:
            k = side // 2
            f_lr = float(np.sum(lr_s[y - k:y + k + 1, x - k:x + k + 1]))
            f_sr = float(np.sum(sr_blocks[y - k:y + k + 1, x - k:x + k + 1]))
            if not (np.isfinite(f_lr) and np.isfinite(f_sr)) or f_lr <= 0:
                ratios = []
                break
            ratios.append(f_sr / f_lr)
        if not ratios:
            n_edge += 1
            continue
        rows.append([float(x), float(y), value, float(cpf), float(min(ratios))])
    r_values = np.asarray([row[4] for row in rows], np.float64)
    n = int(r_values.size)
    metrics: dict[str, Any] = {
        "hole_pct": hole_pct,
        "n_bright_px": n_bright,
        "hole_pct_100sigma": hole_bright,
        "n_bright_100sigma_px": n_bright_sigma,
        "flux_ratio": flux_ratio,
        "lr_flux_e": lr_sum,
        "sr_flux_e": sr_sum,
        "background_e": background,
        "sigma_e": sigma,
        "n_peaks": n,
        "n_artifacts": n_artifacts,
        "n_edge": n_edge,
        "pct_R_lt_0p8": 100.0 * float(np.mean(r_values < 0.8)) if n else None,
        "pct_R_lt_0p5": 100.0 * float(np.mean(r_values < 0.5)) if n else None,
        "median_R": float(np.median(r_values)) if n else None,
        "min_R": float(np.min(r_values)) if n else None,
    }
    return metrics, rows


def tile_metrics(lr: np.ndarray, sr: np.ndarray,
                 band_names: Sequence[str] = Config.LR_INPUT_BAND_NAMES, *,
                 lr_pixel_arcsec: float = Config.VIS_PIXEL_SCALE_ARCSEC) -> dict[str, Any]:
    """Every band's metrics for one (tile, model): ``{version, factor, bands,
    per_band{band: {...}}, peaks{band: [[x, y, peak_e, cpf, R], …]}, summary}``."""
    lr = np.asarray(lr, np.float32)
    sr = np.asarray(sr, np.float32)
    factor = _factor(lr, sr)
    bands = list(band_names)[:lr.shape[-1]]
    per_band: dict[str, Any] = {}
    peaks: dict[str, list[list[float]]] = {}
    for index, band in enumerate(bands):
        metrics, rows = band_metrics(lr[..., index], sr[..., index], band,
                                     factor=factor, lr_pixel_arcsec=lr_pixel_arcsec)
        per_band[band] = metrics
        peaks[band] = rows[:MAX_REPORTED_PEAKS]
    return {
        "version": METRICS_VERSION,
        "factor": factor,
        "bands": bands,
        "per_band": per_band,
        "peaks": peaks,
        "summary": _summary(per_band, peaks),
    }


def _summary(per_band: Mapping[str, Mapping[str, Any]],
             peaks: Mapping[str, Sequence[Sequence[float]]]) -> dict[str, Any]:
    r_all = np.asarray([row[4] for rows in peaks.values() for row in rows], np.float64)
    holes = [m["hole_pct"] for m in per_band.values() if m.get("hole_pct") is not None]
    n = int(sum(int(m.get("n_peaks") or 0) for m in per_band.values()))
    return {
        "n_peaks": n,
        "hole_pct_max": max(holes) if holes else None,
        "hole_pct_mean": float(np.mean(holes)) if holes else None,
        "pct_R_lt_0p8": 100.0 * float(np.mean(r_all < 0.8)) if r_all.size else None,
        "pct_R_lt_0p5": 100.0 * float(np.mean(r_all < 0.5)) if r_all.size else None,
        "median_R": float(np.median(r_all)) if r_all.size else None,
    }


def aggregate(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool per-tile metrics of one model: holes weighted by bright pixels,
    R statistics over the union of peaks, flux ratio from summed fluxes."""
    items = [item for item in results if item]
    bands: list[str] = []
    for item in items:
        for band in item.get("bands", []):
            if band not in bands:
                bands.append(band)
    per_band: dict[str, Any] = {}
    pooled_peaks: dict[str, list[list[float]]] = {}
    for band in bands:
        metrics = [item["per_band"][band] for item in items if band in item.get("per_band", {})]
        rows = [row for item in items for row in item.get("peaks", {}).get(band, [])]
        weight = sum(int(m.get("n_bright_px") or 0) for m in metrics)
        holes = sum(float(m["hole_pct"]) * int(m.get("n_bright_px") or 0)
                    for m in metrics if m.get("hole_pct") is not None)
        weight_sigma = sum(int(m.get("n_bright_100sigma_px") or 0) for m in metrics)
        holes_sigma = sum(float(m["hole_pct_100sigma"]) * int(m.get("n_bright_100sigma_px") or 0)
                          for m in metrics if m.get("hole_pct_100sigma") is not None)
        lr_flux = sum(float(m.get("lr_flux_e") or 0.0) for m in metrics)
        sr_flux = sum(float(m.get("sr_flux_e") or 0.0) for m in metrics)
        r_values = np.asarray([row[4] for row in rows], np.float64)
        n = int(r_values.size)
        per_band[band] = {
            "hole_pct": holes / weight if weight else None,
            "n_bright_px": weight,
            "hole_pct_100sigma": holes_sigma / weight_sigma if weight_sigma else None,
            "n_bright_100sigma_px": weight_sigma,
            "flux_ratio": sr_flux / lr_flux if lr_flux else None,
            "n_peaks": n,
            "n_artifacts": sum(int(m.get("n_artifacts") or 0) for m in metrics),
            "pct_R_lt_0p8": 100.0 * float(np.mean(r_values < 0.8)) if n else None,
            "pct_R_lt_0p5": 100.0 * float(np.mean(r_values < 0.5)) if n else None,
            "median_R": float(np.median(r_values)) if n else None,
            "min_R": float(np.min(r_values)) if n else None,
        }
        pooled_peaks[band] = rows
    return {
        "version": METRICS_VERSION,
        "n_tiles": len(items),
        "bands": bands,
        "per_band": per_band,
        "summary": _summary(per_band, pooled_peaks),
    }


def definitions() -> dict[str, Any]:
    """Machine-readable definition constants (echoed in experiment records)."""
    return {
        "version": METRICS_VERSION,
        "bright_percentile": BRIGHT_PERCENTILE,
        "hole_fraction_of_lr_per_sr_pixel": HOLE_FRACTION,
        "peak_sigma": PEAK_SIGMA,
        "dominance_arcsec": DOMINANCE_ARCSEC,
        "box_arcsec": list(BOX_ARCSEC),
        "cpf_max": dict(CPF_MAX),
        "r_thresholds": list(R_THRESHOLDS),
    }


__all__ = [
    "METRICS_VERSION",
    "aggregate",
    "band_metrics",
    "definitions",
    "tile_metrics",
]
