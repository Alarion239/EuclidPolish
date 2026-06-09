#!/usr/bin/env python
"""Measure the REAL saturation/clip level of bright stars in the downloaded
MER-mosaic cutouts — per band, in electrons-over-the-stack (the exact units the
model's dirty image uses).

For each star in ``stars.csv`` (brightest first) and each band it:
  1. loads the cutout, reads ``MAGZERO`` from the header and converts ADU/s →
     electrons (``e⁻ = pix · 10**((sim_zeropoint_e − MAGZERO)/2.5)`` — the same
     calibration the model input uses);
  2. measures, in a central window: the **peak** pixel (e⁻), the **flat-top**
     size (pixels within 1% of the peak — a saturated core clips to a plateau),
     and any **non-finite** (NaN / MER-flagged) core pixels.

It aggregates by VIS magnitude so you can read off the **empirical saturation
ceiling** per band (the value to clip synthetic saturation to) and whether MER
**masks** saturated cores (NaN). Reports only — changes nothing.

Imports ONLY ``euclid_polish.config`` (no astroquery / network), so it runs on an
offline compute node. Run on FASRC where the cutouts live:

    python scripts/measure_star_saturation.py --n 600 --size 255
    python scripts/measure_star_saturation.py --output-dir $DATA_DIR/euclid_stars
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config            # config-only: no astroquery

_FNAME_RE = re.compile(r"star_(\d+)_(\d+)\.fits$")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _adu_to_e_factor(magzero: float, band) -> float:
    """Archive ADU/s → electrons-over-stack (inlined photometry, no imports)."""
    return 10.0 ** ((band.sim_zeropoint_e - float(magzero)) / 2.5)


def load_cutout_electrons(path: str, band) -> Tuple[np.ndarray, float]:
    """Read a star cutout → electrons-over-stack (via the header MAGZERO)."""
    with fits.open(path, memmap=False) as hdul:
        arr = np.asarray(hdul[0].data, dtype=np.float64)
        magzero = float(hdul[0].header.get("MAGZERO", band.sim_zeropoint_e))
    return arr * _adu_to_e_factor(magzero, band), magzero


def measure_core_saturation(img_e: np.ndarray, *, window_px: int = 51,
                            flat_frac: float = 0.99) -> Dict[str, float]:
    """Peak / flat-top size / non-finite count in the central window."""
    H, W = img_e.shape
    cy, cx = H // 2, W // 2
    half = min(window_px // 2, cy, cx)
    core = img_e[cy - half:cy + half + 1, cx - half:cx + half + 1]
    finite = np.isfinite(core)
    if not finite.any():
        return {"peak_e": float("nan"), "central_e": float("nan"),
                "flattop_px": 0, "nan_core_px": int((~finite).sum())}
    peak = float(np.nanmax(core))
    flattop = int(np.nansum(core >= flat_frac * peak)) if peak > 0 else 0
    central = float(img_e[cy, cx]) if np.isfinite(img_e[cy, cx]) else float("nan")
    return {"peak_e": peak, "central_e": central,
            "flattop_px": flattop, "nan_core_px": int((~finite).sum())}


def _index_band_dir(band_dir: str, size: int) -> Dict[int, str]:
    """One ``os.listdir`` → ``{star_id: path}`` (exact ``size``, else largest).

    Avoids a per-star ``glob`` over a 10k+-file directory on netscratch."""
    out: Dict[int, Tuple[int, str]] = {}
    try:
        names = os.listdir(band_dir)
    except OSError:
        return {}
    for name in names:
        m = _FNAME_RE.search(name)
        if not m:
            continue
        sid, sz = int(m.group(1)), int(m.group(2))
        rank = (1 << 30) if sz == size else sz      # exact size always wins
        prev = out.get(sid)
        if prev is None or rank > prev[0]:
            out[sid] = (rank, os.path.join(band_dir, name))
    return {sid: p for sid, (_, p) in out.items()}


def scan_stars(stars_csv: str, output_dir: str, *, size: int, n: int,
               window_px: int = 51) -> Dict[str, List[dict]]:
    """Per band → list of {mag, peak_e, central_e, flattop_px, nan_core_px}."""
    cutouts_root = os.path.join(output_dir, Config.CUTOUTS_SUBDIR)
    bands = [Config.get_band(bn) for bn in Config.LR_INPUT_BAND_NAMES]
    with open(stars_csv, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("magnitude")]
    rows.sort(key=lambda r: float(r["magnitude"]))      # brightest first
    rows = rows[:n]

    # Index each band's directory ONCE (not per star).
    index: Dict[str, Dict[int, str]] = {}
    for b in bands:
        bd = Config.cutout_dir_for_band(b.name, root=cutouts_root)
        index[b.name] = _index_band_dir(bd, size)
        _log(f"  indexed {b.name}: {len(index[b.name])} cutouts under {bd}")

    out: Dict[str, List[dict]] = {b.name: [] for b in bands}
    for i, row in enumerate(rows):
        try:
            sid, mag = int(row["id"]), float(row["magnitude"])
        except (KeyError, TypeError, ValueError):
            continue
        for b in bands:
            path = index[b.name].get(sid)
            if path is None:
                continue
            try:
                img_e, _ = load_cutout_electrons(path, b)
            except Exception:
                continue
            rec = measure_core_saturation(img_e, window_px=window_px)
            rec["mag"] = mag
            out[b.name].append(rec)
        if (i + 1) % 100 == 0:
            _log(f"  scanned {i + 1}/{len(rows)} stars...")
    return out


def model_clip_e(band) -> float:
    """The model's current well-depth clip for ``band`` (inlined to avoid the
    sky.saturation import, which pulls astroquery)."""
    sigma = Config.STAR_SATURATION_FWHM_ARCSEC / 2.3548200450309493
    pix = getattr(band, "native_detector_scale_arcsec", None) or band.pixel_scale_lr_arcsec
    f_peak = math.erf(pix / (2.0 * math.sqrt(2.0) * sigma)) ** 2
    calib = Config.STAR_SATURATION_CALIB_MAG[band.name]
    offset = Config.STAR_BAND_OFFSETS_MAG.get(band.name, 0.0)
    e_total = 10.0 ** (-0.4 * (calib + offset - band.sim_zeropoint_e))
    return e_total * f_peak


_MAG_BINS = [(-99, 13), (13, 15), (15, 16), (16, 17), (17, 18), (18, 20), (20, 99)]


def _summarize_band(recs: List[dict]) -> None:
    if not recs:
        print("    (no cutouts found)")
        return
    mags = np.array([r["mag"] for r in recs])
    peaks = np.array([r["peak_e"] for r in recs])
    flats = np.array([r["flattop_px"] for r in recs])
    nans = np.array([r["nan_core_px"] for r in recs])
    print(f"    {'mag bin':>9} {'n':>4} {'med_peak_e':>12} {'max_peak_e':>12} "
          f"{'med_flat_px':>11} {'nan_core%':>9}")
    for lo, hi in _MAG_BINS:
        m = (mags >= lo) & (mags < hi)
        if not m.any():
            continue
        pk = peaks[m][np.isfinite(peaks[m])]
        med_pk = np.median(pk) if pk.size else float("nan")
        max_pk = np.max(pk) if pk.size else float("nan")
        label = f"{lo if lo > -99 else ''}-{hi if hi < 99 else ''}"
        print(f"    {label:>9} {int(m.sum()):>4} {med_pk:12.4g} {max_pk:12.4g} "
              f"{np.median(flats[m]):11.1f} {100.0*np.mean(nans[m] > 0):8.0f}%")
    bright = peaks[(mags < 15) & np.isfinite(peaks)]
    ceiling = np.median(bright) if bright.size else float("nan")
    print(f"    → empirical ceiling (median peak, mag<15): {ceiling:.4g} e⁻   "
          f"| global max peak: {np.nanmax(peaks):.4g} e⁻   "
          f"| NaN-core stars: {100.0*np.mean(nans > 0):.0f}%")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=Config.DEFAULT_OUTPUT_DIR,
                    help="Star root (holds stars.csv + cutouts/).")
    ap.add_argument("--stars-csv", default=None,
                    help="Defaults to <output-dir>/stars.csv.")
    ap.add_argument("--n", type=int, default=600,
                    help="Number of (brightest) stars to scan.")
    ap.add_argument("--size", type=int, default=Config.DEFAULT_CUTOUT_SIZE,
                    help="Cutout size to read (falls back to any available).")
    ap.add_argument("--window", type=int, default=51,
                    help="Central window (px) for the peak/flat-top measurement.")
    args = ap.parse_args()

    stars_csv = args.stars_csv or os.path.join(args.output_dir, Config.CATALOG_FILE)
    _log(f"Scanning up to {args.n} brightest stars from {stars_csv}")
    _log("Electron scale = electrons-over-stack (model dirty-image units).")
    data = scan_stars(stars_csv, args.output_dir, size=args.size, n=args.n,
                      window_px=args.window)
    for bn in Config.LR_INPUT_BAND_NAMES:
        band = Config.get_band(bn)
        print(f"\n=== {bn}  (model clip = {model_clip_e(band):.4g} e⁻) ===",
              flush=True)
        _summarize_band(data[bn])
    print("\nInterpret: 'empirical ceiling' is the real saturation level to clip "
          "synthetic\nsaturation to (per band, same e⁻-over-stack units). A high "
          "'nan_core%' for\nbright stars means MER masks saturated cores → the "
          "synthetic model should set\nNaN there instead of a flat value.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
