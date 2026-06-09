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


def _index_dir(band_dir: str, size: int) -> Dict[int, str]:
    """One ``os.listdir`` → ``{star_id: path}`` (exact ``size``, else largest)."""
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


def _discover_band_dirs(output_dir: str, bands, size: int
                        ) -> Dict[str, Tuple[str, Dict[int, str]]]:
    """Resolve each band → (dir, {sid: path}). Tries the canonical
    ``<output_dir>/cutouts/<band>`` first; if any band's dir is empty, falls back
    to a bounded walk that finds *any* directory holding ``star_<id>_<size>.fits``
    and maps it to a band by the dir name (band name, archive filter, or prefix).
    """
    cutouts_root = os.path.join(output_dir, Config.CUTOUTS_SUBDIR)
    result: Dict[str, Tuple[str, Dict[int, str]]] = {}
    need_walk = False
    for b in bands:
        d = Config.cutout_dir_for_band(b.name, root=cutouts_root)
        idx = _index_dir(d, size)
        result[b.name] = (d, idx)
        need_walk = need_walk or not idx
    if not need_walk:
        return result

    _log(f"  canonical dirs empty for some bands — walking {output_dir} ...")
    discovered: Dict[str, Tuple[str, Dict[int, str]]] = {}
    base_depth = output_dir.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(output_dir):
        if root.count(os.sep) - base_depth > 4:
            dirs[:] = []
            continue
        idx: Dict[int, Tuple[int, str]] = {}
        for name in files:
            m = _FNAME_RE.search(name)
            if not m:
                continue
            sid, sz = int(m.group(1)), int(m.group(2))
            rank = (1 << 30) if sz == size else sz
            prev = idx.get(sid)
            if prev is None or rank > prev[0]:
                idx[sid] = (rank, os.path.join(root, name))
        if idx:
            discovered[os.path.basename(root)] = (
                root, {s: p for s, (_, p) in idx.items()})
    if discovered:
        print("Discovered star-cutout directories (name: count):", flush=True)
        for name, (d, idx) in sorted(discovered.items()):
            print(f"  {name}: {len(idx)}  ({d})", flush=True)
    else:
        print(f"NO star_<id>_<size>.fits found anywhere under {output_dir}",
              flush=True)
    # map each band to a discovered dir
    for b in bands:
        if result[b.name][1]:
            continue
        for key in (b.name, getattr(b, "archive_filter", "") or "",
                    b.name.split("_")[0]):
            if key and key in discovered:
                result[b.name] = discovered[key]
                break
    return result


def _mag_by_id(stars_csv: str) -> Dict[int, float]:
    """``{star_id: VIS magnitude}`` for catalog rows with a finite magnitude."""
    out: Dict[int, float] = {}
    try:
        with open(stars_csv, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    mid, mg = int(r["id"]), float(r["magnitude"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(mg):
                    out[mid] = mg
    except OSError:
        pass
    return out


def scan_stars(stars_csv: str, output_dir: str, *, size: int, n: int,
               window_px: int = 51) -> Dict[str, List[dict]]:
    """Per band → list of {mag, peak_e, central_e, flattop_px, nan_core_px}.

    Driven by the cutout FILES on disk (robust to a desynced ``stars.csv``):
    scan up to ``n`` cutouts per band — brightest first when a catalog magnitude
    is available, else by id — annotating each with its catalog magnitude if the
    id matches."""
    bands = [Config.get_band(bn) for bn in Config.LR_INPUT_BAND_NAMES]
    mag_by_id = _mag_by_id(stars_csv)
    print(f"  catalog: {len(mag_by_id)} stars with a finite magnitude", flush=True)
    resolved = _discover_band_dirs(output_dir, bands, size)

    out: Dict[str, List[dict]] = {b.name: [] for b in bands}
    for b in bands:
        d, index = resolved[b.name]
        with_mag = sorted((s for s in index if s in mag_by_id),
                          key=lambda s: mag_by_id[s])          # brightest first
        without = sorted(s for s in index if s not in mag_by_id)
        order = (with_mag + without)[:n]
        print(f"  {b.name:4s} -> {len(index):6d} cutouts ({len(with_mag)} w/ mag); "
              f"scanning {len(order)}   {d}", flush=True)
        for i, sid in enumerate(order):
            try:
                img_e, _ = load_cutout_electrons(index[sid], b)
            except Exception:
                continue
            rec = measure_core_saturation(img_e, window_px=window_px)
            rec["mag"] = mag_by_id.get(sid, float("nan"))
            out[b.name].append(rec)
            if (i + 1) % 200 == 0:
                _log(f"    {b.name} {i + 1}/{len(order)}...")
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
    mags = np.array([r["mag"] for r in recs], dtype=float)
    peaks = np.array([r["peak_e"] for r in recs], dtype=float)
    flats = np.array([r["flattop_px"] for r in recs], dtype=float)
    nans = np.array([r["nan_core_px"] for r in recs], dtype=float)
    fin = np.isfinite(peaks)
    pk = peaks[fin]
    if pk.size == 0:
        print("    (all peaks non-finite)")
        return

    # Peak-electron distribution (works with or without catalog magnitudes).
    qs = {q: np.percentile(pk, q) for q in (50, 90, 99, 100)}
    print(f"    scanned {len(recs)} cutouts | peak e⁻  "
          f"P50={qs[50]:.4g}  P90={qs[90]:.4g}  P99={qs[99]:.4g}  "
          f"max={qs[100]:.4g}")
    # The brightest cutouts plateau at the saturation level → ceiling.
    k = max(5, pk.size // 20)                 # top ~5%
    top_idx = np.argsort(peaks)[-k:]
    print(f"    → SATURATION CEILING (median of top ~5% peaks): "
          f"{np.median(peaks[top_idx]):.4g} e⁻")
    print(f"      top-peak cutouts: median flat-top = {np.median(flats[top_idx]):.0f} px, "
          f"NaN-core = {100.0*np.mean(nans[top_idx] > 0):.0f}%")

    # Magnitude-binned view, only if the catalog magnitudes matched.
    if np.isfinite(mags).sum() >= 10:
        print(f"    by VIS magnitude: {'mag bin':>9} {'n':>4} {'med_peak_e':>12} "
              f"{'med_flat_px':>11} {'nan_core%':>9}")
        for lo, hi in _MAG_BINS:
            m = np.isfinite(mags) & (mags >= lo) & (mags < hi) & fin
            if not m.any():
                continue
            label = f"{lo if lo > -99 else ''}-{hi if hi < 99 else ''}"
            print(f"      {label:>16} {int(m.sum()):>4} "
                  f"{np.median(peaks[m]):12.4g} {np.median(flats[m]):11.1f} "
                  f"{100.0*np.mean(nans[m] > 0):8.0f}%")


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
