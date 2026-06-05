#!/usr/bin/env python
"""Render TNG50 SKIRT-atlas infographics on FASRC, streamed to the WebUI.

Runs on a FASRC login node (invoked over SSH by ``run_remote_python``), where
the downloaded galaxies and the TNG API key both live. Like
``scripts/fasrc_inspect_tile.py`` it writes **binary bytes to stdout** (PNG or
FITS) so the route can ``send_file`` them straight to the browser; all chatter
goes to stderr so the byte stream stays clean.

Galaxies are self-enumerated from the download tree — every
``<tng_dir>/<subhalo_id>/`` that holds a ``.done`` marker (written by
``scripts/fasrc_download_tng_skirt_atlas.py``). Three modes:

  ``--mode histograms``
      2×2 matplotlib panel of SFR / stellar mass / halo mass / effective radius
      for the downloaded galaxies. Per-galaxy properties come from the TNG API
      (``…/subhalos/{id}/info.json`` + the parent halo's ``Group_M_Crit200``)
      and are cached to ``<tng_dir>/tng_properties.csv`` so only new galaxies
      hit the network; ``--max-new`` bounds the queries per call.

  ``--mode grid --band {VIS|Y|J|H|RGB} --downsample {1|2|4} --seed S``
      A 5×5 PNG grid: 5 seeded-random galaxies (rows) × their 5 viewpoints
      (cols). Single Euclid band (asinh-grayscale) or an ``make_lupton_rgb``
      RGB composite from VIS+NISP. ``block_mean`` downsamples ×1/×2/×4.

  ``--mode stack --band B [--id N] [--seed S]``
      Bundle the 5 viewpoint frames of one band for a galaxy (random if no id)
      into one multi-extension FITS (PrimaryHDU + ImageHDU O1..O5) → stdout.

The TNG API uses Planck-2015 cosmology; masses are converted code→Msun
(``×1e10/h``) and lengths code→kpc (``/h`` at z=0). Field names are matched
loosely (raw SubFind ``SubhaloSFR``/``SubhaloMassType_4`` *and* the cleaned
``sfr``/``mass_stars`` forms) so the parse survives either API representation.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")                       # headless, before pyplot
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from euclid_polish.config import Config
from euclid_polish.sky.tng_galaxy import (
    TNG_FITS_BANDS,
    block_mean,
    load_tng_frame,
    tng_fits_path,
)

API_BASE = "https://www.tng-project.org/api/TNG50-1/snapshots/99"
H_LITTLE = 0.6774                            # TNG (Planck15) dimensionless Hubble
ORIENTATIONS: Tuple[int, ...] = (1, 2, 3, 4, 5)
GRID_GALAXIES = 5
# RGB channels as FITS band tokens (R, G, B) = (H, J, VIS), matching
# Config.Color.RGB_SCHEMES["vis_nisp"] = (H_E, J_E, VIS).
RGB_FITS_BANDS: Tuple[str, str, str] = ("H", "J", "VIS")
SINGLE_BANDS = tuple(TNG_FITS_BANDS)         # ("VIS", "Y", "J", "H")
PROPERTIES_CSV = "tng_properties.csv"
_CSV_FIELDS = ("id", "sfr", "mass_stars", "m_halo", "reff")


# ---------------------------------------------------------------------------
# Galaxy enumeration + selection
# ---------------------------------------------------------------------------

def list_downloaded_ids(tng_dir: str) -> List[str]:
    """Subhalo ids that finished downloading (their folder holds a .done)."""
    if not os.path.isdir(tng_dir):
        return []
    ids = [
        name for name in os.listdir(tng_dir)
        if os.path.isfile(os.path.join(tng_dir, name, Config.Tng.DONE_MARKER))
    ]
    try:
        return sorted(ids, key=int)
    except ValueError:
        return sorted(ids)


def pick_ids(ids: List[str], k: int, seed: int) -> List[str]:
    """Deterministic seeded pick of up to ``k`` ids (shuffled if fewer)."""
    rng = random.Random(seed)
    pool = list(ids)
    if not pool:
        return []
    if len(pool) <= k:
        rng.shuffle(pool)
        return pool
    return rng.sample(pool, k)


# ---------------------------------------------------------------------------
# TNG API → per-galaxy properties (cached)
# ---------------------------------------------------------------------------

def load_api_key(path: Optional[str] = None) -> str:
    """Resolve the TNG token: $TNG_API_KEY, else the (expanded) key file."""
    env = os.environ.get("TNG_API_KEY", "").strip()
    if env:
        return env
    p = os.path.expanduser(path or Config.Tng.API_KEY_FILE)
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            return f.readline().strip()
    return ""


def _get_json(url: str, key: str, timeout: int = 10) -> dict:
    import requests
    r = requests.get(url, headers={"api-key": key}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _pick(d: dict, *keys: str):
    """First present, non-null value among ``keys`` (else None)."""
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return None


def _mass_type_stars(d: dict) -> Optional[float]:
    """Stellar mass (code units) from either the flattened ``*_4`` field or a
    ``SubhaloMassType`` list (particle type 4 = stars)."""
    flat = _pick(d, "SubhaloMassType_4", "mass_stars")
    if flat is not None:
        return float(flat)
    lst = _pick(d, "SubhaloMassType", "mass_type")
    if isinstance(lst, (list, tuple)) and len(lst) > 4:
        return float(lst[4])
    return None


def _halfmassrad_stars(d: dict) -> Optional[float]:
    flat = _pick(d, "SubhaloHalfmassRadType_4", "halfmassrad_stars")
    if flat is not None:
        return float(flat)
    lst = _pick(d, "SubhaloHalfmassRadType", "halfmassrad_type")
    if isinstance(lst, (list, tuple)) and len(lst) > 4:
        return float(lst[4])
    return None


def parse_subhalo(sub: dict) -> Dict[str, float]:
    """Map a subhalo info dict → physical SFR / stellar mass / effective radius.

    Masses code→Msun (×1e10/h), lengths code→kpc (/h at z=0). Missing → NaN.
    Halo mass is filled separately (needs the parent-group call).
    """
    out = {"sfr": float("nan"), "mass_stars": float("nan"),
           "m_halo": float("nan"), "reff": float("nan")}
    sfr = _pick(sub, "SubhaloSFR", "sfr")
    if sfr is not None:
        out["sfr"] = float(sfr)
    ms = _mass_type_stars(sub)
    if ms is not None:
        out["mass_stars"] = ms * 1e10 / H_LITTLE
    re = _halfmassrad_stars(sub)
    if re is not None:
        out["reff"] = re / H_LITTLE
    return out


def fetch_properties(subhalo_id: str, key: str, *, timeout: int = 10
                     ) -> Dict[str, float]:
    """Query the TNG API for one galaxy's properties (NaN-filled on failure)."""
    out = {"sfr": float("nan"), "mass_stars": float("nan"),
           "m_halo": float("nan"), "reff": float("nan")}
    try:
        sub = _get_json(f"{API_BASE}/subhalos/{subhalo_id}/info.json",
                        key, timeout=timeout)
    except Exception:
        return out
    out.update(parse_subhalo(sub))
    grnr = _pick(sub, "SubhaloGrNr", "grnr")
    if grnr is not None:
        try:
            halo = _get_json(f"{API_BASE}/halos/{int(grnr)}/info.json",
                             key, timeout=timeout)
            mh = _pick(halo, "Group_M_Crit200", "m_crit200", "mass_crit200",
                       "GroupMass", "mass")
            if mh is not None:
                out["m_halo"] = float(mh) * 1e10 / H_LITTLE
        except Exception:
            pass
    return out


def _read_cache(path: str) -> Dict[str, Dict[str, float]]:
    if not os.path.isfile(path):
        return {}
    rows: Dict[str, Dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gid = str(row.get("id", "")).strip()
            if not gid:
                continue
            rows[gid] = {k: float(row.get(k, "nan") or "nan")
                         for k in _CSV_FIELDS if k != "id"}
    return rows


def _write_cache(path: str, rows: Dict[str, Dict[str, float]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for gid in sorted(rows, key=lambda s: (int(s) if s.isdigit() else 1 << 62, s)):
            r = rows[gid]
            w.writerow({"id": gid, **{k: r.get(k, float("nan"))
                                      for k in _CSV_FIELDS if k != "id"}})
    os.replace(tmp, path)


def gather_properties(tng_dir: str, ids: List[str], key: str, *,
                      max_new: int = 120) -> Dict[str, Dict[str, float]]:
    """Return {id: props} for ``ids``, querying+caching only the missing ones.

    Bounded by ``max_new`` per call so the SSH request stays short; repeated
    renders fill in the rest of the catalogue over time.
    """
    cache_path = os.path.join(tng_dir, PROPERTIES_CSV)
    cache = _read_cache(cache_path)
    missing = [g for g in ids if g not in cache]
    n_query = 0
    if key and missing and max_new > 0:
        for gid in missing[:max_new]:
            cache[gid] = fetch_properties(gid, key)
            n_query += 1
        _write_cache(cache_path, cache)
    sys.stderr.write(
        f"properties: {len(ids)} galaxies, {len(missing)} uncached, "
        f"{n_query} queried this call\n")
    return {g: cache[g] for g in ids if g in cache}


# ---------------------------------------------------------------------------
# Rendering — histograms / grid / stack (each returns bytes)
# ---------------------------------------------------------------------------

def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _placeholder_png(message: str) -> bytes:
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13,
            wrap=True)
    ax.axis("off")
    return _fig_to_png(fig)


def render_histograms(tng_dir: str, *, api_key: str = "",
                      max_new: int = 120) -> bytes:
    ids = list_downloaded_ids(tng_dir)
    if not ids:
        return _placeholder_png("No galaxies downloaded yet.")
    props = gather_properties(tng_dir, ids, api_key, max_new=max_new)
    if not props:
        return _placeholder_png(
            f"{len(ids)} galaxies downloaded — properties not cached yet.\n"
            "Set the TNG API token and re-render.")

    def col(name: str) -> np.ndarray:
        v = np.array([props[g].get(name, np.nan) for g in props], dtype=float)
        return v[np.isfinite(v)]

    panels = [
        ("sfr",        "Star formation rate",  "log$_{10}$ SFR [M$_\\odot$/yr]", True),
        ("mass_stars", "Stellar mass",         "log$_{10}$ M$_\\star$ [M$_\\odot$]", True),
        ("m_halo",     "Halo mass (M$_{200c}$)", "log$_{10}$ M$_{200c}$ [M$_\\odot$]", True),
        ("reff",       "Effective radius",     "R$_{1/2,\\star}$ [kpc]", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    n_used = len(props)
    for ax, (key, title, xlabel, logx) in zip(axes.ravel(), panels):
        vals = col(key)
        if logx:
            vals = vals[vals > 0]
            vals = np.log10(vals) if vals.size else vals
        if vals.size:
            ax.hist(vals, bins=min(30, max(5, vals.size // 3)),
                    color="#3b6ea5", edgecolor="white", linewidth=0.4)
            ax.set_title(f"{title}  (n={vals.size})", fontsize=11)
        else:
            ax.set_title(f"{title}  (no data)", fontsize=11)
            ax.text(0.5, 0.5, "no values", ha="center", va="center",
                    transform=ax.transAxes, color="#999")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("count", fontsize=10)
        ax.grid(alpha=0.25)
    fig.suptitle(f"TNG50-1 downloaded galaxies — {n_used} with properties",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _fig_to_png(fig)


def _grayscale_norm(arr: np.ndarray) -> np.ndarray:
    """Asinh stretch (scale = 90th pct of positive flux) + [0.5, 99.5] clip."""
    d = np.clip(np.asarray(arr, dtype=np.float32), 0.0, None)
    pos = d[d > 0]
    scale = float(np.percentile(pos, 90)) if pos.size else 1.0
    s = np.arcsinh(d / max(scale, 1e-12))
    lo, hi = np.percentile(s, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((s - lo) / (hi - lo), 0.0, 1.0)


def _load_cell_band(gdir: str, gid: str, orient: int, fits_band: str,
                    downsample: int) -> Optional[np.ndarray]:
    path = tng_fits_path(gdir, gid, orient, fits_band)
    if not os.path.isfile(path):
        return None
    arr = load_tng_frame(path)
    if downsample > 1:
        arr = block_mean(arr, downsample)
    return arr


def render_cell(gdir: str, gid: str, orient: int, band: str,
                downsample: int) -> Optional[np.ndarray]:
    """One panel: 2-D [0,1] grayscale, or (H,W,3) uint8 RGB, or None if missing."""
    if band == "RGB":
        chans = []
        for tok in RGB_FITS_BANDS:                 # H, J, VIS
            a = _load_cell_band(gdir, gid, orient, tok, downsample)
            if a is None:
                return None
            chans.append(np.clip(a, 0.0, None))
        r, g, b = chans
        from astropy.visualization import make_lupton_rgb
        ref = float(np.percentile(
            np.concatenate([r.ravel(), g.ravel(), b.ravel()]), 99.5))
        return make_lupton_rgb(r, g, b, Q=8, stretch=max(ref, 1e-12))
    a = _load_cell_band(gdir, gid, orient, band, downsample)
    return None if a is None else _grayscale_norm(a)


def render_grid(tng_dir: str, band: str, downsample: int, seed: int) -> bytes:
    ids = pick_ids(list_downloaded_ids(tng_dir), GRID_GALAXIES, seed)
    if not ids:
        return _placeholder_png("No galaxies downloaded yet.")
    nrows = len(ids)
    fig, axes = plt.subplots(nrows, len(ORIENTATIONS),
                             figsize=(len(ORIENTATIONS) * 2.1, nrows * 2.1),
                             squeeze=False)
    for r, gid in enumerate(ids):
        gdir = os.path.join(tng_dir, gid)
        for c, orient in enumerate(ORIENTATIONS):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            img = render_cell(gdir, gid, orient, band, downsample)
            if img is None:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, color="#bbb")
            elif img.ndim == 2:
                ax.imshow(img, cmap="gray", origin="lower",
                          interpolation="nearest")
            else:
                ax.imshow(img, origin="lower", interpolation="nearest")
            if r == 0:
                ax.set_title(f"O{orient}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"TNG{gid}", fontsize=9)
    fig.suptitle(f"TNG50-1 — {band} — downsample ×{downsample}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _fig_to_png(fig)


def build_stack_hdul(tng_dir: str, gid: str, band: str) -> fits.HDUList:
    """Multi-extension FITS: PrimaryHDU + one ImageHDU (O1..O5) per viewpoint.

    Original pixel data and per-frame headers are preserved verbatim.
    """
    gdir = os.path.join(tng_dir, gid)
    primary = fits.PrimaryHDU()
    primary.header["TNGID"] = (str(gid), "IllustrisTNG TNG50-1 subhalo id")
    primary.header["BAND"] = (band, "Euclid band of the stacked frames")
    primary.header["NORIENT"] = (len(ORIENTATIONS), "number of viewpoints")
    hdul = fits.HDUList([primary])
    for orient in ORIENTATIONS:
        path = tng_fits_path(gdir, gid, orient, band)
        if not os.path.isfile(path):
            continue
        with fits.open(path, memmap=False) as src:
            data = np.asarray(src[0].data)
            hdr = src[0].header.copy(strip=True)
        hdr["EXTNAME"] = f"O{orient}"
        hdr["TNGID"] = str(gid)
        hdr["ORIENT"] = (orient, "SKIRT viewpoint index")
        hdul.append(fits.ImageHDU(data=data, header=hdr, name=f"O{orient}"))
    if len(hdul) == 1:
        raise FileNotFoundError(
            f"no {band} frames found for galaxy {gid} in {gdir}")
    return hdul


def build_stack_bytes(tng_dir: str, gid: str, band: str) -> bytes:
    buf = io.BytesIO()
    build_stack_hdul(tng_dir, gid, band).writeto(buf, overwrite=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True,
                   choices=("histograms", "grid", "stack"))
    p.add_argument("--tng-dir", default=Config.TNG_SKIRT_DIR,
                   help=f"Download root. Default: {Config.TNG_SKIRT_DIR}")
    p.add_argument("--band", default="VIS",
                   help="VIS|Y|J|H|RGB (grid) or VIS|Y|J|H (stack).")
    p.add_argument("--downsample", type=int, default=1, choices=(1, 2, 4))
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the random galaxy pick (grid / random stack).")
    p.add_argument("--id", default="",
                   help="Subhalo id for stack mode (blank → seeded random).")
    p.add_argument("--max-new", type=int, default=120,
                   help="Max new TNG-API galaxy queries per histogram render.")
    p.add_argument("--api-key-file", default=Config.Tng.API_KEY_FILE)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    # Binary stdout — mute every chatty warning so the PNG/FITS stays clean.
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    args = parse_args(argv)

    if args.mode == "histograms":
        key = load_api_key(args.api_key_file)
        out = render_histograms(args.tng_dir, api_key=key, max_new=args.max_new)
    elif args.mode == "grid":
        band = args.band.upper()
        if band not in (*SINGLE_BANDS, "RGB"):
            sys.stderr.write(f"bad band {band!r}\n")
            return 2
        out = render_grid(args.tng_dir, band, args.downsample, args.seed)
    elif args.mode == "stack":
        band = args.band.upper()
        if band not in SINGLE_BANDS:
            sys.stderr.write(f"bad stack band {band!r}\n")
            return 2
        gid = args.id.strip()
        if not gid:
            picked = pick_ids(list_downloaded_ids(args.tng_dir), 1, args.seed)
            if not picked:
                sys.stderr.write("no downloaded galaxies\n")
                return 3
            gid = picked[0]
        try:
            out = build_stack_bytes(args.tng_dir, gid, band)
        except FileNotFoundError as e:
            sys.stderr.write(f"{e}\n")
            return 4
    else:
        sys.stderr.write(f"unknown mode {args.mode!r}\n")
        return 1

    sys.stdout.buffer.write(out)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
