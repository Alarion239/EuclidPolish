"""TNG50 galaxy properties (from the TNG API) + histogram plotting.

The histogram infographic is generated **locally** in the WebUI: the caller
supplies the downloaded-galaxy id list (pulled from FASRC), and this module
fetches the physical properties from the public TNG API and draws the plot —
no FITS pixels are involved, so nothing heavy is transferred.

Per galaxy we make ONE request to the cleaned subhalo endpoint
``/subhalos/{id}/?format=json`` (~0.4 s). It already carries everything we
need as flat scalars:

  * ``sfr``               → star-formation rate [M☉/yr]
  * ``mass_stars``        → stellar mass (code units → ×1e10/h M☉)
  * ``halfmassrad_stars`` → effective radius (code units → /h kpc)
  * ``mass_log_msun``     → total bound mass, already log₁₀ M☉ (used as the
                            "halo mass" proxy — for the atlas's central
                            galaxies it tracks the FoF halo mass, and avoids a
                            separate group call)

Why not the obvious alternatives (measured against the live API, TNG50-1 z=0,
5.7M subhalos): ``/subhalos/{id}/info.json`` does heavy server-side field
flattening and takes ~15 s each; the group-catalog field-subset endpoint
``files/groupcat-99/?Subhalo=<field>`` 504-times-out (the server can't build a
field over 5.7M objects in its gateway window); and the full group catalog is
~50 GB across 680 chunks. So a parallel pass of the fast per-galaxy endpoint —
cached to ``tng_properties.csv`` — is the practical path: ~30 s for the whole
atlas, then instant.
"""

from __future__ import annotations

import csv
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")                       # headless, before pyplot
import matplotlib.pyplot as plt
import numpy as np

from euclid_polish.config import Config

API_BASE = "https://www.tng-project.org/api/TNG50-1/snapshots/99"
H_LITTLE = 0.6774                            # TNG (Planck15) dimensionless Hubble
PROPERTIES_CSV = "tng_properties.csv"
_CSV_FIELDS = ("id", "sfr", "mass_stars", "m_halo", "reff")
_NAN_PROPS = {"sfr": float("nan"), "mass_stars": float("nan"),
              "m_halo": float("nan"), "reff": float("nan")}
DEFAULT_WORKERS = 16                         # concurrent TNG-API requests


# ---------------------------------------------------------------------------
# API key
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


# ---------------------------------------------------------------------------
# Per-galaxy property fetch (one fast ?format=json request)
# ---------------------------------------------------------------------------

def _get_json(url: str, key: str, timeout: int = 30) -> dict:
    import requests
    # requests follows the API's 302 and keeps the custom ``api-key`` header
    # across the redirect (only Authorization is stripped on host change).
    r = requests.get(url, headers={"api-key": key}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _pick(d: dict, *keys: str):
    """First present, non-null value among ``keys`` (else None)."""
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return None


def parse_subhalo(sub: dict) -> Dict[str, float]:
    """Map a ``/subhalos/{id}/?format=json`` dict → physical properties.

    Masses: code units → M☉ (×1e10/h); ``mass_log_msun`` is already log₁₀ M☉.
    Lengths: code units → kpc (/h at z=0). Missing → NaN. "Halo mass" uses the
    subhalo's total bound mass (``mass_log_msun``), which for central galaxies
    tracks the FoF halo mass — getting the true Group_M_Crit200 would need a
    separate ~15 s call per galaxy.
    """
    out = dict(_NAN_PROPS)
    sfr = _pick(sub, "sfr", "SubhaloSFR")
    if sfr is not None:
        out["sfr"] = float(sfr)
    ms = _pick(sub, "mass_stars", "SubhaloMassType_4")
    if ms is not None:
        out["mass_stars"] = float(ms) * 1e10 / H_LITTLE
    re = _pick(sub, "halfmassrad_stars", "SubhaloHalfmassRadType_4")
    if re is not None:
        out["reff"] = float(re) / H_LITTLE
    mlog = _pick(sub, "mass_log_msun")
    if mlog is not None:
        out["m_halo"] = 10.0 ** float(mlog)
    else:                                   # fallback: total mass in code units
        mtot = _pick(sub, "mass", "SubhaloMass")
        if mtot is not None:
            out["m_halo"] = float(mtot) * 1e10 / H_LITTLE
    return out


def fetch_properties(subhalo_id: str, key: str, *, timeout: int = 30
                     ) -> Dict[str, float]:
    """One TNG-API call for a galaxy's properties (NaN-filled on failure)."""
    try:
        sub = _get_json(f"{API_BASE}/subhalos/{subhalo_id}/?format=json",
                        key, timeout=timeout)
    except Exception:
        return dict(_NAN_PROPS)
    return parse_subhalo(sub)


# ---------------------------------------------------------------------------
# Cache (per-galaxy CSV)
# ---------------------------------------------------------------------------

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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for gid in sorted(rows, key=lambda s: (int(s) if s.isdigit() else 1 << 62, s)):
            r = rows[gid]
            w.writerow({"id": gid, **{k: r.get(k, float("nan"))
                                      for k in _CSV_FIELDS if k != "id"}})
    os.replace(tmp, path)


def gather_properties(work_dir: str, ids: List[str], key: str, *,
                      max_workers: int = DEFAULT_WORKERS, reporter=None
                      ) -> Dict[str, Dict[str, float]]:
    """Return {id: props} for ``ids``, caching to ``work_dir/tng_properties.csv``.

    Only missing galaxies are fetched, and they're fetched **concurrently** (the
    TNG API is ~0.4 s/request, so ~30 s for the whole atlas; cached afterwards).
    ``reporter`` (optional) drives a live progress bar.
    """
    cache_path = os.path.join(work_dir, PROPERTIES_CSV)
    cache = _read_cache(cache_path)
    missing = [g for g in ids if g not in cache]
    if key and missing:
        if reporter is not None:
            reporter.set_stage(f"querying TNG API ({len(missing)} galaxies)")
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            futs = {pool.submit(fetch_properties, g, key): g for g in missing}
            for fut in as_completed(futs):
                gid = futs[fut]
                try:
                    cache[gid] = fut.result()
                except Exception:
                    cache[gid] = dict(_NAN_PROPS)
                done += 1
                if reporter is not None:
                    reporter.set_step(done, len(missing), f"TNG{gid}")
        _write_cache(cache_path, cache)
    n_have = sum(1 for g in ids if g in cache)
    sys.stderr.write(f"properties: {len(ids)} galaxies, {n_have} resolved\n")
    return {g: cache[g] for g in ids if g in cache}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def placeholder_png(message: str) -> bytes:
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13, wrap=True)
    ax.axis("off")
    return _fig_to_png(fig)


def plot_histograms(props: Dict[str, Dict[str, float]]) -> bytes:
    """2×2 PNG: SFR / stellar mass / total mass / effective radius."""
    if not props:
        return placeholder_png("No galaxy properties to plot.")

    def col(name: str) -> np.ndarray:
        v = np.array([props[g].get(name, np.nan) for g in props], dtype=float)
        return v[np.isfinite(v)]

    panels = [
        ("sfr",        "Star formation rate",  "log$_{10}$ SFR [M$_\\odot$/yr]", True),
        ("mass_stars", "Stellar mass",         "log$_{10}$ M$_\\star$ [M$_\\odot$]", True),
        ("m_halo",     "Total mass (bound)",   "log$_{10}$ M [M$_\\odot$]", True),
        ("reff",       "Effective radius",     "R$_{1/2,\\star}$ [kpc]", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
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
    fig.suptitle(f"TNG50-1 downloaded galaxies — {len(props)} with properties",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _fig_to_png(fig)


def render_histograms_for_ids(work_dir: str, ids: List[str], key: str, *,
                              max_workers: int = DEFAULT_WORKERS,
                              reporter=None) -> bytes:
    """End-to-end: ids → properties (cached, concurrent) → 2×2 histogram PNG."""
    if not ids:
        return placeholder_png("No galaxies downloaded yet.")
    props = gather_properties(work_dir, ids, key,
                              max_workers=max_workers, reporter=reporter)
    if not props:
        return placeholder_png(
            f"{len(ids)} galaxies downloaded — properties not available yet.\n"
            "Set the TNG API token, then re-render.")
    return plot_histograms(props)
