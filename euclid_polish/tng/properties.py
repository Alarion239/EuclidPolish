"""TNG50 galaxy properties (from the TNG API) + histogram plotting.

The histogram infographic is generated **locally** in the WebUI: the caller
supplies the downloaded-galaxy id list (pulled from FASRC), and this module
fetches the physical properties from the public TNG API and draws the plot —
no FITS pixels are involved, so nothing heavy is transferred.

Properties come from the group-catalog **field-subset** endpoint: each request
returns one field as an array over ALL subhalos/groups (indexed by id), and the
API allows only one field per request, so five bulk requests
(SubhaloSFR / SubhaloMassType / SubhaloHalfmassRadType / SubhaloGrNr /
Group_M_Crit200) resolve the entire atlas regardless of galaxy count. The
snapshot is static, so the downloaded arrays — and the derived per-galaxy CSV —
cache forever. A per-galaxy ``info.json`` path is kept only as a fallback.

Masses are converted code→Msun (``×1e10/h``) and lengths code→kpc (``/h`` at
z=0), matching the raw SubFind units these fields carry.
"""

from __future__ import annotations

import csv
import io
import os
import sys
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

GROUPCAT_SUBDIR = "_groupcat"
_GROUPCAT_URL = "https://www.tng-project.org/api/TNG50-1/files/groupcat-99/"
_FIELD_REQUESTS = (
    ("Subhalo", "SubhaloSFR"),
    ("Subhalo", "SubhaloMassType"),
    ("Subhalo", "SubhaloHalfmassRadType"),
    ("Subhalo", "SubhaloGrNr"),
    ("Group",   "Group_M_Crit200"),
)


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
# Per-galaxy info.json fallback
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Bulk group-catalog fetch (5 requests for the whole atlas)
# ---------------------------------------------------------------------------

def _read_field_array(path: str, field: str) -> np.ndarray:
    """Read a field array from a groupcat-subset download (HDF5, else numpy).

    The subset HDF5 layout isn't contractually fixed, so we walk it and pick the
    dataset whose basename matches ``field`` (else the largest dataset)."""
    import h5py
    try:
        found: Dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as f:
            f.visititems(lambda n, o: found.__setitem__(n, o[()])
                         if isinstance(o, h5py.Dataset) else None)
        if found:
            for name, arr in found.items():
                if name.split("/")[-1] == field:
                    return np.asarray(arr)
            return np.asarray(max(found.values(),
                                  key=lambda a: getattr(a, "size", 0)))
    except (OSError, ValueError):
        pass
    return np.asarray(np.load(path, allow_pickle=False))


def _download_field(objtype: str, field: str, key: str,
                    cache_dir: str) -> np.ndarray:
    import requests
    path = os.path.join(cache_dir, f"{objtype}_{field}.hdf5")
    if not (os.path.isfile(path) and os.path.getsize(path) > 0):
        url = f"{_GROUPCAT_URL}?{objtype}={field}"
        r = requests.get(url, headers={"api-key": key}, timeout=180)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return _read_field_array(path, field)


def bulk_fetch_fields(key: str, cache_dir: str, *, reporter=None
                      ) -> Dict[str, np.ndarray]:
    """Download (once, cached) the per-field group-catalog arrays — 5 requests
    total, covering every subhalo/group in the snapshot."""
    os.makedirs(cache_dir, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    for i, (objtype, field) in enumerate(_FIELD_REQUESTS, 1):
        if reporter is not None:
            reporter.set_step(i, len(_FIELD_REQUESTS), field)
        arrays[field] = _download_field(objtype, field, key, cache_dir)
    return arrays


def _row_stars(row) -> float:
    """Particle-type-4 (stars) element of a MassType / HalfmassRadType row."""
    a = np.asarray(row)
    return float(a[..., 4]) if (a.ndim >= 1 and a.shape[-1] > 4) else float(a)


def properties_from_arrays(arrays: Dict[str, np.ndarray],
                           ids: List[str]) -> Dict[str, Dict[str, float]]:
    """Index the bulk field arrays by subhalo id → physical properties (masses
    code→Msun ×1e10/h, lengths code→kpc /h). Halo mass via SubhaloGrNr →
    Group_M_Crit200."""
    sfr  = arrays.get("SubhaloSFR")
    mt   = arrays.get("SubhaloMassType")
    hr   = arrays.get("SubhaloHalfmassRadType")
    grnr = arrays.get("SubhaloGrNr")
    m200 = arrays.get("Group_M_Crit200")
    props: Dict[str, Dict[str, float]] = {}
    for gid in ids:
        try:
            i = int(gid)
        except (TypeError, ValueError):
            continue
        p = {"sfr": float("nan"), "mass_stars": float("nan"),
             "m_halo": float("nan"), "reff": float("nan")}
        if sfr is not None and 0 <= i < len(sfr):
            p["sfr"] = float(sfr[i])
        if mt is not None and 0 <= i < len(mt):
            p["mass_stars"] = _row_stars(mt[i]) * 1e10 / H_LITTLE
        if hr is not None and 0 <= i < len(hr):
            p["reff"] = _row_stars(hr[i]) / H_LITTLE
        if grnr is not None and m200 is not None and 0 <= i < len(grnr):
            g = int(grnr[i])
            if 0 <= g < len(m200):
                p["m_halo"] = float(m200[g]) * 1e10 / H_LITTLE
        props[gid] = p
    return props


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
                      max_new: int = 120, reporter=None
                      ) -> Dict[str, Dict[str, float]]:
    """Return {id: props} for ``ids``, caching to ``work_dir/tng_properties.csv``.

    Preferred path: 5 bulk group-catalog requests resolve EVERY missing galaxy
    at once (independent of count). If that fails, fall back to per-galaxy
    info.json queries bounded by ``max_new``. ``reporter`` (optional) drives a
    live progress bar.
    """
    cache_path = os.path.join(work_dir, PROPERTIES_CSV)
    cache = _read_cache(cache_path)
    missing = [g for g in ids if g not in cache]
    if key and missing:
        try:
            if reporter is not None:
                reporter.set_stage("fetching TNG group catalog (5 bulk requests)")
            arrays = bulk_fetch_fields(
                key, os.path.join(work_dir, GROUPCAT_SUBDIR), reporter=reporter)
            cache.update(properties_from_arrays(arrays, missing))
            _write_cache(cache_path, cache)
            missing = []
        except Exception as e:                       # noqa: BLE001 — degrade
            sys.stderr.write(
                f"bulk groupcat fetch failed ({type(e).__name__}: {e}); "
                "falling back to per-galaxy queries\n")
            if reporter is not None:
                reporter.set_stage("querying TNG API (per-galaxy fallback)")
            todo = missing[:max_new] if max_new > 0 else []
            for n, gid in enumerate(todo, 1):
                cache[gid] = fetch_properties(gid, key)
                if reporter is not None:
                    reporter.set_step(n, len(todo), f"TNG{gid}")
            if todo:
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
    """2×2 PNG: SFR / stellar mass / halo mass / effective radius."""
    if not props:
        return placeholder_png("No galaxy properties to plot.")

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
                              max_new: int = 120, reporter=None) -> bytes:
    """End-to-end: ids → properties (cached) → 2×2 histogram PNG."""
    if not ids:
        return placeholder_png("No galaxies downloaded yet.")
    props = gather_properties(work_dir, ids, key,
                              max_new=max_new, reporter=reporter)
    if not props:
        return placeholder_png(
            f"{len(ids)} galaxies downloaded — properties not available yet.\n"
            "Set the TNG API token, then re-render.")
    return plot_histograms(props)
