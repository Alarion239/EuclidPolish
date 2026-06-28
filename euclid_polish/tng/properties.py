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

import matplotlib
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

def load_api_key(path: str | None = None) -> str:
    """Resolve the TNG token: $TNG_API_KEY, else the (expanded) key file."""
    env = os.environ.get("TNG_API_KEY", "").strip()
    if env:
        return env
    p = os.path.expanduser(path or Config.Tng.API_KEY_FILE)
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
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


def parse_subhalo(sub: dict) -> dict[str, float]:
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
    # Total bound mass. Prefer the *documented* raw field ``mass`` (SubhaloMass,
    # code units 1e10 M☉/h) over the handy-but-undocumented ``mass_log_msun``;
    # both give the same physical value (verified mass*1e10/h == 10**log).
    mtot = _pick(sub, "mass", "SubhaloMass")
    if mtot is not None:
        out["m_halo"] = float(mtot) * 1e10 / H_LITTLE
    else:
        mlog = _pick(sub, "mass_log_msun")
        if mlog is not None:
            out["m_halo"] = 10.0 ** float(mlog)
    return out


def fetch_properties(subhalo_id: str, key: str, *, timeout: int = 30
                     ) -> dict[str, float]:
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

def _read_cache(path: str) -> dict[str, dict[str, float]]:
    if not os.path.isfile(path):
        return {}
    rows: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gid = str(row.get("id", "")).strip()
            if not gid:
                continue
            rows[gid] = {k: float(row.get(k, "nan") or "nan")
                         for k in _CSV_FIELDS if k != "id"}
    return rows


def _is_failed_row(props: dict[str, float]) -> bool:
    """True for a cache row left by a *fully-failed* fetch (every property NaN).

    A transient API timeout / rate-limit during the concurrent sweep caches an
    all-NaN row; without re-querying it would stay NaN forever, since the cache
    otherwise only fetches ids it has never seen. A *quenched* galaxy (SFR == 0
    but masses/radius present) is NOT failed — it is real data and kept as-is.
    """
    vals = [props.get(k, float("nan")) for k in _CSV_FIELDS if k != "id"]
    return not any(np.isfinite(v) for v in vals)


def _write_cache(path: str, rows: dict[str, dict[str, float]]) -> None:
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


def gather_properties(work_dir: str, ids: list[str], key: str, *,
                      max_workers: int = DEFAULT_WORKERS, reporter=None
                      ) -> dict[str, dict[str, float]]:
    """Return {id: props} for ``ids``, caching to ``work_dir/tng_properties.csv``.

    Only missing galaxies are fetched, and they're fetched **concurrently** (the
    TNG API is ~0.4 s/request, so ~30 s for the whole atlas; cached afterwards).
    ``reporter`` (optional) drives a live progress bar.
    """
    cache_path = os.path.join(work_dir, PROPERTIES_CSV)
    cache = _read_cache(cache_path)
    # Re-query ids never seen AND ids whose cached row is an all-NaN fetch
    # failure (so a transient API error during the first sweep is retried,
    # not frozen forever). Resolved rows — including quenched SFR==0 — are kept.
    missing = [g for g in ids if g not in cache or _is_failed_row(cache[g])]
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
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def placeholder_png(message: str) -> bytes:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13, wrap=True)
    ax.axis("off")
    return _fig_to_png(fig)


def sfr_population(props: dict[str, dict[str, float]]):
    """Split the SFR column into ``(log10 star-forming, n_quenched, n_missing)``.

    ``n_quenched`` = galaxies with SFR **== 0** exactly (quenched / gas-poor —
    real data, common for the atlas's massive z=0 galaxies, not a fetch miss).
    ``n_missing`` = NaN (fetch never resolved). The log-SFR histogram can only
    show the star-forming ones, so the other two are surfaced as counts rather
    than silently dropped by ``log10`` / the finite filter.
    """
    v = np.array([props[g].get("sfr", np.nan) for g in props], dtype=float)
    finite = v[np.isfinite(v)]
    pos = finite[finite > 0]
    n_quenched = int(np.count_nonzero(finite == 0))
    n_missing = int(np.count_nonzero(~np.isfinite(v)))
    return (np.log10(pos) if pos.size else pos), n_quenched, n_missing


def _plot_sfr_panel(ax, props: dict[str, dict[str, float]],
                    title: str, xlabel: str) -> None:
    """SFR panel: log-histogram of star-forming galaxies + an explicit
    'quenched (SFR=0)' bar, so the panel reconciles with the download count
    instead of silently hiding the SFR==0 population under ``log10``."""
    logv, n_quenched, n_missing = sfr_population(props)
    if logv.size:
        ax.hist(logv, bins=min(30, max(5, logv.size // 3)),
                color="#3b6ea5", edgecolor="white", linewidth=0.4,
                label=f"star-forming (n={logv.size})")
    if n_quenched:
        # A distinct red bar just left of the star-forming distribution.
        if logv.size:
            lo, hi = float(logv.min()), float(logv.max())
            span = (hi - lo) or 1.0
            xq, width = lo - 0.13 * span, 0.07 * span
        else:
            xq, width = -1.0, 0.2
        ax.bar([xq], [n_quenched], width=width, color="#b03a3a",
               edgecolor="white", linewidth=0.4, align="center",
               label=f"quenched, SFR=0 (n={n_quenched})")
    bits = []
    if logv.size:
        bits.append(f"{logv.size} SF")
    if n_quenched:
        bits.append(f"{n_quenched} quenched")
    if n_missing:
        bits.append(f"{n_missing} no data")
    ax.set_title(f"{title}  ({', '.join(bits) or 'no data'})", fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("count", fontsize=10)
    ax.grid(alpha=0.25)
    if logv.size or n_quenched:
        ax.legend(fontsize=8, loc="upper right")


def plot_histograms(props: dict[str, dict[str, float]]) -> bytes:
    """2×2 PNG: SFR / stellar mass / total mass / effective radius."""
    if not props:
        return placeholder_png("No galaxy properties to plot.")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    for ax, (key, title, xlabel, logx) in zip(axes.ravel(), panels, strict=False):
        if key == "sfr":
            _plot_sfr_panel(ax, props, title, xlabel)
            continue
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
    fig.suptitle(f"TNG50-1 downloaded galaxies — n={len(props)}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _fig_to_png(fig)


def render_histograms_for_ids(work_dir: str, ids: list[str], key: str, *,
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
