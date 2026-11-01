"""Test 3 — resolution against JWST and the P8–16 texture, on the NEXUS tiles
(``scripts/jwst_resolution.py``, run here on the SR cache: one inference per
tile serves all four bands).

Per band: the Fourier ring correlation of the model's band with the nearest
NIRCam filter in an asinh stretch with the knee at the Euclid band's per-SR-pixel
noise (``asinh_euclid``); the period where the coherence falls to ½ is the
resolution (``1σ raw``, arcsec), with the noise-corrected value and a knee scan
(0.3σ … 3σ) beside it. **P8–16** is the mean over periods 8 ≤ p < 16 SR px of
``(gain / coherence)²`` — power the model puts there beyond what JWST
confirms (fake texture). Compared with the first model as the mean over the
four bands of the resolution ratio, in "x% sharper/blurrier" words.
"""
from __future__ import annotations

import os

import numpy as np

from euclid_polish.eval import jwst_resolution as jr
from euclid_polish.eval.battery.common import (
    BANDS,
    RESOLUTION_CACHE,
    Battery,
    ModelInfo,
    nexus_entries,
    nexus_key,
)
from euclid_polish.web.helpers import jwst_euclid
from scripts import jwst_resolution as script

STAGE = "resolution"
MULTIPLES = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
MIN_COVERAGE = 0.95
TEXTURE_PERIODS = (8.0, 16.0)


def mosaic_path(band: str) -> str | None:
    """The NIRCam mosaic a band samples (``None``: H's own per-tile F200W)."""
    if band == "H_E":
        return None
    return os.path.join(str(jwst_euclid.overlap_root()),
                        script.EDR_MOSAIC.format(filter=jr.JWST_FILTER[band].lower()))


def cached_shifts(band: str) -> np.ndarray:
    """Every NEXUS tile's measured registration shift in ``band`` (the
    resolution cache's), for a capped run whose tiles have none."""
    mosaic = mosaic_path(band)
    folder = (RESOLUTION_CACHE if mosaic is None else
              os.path.join(RESOLUTION_CACHE, f"{band}-{os.path.basename(mosaic).split('_i2d')[0]}"))
    out = []
    for entry in nexus_entries():
        path = os.path.join(folder, f"{entry.id}.npz")
        if os.path.isfile(path):
            with np.load(path) as data:
                if data["shift"].size:
                    out.append(tuple(data["shift"]))
    return np.array(out)


def band_tiles(band: str, limit: int | None = None):
    """``(tiles [(id, lr, registered JWST)], σ_Euclid per SR px, σ_JWST, meta)``
    of one band, as the script builds them (reads the resolution cache)."""
    entries = nexus_entries(limit)
    mosaic = mosaic_path(band)
    raw = (script._mosaic_jwst(RESOLUTION_CACHE, entries, band, mosaic) if mosaic
           else script._tile_jwst(RESOLUTION_CACHE, entries, band))
    raw = {tid: v for tid, v in raw.items() if np.isfinite(v[1]).any()}
    shifts = np.array([s for _l, _j, s in raw.values() if s is not None])
    if not len(shifts):
        shifts = cached_shifts(band)
    field_shift = tuple(float(v) for v in np.median(shifts, axis=0))
    tiles = [(tid, lr, jr.register(jwst, field_shift)) for tid, (lr, jwst, _s) in raw.items()]
    del raw
    b = BANDS.index(band)
    sigma_j = script._noise_sigma([j for _t, _l, j in tiles])
    sigma_e = script._noise_sigma([lr[..., b] for _t, lr, _j in tiles]) / 4.0
    meta = {"band": band, "jwst_filter": jr.JWST_FILTER[band],
            "jwst_source": os.path.basename(mosaic) if mosaic else "nexus field tiles (QDR deep ep05)",
            "n_tiles": len(tiles), "jwst_noise_e_per_sr_px": round(sigma_j, 5),
            "euclid_noise_e_per_sr_px": round(sigma_e, 5),
            "euclid_noise_sb": round(jr.sb_of_knee(band, sigma_e), 3),
            "field_shift_px": [round(v, 3) for v in field_shift]}
    return tiles, sigma_e, sigma_j, meta


def measure_band(bt: Battery, info: ModelInfo, band: str) -> tuple[dict, dict]:
    """``(model report, band meta)``: the script's per-model JSON entry."""
    tiles, sigma_e, sigma_j, meta = band_tiles(band, bt.limits.tiles)
    b = BANDS.index(band)
    cover = MIN_COVERAGE
    accs = {"linear": jr.RingCorrelation(lambda x: x, cover),
            "asinh_euclid": jr.RingCorrelation(lambda x, k=sigma_e: np.arcsinh(x / k), cover),
            "asinh_jwst": jr.RingCorrelation(lambda x, k=sigma_j: np.arcsinh(x / k), cover)}
    scan = {m: jr.RingCorrelation(lambda x, k=m * sigma_e: np.arcsinh(x / k), cover) for m in MULTIPLES}
    for tid, lr, jwst in tiles:
        sr = bt.cache.get(info, nexus_key(tid), lr)[..., b]
        for acc in (*accs.values(), *scan.values()):
            acc.add(sr, jwst)
    report = {name: jr.summary(acc) for name, acc in accs.items()}
    report["knee_scan"] = [
        {"multiple": m, "knee_e": round(m * sigma_e, 5), "sb": round(jr.sb_of_knee(band, m * sigma_e), 3),
         **{key: jr.summary(acc)[key] for key in ("resolution_coherence", "resolution_gain")}}
        for m, acc in scan.items()]
    return report, meta


def run(bt: Battery, info: ModelInfo) -> dict:
    def compute():
        bands, metas = {}, {}
        for band in BANDS:
            bands[band], metas[band] = measure_band(bt, info, band)
        return {"bands": bands, "meta": metas}, None

    return bt.stage(info, STAGE, compute)


def _array(values) -> np.ndarray:
    return np.array([np.nan if x is None else x for x in values], np.float64)


def band_entry(report: dict) -> dict:
    """The battery's numbers of one band's model report (``res_table_new.py``)."""
    a = report["asinh_euclid"]
    period = np.array(a["period_px"])
    coh = _array(a["coherence"])
    gain = _array(a["gain"])
    ratio = (gain / np.maximum(coh, 1e-6)) ** 2
    sel = (period >= TEXTURE_PERIODS[0]) & (period < TEXTURE_PERIODS[1])
    scan = {round(x["multiple"], 2): (x["resolution_coherence"] or {}).get("arcsec")
            for x in report["knee_scan"]}
    return {"windows": a["windows"],
            "r1": (a["resolution_coherence"] or {}).get("arcsec"),
            "nc1": (a.get("resolution_coherence_noise_corrected") or {}).get("arcsec"),
            "r03": scan.get(0.3), "r3": scan.get(3.0),
            "p816": float(np.nanmean(ratio[sel]))}


def words(ratio: float) -> str:
    """A resolution ratio (model / reference) in words: "0.8% sharper"."""
    return f"{abs(1 - ratio) * 100:.1f}% {'sharper' if ratio < 1 else 'blurrier'}"


def mean_ratio(cur: dict, base: dict, key: str = "r1") -> tuple[float | None, list[float]]:
    """Mean over the bands both measured of ``cur[band][key] / base[band][key]``."""
    rs = [cur[b][key] / base[b][key] for b in BANDS
          if b in cur and b in base and cur[b].get(key) and base[b].get(key)]
    return (float(np.mean(rs)) if rs else None), [round(float(r), 4) for r in rs]


def summary(bt: Battery, infos: list[ModelInfo]) -> dict:
    out: dict = {"models": {}, "vs_baseline": {}, "meta": None}
    for info in infos:
        res = bt.store.read(info, STAGE)
        if res is None:
            continue
        entries = {band: band_entry(rep) for band, rep in res["bands"].items()}
        out["models"][info.spec] = {"bands": entries,
                                    "p816": float(np.mean([e["p816"] for e in entries.values()]))}
        out["meta"] = out["meta"] or res["meta"]
    if infos and infos[0].spec in out["models"]:
        base = out["models"][infos[0].spec]["bands"]
        for info in infos[1:]:
            if info.spec not in out["models"]:
                continue
            cur = out["models"][info.spec]["bands"]
            row = {}
            for key in ("r1", "nc1", "r03", "r3"):
                m, per_band = mean_ratio(cur, base, key)
                row[key] = {"ratio": m, "per_band": per_band, "words": None if m is None else words(m)}
            row["p816_ratio"] = (out["models"][info.spec]["p816"]
                                 / out["models"][infos[0].spec]["p816"])
            out["vs_baseline"][info.spec] = row
    return out
