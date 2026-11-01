"""Tests 5 and 6 — flux kept.

* **Kron VIS vs MER (real sky).** ``scripts/flux_conservation.py`` on the SR
  cache: the MER catalogue's Kron VIS total and matched-aperture fluxes redone
  on each model's SR of every NEXUS tile; the isolated galaxies (the input gives
  back the catalogue's Kron flux within 0.5σ) set the sample. The table is the
  median SR/input Kron VIS flux per VIS magnitude range (68 % bootstrap CI).
* **Synthetic SR/truth.** Sources detected in the LR VIS of the score's test
  fields (photutils, 3σ, 8 px), isolated (no other segment inside the Kron
  ellipse); fluxes in big Kron apertures (a ≥ 18 LR px) on the SR, the truth
  (target PSF) and the input, binned by the truth's VIS magnitude.

The geometry and the input / truth fluxes are measured once (no model); each
model adds its SR fluxes. Paired comparisons with the first model use the same
galaxies / sources.
"""
from __future__ import annotations

import numpy as np
from astropy.wcs import WCS
from photutils.aperture import EllipticalAperture, aperture_photometry
from photutils.segmentation import SourceCatalog, detect_sources

from euclid_polish.config import Config
from euclid_polish.eval import flux_conservation as fc
from euclid_polish.eval.battery.common import (
    MER_CATALOGUE,
    Battery,
    ModelInfo,
    field_key,
    field_truth,
    nexus_entries,
    nexus_key,
    nexus_lr,
    target_fwhm,
    test_fields,
)
from euclid_polish.photometry import electrons_to_ab_mag, uJy_to_ab_mag, uJy_to_electrons
from euclid_polish.training.inference import scaled_wcs_header
from euclid_polish.web.helpers import real_tiles as rt
from scripts import flux_conservation as fcs

KRON_RANGES = ((21.0, 23.0), (23.0, 24.0), (24.0, 24.5), (24.5, 25.0), (21.0, 25.01))
SYN_RANGES = ((19.0, 23.0), (23.0, 24.0), (24.0, 25.0), (25.0, 26.0))
GEOMETRY_KEYS = ("kron_radius", "ellipticity", "position_angle", "fwhm")
#: Big synthetic apertures: semi-major axis at least this (LR px).
BIG_A_LR_PX = 18.0


# --------------------------------------------------------------------------- #
# Kron VIS vs MER
# --------------------------------------------------------------------------- #

def measure_keys() -> list[str]:
    return ["kron_vis"] + [f"ap{n}_{b}" for b in fc.BANDS for n in fc.APERTURE_MULTIPLES]


def _columns(measured: list[dict]) -> dict[str, np.ndarray]:
    return {key: np.array([m[key] for m in measured], np.float64) for key in measure_keys()}


def geometry(bt: Battery) -> dict[str, np.ndarray]:
    """Per galaxy (tile order): the tile, the catalogue row, the SR-grid
    position, the tile's matched FWHM and the input's fluxes."""
    def compute():
        cat = fcs.galaxies(fcs.fetch_catalogue(MER_CATALOGUE))
        ra, dec = np.asarray(cat["right_ascension"], float), np.asarray(cat["declination"], float)
        tiles, rows_out, xs, ys, targets, measured = [], [], [], [], [], []
        for entry in nexus_entries(bt.limits.tiles):
            lr = np.asarray(rt.get_tile("nexus", entry.id, entry=entry).lr_e, np.float32)
            header = rt.lr_header(entry)
            x, y = WCS(header).celestial.world_to_pixel_values(ra, dec)
            h, w = lr.shape[:2]
            reach = np.maximum(np.asarray(cat["kron_radius"], float),
                               2.0 * np.asarray(cat["fwhm"], float) / fc.LR_PIXEL_ARCSEC) + fcs.MARGIN_LR_PX
            inside = np.flatnonzero((x - reach > 0) & (x + reach < w - 1)
                                    & (y - reach > 0) & (y + reach < h - 1))
            if not inside.size:
                continue
            rows = [{k: float(cat[k][i]) for k in GEOMETRY_KEYS} for i in inside]
            target = float(np.median([r["fwhm"] for r in rows]))
            sr_wcs = WCS(scaled_wcs_header(header, 2)).celestial
            sx, sy = sr_wcs.world_to_pixel_values(ra[inside], dec[inside])
            xy_sr = np.stack([sx, sy], 1)
            lr_sr = np.repeat(np.repeat(lr, 2, 0), 2, 1) / 4.0
            measured += fc.measure(lr_sr, xy_sr, rows, target, scale=2)
            tiles += [entry.id] * inside.size
            rows_out += [int(i) for i in inside]
            xs += list(sx)
            ys += list(sy)
            targets += [target] * inside.size
        arrays = {"tile": np.array(tiles), "row": np.array(rows_out, np.int64), "x": np.array(xs, np.float64),
                  "y": np.array(ys, np.float64), "target": np.array(targets, np.float64),
                  **{f"geo_{k}": np.array([float(cat[k][i]) for i in rows_out], np.float64)
                     for k in GEOMETRY_KEYS},
                  **{f"lr_{k}": v for k, v in _columns(measured).items()}}
        return {"n_galaxies": len(rows_out), "n_tiles": len(set(tiles))}, arrays

    bt.stage(None, "flux_geometry", compute)
    return bt.store.arrays(None, "flux_geometry")


def run_kron(bt: Battery, info: ModelInfo) -> dict:
    """The model's fluxes of every galaxy of :func:`geometry`."""
    geo = geometry(bt)

    def compute():
        entries = {e.id: e for e in nexus_entries(bt.limits.tiles)}
        measured = []
        tiles = geo["tile"]
        for tid in dict.fromkeys(tiles.tolist()):
            sel = np.flatnonzero(tiles == tid)
            rows = [{k: float(geo[f"geo_{k}"][i]) for k in GEOMETRY_KEYS} for i in sel]
            xy = np.stack([geo["x"][sel], geo["y"][sel]], 1)
            sr = bt.cache.get(info, nexus_key(tid), nexus_lr(entries[tid]))
            measured += fc.measure(sr, xy, rows, float(geo["target"][sel[0]]), scale=2)
        return {"n_galaxies": len(measured)}, _columns(measured)

    return bt.stage(info, "flux", compute)


def catalogue_columns(bt: Battery, geo: dict) -> dict[str, np.ndarray]:
    """``object_id``, VIS magnitude and the MER fluxes / errors (electrons) of the galaxies."""
    cat = fcs.galaxies(fcs.fetch_catalogue(MER_CATALOGUE))
    sub = cat[geo["row"].tolist()]
    conv = {band: uJy_to_electrons(1.0, Config.get_band(name))
            for band, name in zip(fc.BANDS, Config.LR_INPUT_BAND_NAMES, strict=True)}

    def values(name, band):
        col = sub[name]
        return np.asarray(col.filled(np.nan) if hasattr(col, "filled") else col, np.float64) * conv[band]

    out = {"object_id": np.asarray(sub["object_id"]),
           "vis_mag": uJy_to_ab_mag(np.asarray(sub["flux_detection_total"], float)),
           "cat:kron_vis": values("flux_detection_total", "vis"),
           "err:kron_vis": values("fluxerr_detection_total", "vis")}
    for b in fc.BANDS:
        for n in fc.APERTURE_MULTIPLES:
            out[f"cat:ap{n}_{b}"] = values(f"flux_{b}_{n}fwhm_aper", b)
            out[f"err:ap{n}_{b}"] = values(f"fluxerr_{b}_{n}fwhm_aper", b)
    return out


def isolated(lr: np.ndarray, catalogue: np.ndarray, error: np.ndarray) -> np.ndarray:
    """The input gives back the catalogue's Kron flux (MER masked no neighbour)."""
    return (np.abs(lr - catalogue) <= fcs.ISOLATED_PULL * error) & (error > 0) & (catalogue > 0)


def boot_median(x: np.ndarray, n_boot: int = 4000, seed: int = 0) -> dict:
    """Median and 68 % bootstrap CI of the finite values."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"median": None, "ci68": [None, None], "n": 0}
    meds = np.median(np.random.default_rng(seed).choice(x, (n_boot, x.size), replace=True), axis=1)
    return {"median": float(np.median(x)), "ci68": [float(v) for v in np.percentile(meds, [16, 84])],
            "n": int(x.size)}


def paired_median(d: np.ndarray, n_boot: int = 4000, seed: int = 1) -> dict:
    """Median of per-object differences, 68 % bootstrap CI and the share > 0."""
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"median": None, "ci68": [None, None], "frac_positive": None, "n": 0}
    meds = np.median(np.random.default_rng(seed).choice(d, (n_boot, d.size)), axis=1)
    return {"median": float(np.median(d)), "ci68": [float(v) for v in np.percentile(meds, [16, 84])],
            "frac_positive": float(np.mean(d > 0)), "n": int(d.size)}


def range_name(lo: float, hi: float) -> str:
    return f"{lo:g}-{min(hi, 25.0) if hi == 25.01 else hi:g}"


def report(geo: dict, cols: dict, models: dict[str, dict[str, np.ndarray]]) -> dict:
    """``scripts/flux_conservation.py``'s report (input + models) from the
    measured columns."""
    tests = [("kron_vis", "vis")] + [(f"ap{n}_{b}", b) for b in fc.BANDS for n in fc.APERTURE_MULTIPLES]
    kron_c, kron_e = cols["cat:kron_vis"], cols["err:kron_vis"]
    iso = np.abs(geo["lr_kron_vis"] - kron_c) <= fcs.ISOLATED_PULL * kron_e
    out = {"meta": {"n_galaxies": int(len(kron_c)), "n_isolated": int(iso.sum())}, "corrections": {},
           "models": {}}
    for key, _band in tests:
        c, e = cols[f"cat:{key}"].copy(), cols[f"err:{key}"]
        c[~iso] = np.nan
        bright = c >= 20 * e
        corr = 1.0 if key == "kron_vis" else fc.aperture_correction(geo[f"lr_{key}"][bright], c[bright])
        out["corrections"][key] = round(corr, 4)
        lr_values = {k.removeprefix("lr_"): v for k, v in geo.items() if k.startswith("lr_")}
        for name, values in (("lr", lr_values),
                             *models.items()):
            out["models"].setdefault(name, {})[key] = fc.summarize(values[key], geo[f"lr_{key}"], c, e,
                                                                   cols["vis_mag"], corr)
    return out


def kron_summary(bt: Battery, infos: list[ModelInfo]) -> dict:
    geo = geometry(bt)
    cols = catalogue_columns(bt, geo)
    lr, mag = geo["lr_kron_vis"], cols["vis_mag"]
    cat, err = cols["cat:kron_vis"], cols["err:kron_vis"]
    keep = isolated(lr, cat, err)
    models = {info.spec: bt.store.arrays(info, "flux") for info in infos if bt.store.read(info, "flux")}
    out: dict = {"n_isolated": int(keep.sum()), "ranges": [range_name(a, b) for a, b in KRON_RANGES],
                 "models": {}, "vs_baseline": {}, "pull": {}}
    for spec, values in models.items():
        r = values["kron_vis"] / lr
        out["models"][spec] = {range_name(a, b): boot_median(r[keep & (mag >= a) & (mag < b)
                                                                & np.isfinite(r)])
                               for a, b in KRON_RANGES}
        pull = (values["kron_vis"] - cat) / err
        s = keep & np.isfinite(pull)
        out["pull"][spec] = {"median": float(np.median(pull[s])) if s.any() else None,
                             "faint_24p5_median": (float(np.median(pull[s & (mag >= 24.5)]))
                                                   if (s & (mag >= 24.5)).any() else None)}
    if infos and infos[0].spec in models:
        base = models[infos[0].spec]["kron_vis"] / lr
        for info in infos[1:]:
            if info.spec not in models:
                continue
            cur = models[info.spec]["kron_vis"] / lr
            out["vs_baseline"][info.spec] = {
                range_name(a, b): paired_median((cur - base)[keep & (mag >= a) & (mag < b)
                                                             & np.isfinite(cur) & np.isfinite(base)])
                for a, b in KRON_RANGES}
    out["report"] = report(geo, cols, models)
    return out


def fluxes_npz(bt: Battery, infos: list[ModelInfo]) -> dict[str, np.ndarray]:
    """The per-galaxy fluxes in ``scripts/flux_conservation.py``'s
    ``<report>_fluxes.npz`` layout (``scripts/plot_flux_conservation.py`` reads it)."""
    geo = geometry(bt)
    cols = catalogue_columns(bt, geo)
    out = {"object_id": cols["object_id"], "vis_mag": cols["vis_mag"],
           **{f"lr:{k.removeprefix('lr_')}": v for k, v in geo.items() if k.startswith("lr_")}}
    for info in infos:
        if bt.store.read(info, "flux"):
            out.update({f"{info.spec}:{k}": v for k, v in bt.store.arrays(info, "flux").items()})
    return out


# --------------------------------------------------------------------------- #
# synthetic SR / truth
# --------------------------------------------------------------------------- #

def kron(img: np.ndarray, x: float, y: float, a: float, b: float, th: float) -> float:
    table = aperture_photometry(img, EllipticalAperture((x, y), a, b, th), method="exact")
    return float(table["aperture_sum"][0])


def field_sources(lr: np.ndarray) -> tuple[list[dict], float]:
    """Isolated sources of one field's LR VIS: ``([{x, y, a, b, th, A, B, peak}], σ)``."""
    v = lr[..., 0]
    med = np.median(v)
    sig = 1.4826 * np.median(np.abs(v - med))
    seg = detect_sources(v - med, 3 * sig, npixels=8)
    if seg is None:
        return [], float(sig)
    out = []
    H, W = v.shape
    for s in SourceCatalog(v - med, seg):
        a_k = float(s.kron_aperture.a) if s.kron_aperture is not None else np.nan
        if not np.isfinite(a_k) or a_k <= 0:
            continue
        b_k = float(s.kron_aperture.b)
        theta = s.kron_aperture.theta
        th = float(theta.to_value("rad")) if hasattr(theta, "to_value") else float(theta)
        x, y = float(s.xcentroid), float(s.ycentroid)
        if x - a_k < 2 or y - a_k < 2 or x + a_k > W - 3 or y + a_k > H - 3:
            continue
        ap = EllipticalAperture((x, y), a_k, b_k, th).to_mask(method="center")
        labs = np.unique(ap.multiply(seg.data))
        labs = labs[(labs > 0) & (labs != s.label)]
        if labs.size:
            continue
        q = b_k / a_k
        A = max(a_k, BIG_A_LR_PX)
        B = A * q
        if x - A < 2 or y - A < 2 or x + A > W - 3 or y + A > H - 3:
            continue
        out.append({"x": x, "y": y, "a": a_k, "b": b_k, "th": th, "A": A, "B": B,
                    "peak_lr": float(v[int(round(y)), int(round(x))])})
    return out, float(sig)


def sr_fluxes(img: np.ndarray, sources: list[dict]) -> tuple[list[float], list[float]]:
    """``(small, big)`` Kron fluxes of the sources in a SR-grid VIS plane."""
    small = [kron(img, 2 * s["x"] + 0.5, 2 * s["y"] + 0.5, 2 * s["a"], 2 * s["b"], s["th"]) for s in sources]
    big = [kron(img, 2 * s["x"] + 0.5, 2 * s["y"] + 0.5, 2 * s["A"], 2 * s["B"], s["th"]) for s in sources]
    return small, big


def synthetic_sources(bt: Battery) -> dict[str, np.ndarray]:
    def compute():
        fwhm = target_fwhm()
        cols: dict[str, list] = {k: [] for k in ("field", "x", "y", "a", "b", "th", "A", "B", "peak_lr",
                                                  "err",
                                                  "lr", "lr_big", "truth", "truth_big")}
        for index, lr, target in test_fields(bt.limits.fields):
            sources, sig = field_sources(lr)
            if not sources:
                continue
            truth = field_truth(target, fwhm)
            v = lr[..., 0]
            t_small, t_big = sr_fluxes(truth[..., 0], sources)
            for s, ts, tb in zip(sources, t_small, t_big, strict=True):
                for k in ("x", "y", "a", "b", "th", "A", "B", "peak_lr"):
                    cols[k].append(s[k])
                cols["field"].append(int(index))
                cols["lr"].append(kron(v, s["x"], s["y"], s["a"], s["b"], s["th"]))
                cols["lr_big"].append(kron(v, s["x"], s["y"], s["A"], s["B"], s["th"]))
                cols["truth"].append(ts)
                cols["truth_big"].append(tb)
                cols["err"].append(sig * np.sqrt(np.pi * s["a"] * s["b"]))
        arrays = {k: np.asarray(v, np.int64 if k == "field" else np.float64) for k, v in cols.items()}
        return {"n_sources": int(len(arrays["x"]))}, arrays

    bt.stage(None, "synflux_sources", compute)
    return bt.store.arrays(None, "synflux_sources")


def run_synthetic(bt: Battery, info: ModelInfo) -> dict:
    src = synthetic_sources(bt)

    def compute():
        small, big = [], []
        for index, lr, _target in test_fields(bt.limits.fields):
            sel = np.flatnonzero(src["field"] == index)
            if not sel.size:
                continue
            sources = [{k: float(src[k][i]) for k in ("x", "y", "a", "b", "th", "A", "B")} for i in sel]
            sr = bt.cache.get(info, field_key(index), lr)
            s, b = sr_fluxes(sr[..., 0], sources)
            small += s
            big += b
        return {"n_sources": len(small)}, {"sr": np.asarray(small, np.float64),
                                           "sr_big": np.asarray(big, np.float64)}

    return bt.stage(info, "synflux", compute)


def truth_mag(truth_big: np.ndarray) -> np.ndarray:
    vis = Config.get_band("VIS")
    return np.array([electrons_to_ab_mag(max(t, 1e-9), vis) for t in truth_big], np.float64)


def synthetic_summary(bt: Battery, infos: list[ModelInfo]) -> dict:
    src = synthetic_sources(bt)
    mag = truth_mag(src["truth_big"])
    tr, lr = src["truth_big"], src["lr_big"]
    names = [f"{a:g}-{b:g}" for a, b in SYN_RANGES]

    def binned(values):
        return {name: boot_median(values[(mag >= a) & (mag < b)], n_boot=2000)
                for name, (a, b) in zip(names, SYN_RANGES, strict=True)}

    with np.errstate(divide="ignore", invalid="ignore"):
        out: dict = {"n_sources": int(len(tr)), "ranges": names, "input_over_truth": binned(lr / tr),
                     "models": {}, "vs_baseline": {}}
        models = {info.spec: bt.store.arrays(info, "synflux") for info in infos
                  if bt.store.read(info, "synflux")}
        for spec, values in models.items():
            out["models"][spec] = {"sr_over_truth": binned(values["sr_big"] / tr),
                                   "sr_over_input": binned(values["sr_big"] / lr)}
        if infos and infos[0].spec in models:
            base = models[infos[0].spec]["sr_big"] / tr
            for info in infos[1:]:
                if info.spec not in models:
                    continue
                cur = models[info.spec]["sr_big"] / tr
                ok = np.isfinite(cur) & np.isfinite(base)
                out["vs_baseline"][info.spec] = {
                    name: paired_median((cur - base)[ok & (mag >= a) & (mag < b)])
                    for name, (a, b) in zip(names, SYN_RANGES, strict=True)}
    return out
