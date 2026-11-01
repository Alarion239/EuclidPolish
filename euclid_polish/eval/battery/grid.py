"""Test 2 — the grid: AxisExcess (:mod:`euclid_polish.training.grid_penalty`)
z-scores of 64-px windows, the share with z ≥ 3 (and z ≥ 6), the four bands
pooled, on three samples:

* **real sky** — the clean NEXUS tiles (no blank LR pixel, VIS < 5e4 and NISP
  < 5e3 e⁻), 32-px border dropped, the detector's own non-overlapping windows;
* **synthetic** — the score's test fields, every non-overlapping window;
* **galaxy outskirts** — stride-32 windows inside the 32-px border with ≥ 50 %
  of the footprint's LR VIS (Gaussian σ 2 LR px) at 1–10 σ and its peak ≤ 30 σ;
  on real tiles the window centre also sits in a MER galaxy's Kron ellipse with
  no star near (``clean``), on synthetic fields the truth carries ≥ half the
  input's light (``truth-light``).

Reference rows (no model): the Euclid input (cubic ×2), JWST F200W, the truth
and white noise. Shares are pooled over windows and bands; CIs and the paired
differences resample tiles / fields (2026-10-29 grid test, a93122dd detector).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import tensorflow as tf
from astropy.table import Table
from astropy.wcs import WCS
from scipy import ndimage
from scipy.ndimage import zoom

from euclid_polish.eval import flux_conservation as fc
from euclid_polish.eval.battery.common import (
    MER_CATALOGUE,
    RESOLUTION_CACHE,
    Battery,
    ModelInfo,
    field_key,
    field_truth,
    nexus_key,
    target_fwhm,
    test_fields,
)
from euclid_polish.training.grid_penalty import default_detector
from euclid_polish.web.helpers import real_tiles as rt

W = 64
BORDER = 32
SMOOTH = 2.0                    # Gaussian σ of the LR VIS smoothing (LR px)
KNEE_FRAC = 0.2                 # asinh knee of the "a" statistics, × the per-SR-pixel noise
NAMED = ("f200w-0141", "f200w-0038", "f200w-0106", "f200w-0362")
#: The real tiles and synthetic fields of the figures.
FIGURE_TILES = ("f200w-0141", "f200w-0038", "f200w-0106", "f200w-0362", "f200w-0205", "f200w-0415")
REFERENCES = ("lr", "jwst", "truth", "white")
_DET = None


def detector():
    global _DET
    if _DET is None:
        _DET = default_detector()
    return _DET


# --------------------------------------------------------------------------- #
# window statistics
# --------------------------------------------------------------------------- #

def noise_sigma(x) -> float:
    """Clipped MAD of a set of pixels (as ``scripts/jwst_resolution._noise_sigma``)."""
    values = np.asarray(x, np.float64)
    values = values[np.isfinite(values)].ravel()
    sigma = 1.4826 * float(np.median(np.abs(values - np.median(values))))
    for _ in range(3):
        core = values[np.abs(values - np.median(values)) < 3.0 * sigma]
        sigma = 1.4826 * float(np.median(np.abs(core - np.median(core))))
    return sigma


def stride_origins(h: int, w: int, lo: int = 0, step: int = 32) -> list[tuple[int, int]]:
    ys = list(range(lo, h - lo - W + 1, step))
    xs = list(range(lo, w - lo - W + 1, step))
    return [(y, x) for y in ys for x in xs]


def highpass(img: np.ndarray) -> np.ndarray:
    return detector().highpass(img[None].astype(np.float32)).numpy()[0]


def window_stats(img: np.ndarray, origins) -> dict[str, np.ndarray]:
    """z, amplitude and score of each window of the whole image's high-pass."""
    det = detector()
    hp = highpass(img)
    wins = np.stack([hp[y:y + W, x:x + W, :] for y, x in origins])
    wins = np.transpose(wins, (0, 3, 1, 2))
    on, pred = det.ring_power(wins)
    excess, spread = det._excess_and_spread(on, pred)
    excess, spread = excess.numpy(), spread.numpy()
    on_s = tf.reduce_sum(on, -1).numpy()
    pred_s = tf.reduce_sum(pred, -1).numpy()
    amp = np.sign(excess) * np.sqrt(np.abs(excess) / det._norm)
    return {"z": excess / spread, "amp": amp, "score": on_s / (pred_s + 1e-30)}


def protocol_z(img: np.ndarray, border: int) -> np.ndarray:
    """``border`` px dropped, then the detector's own non-overlapping windows."""
    h, w = img.shape[:2]
    crop = img[border:h - border, border:w - border] if border else img
    return detector().zscore(crop[None].astype(np.float32)).numpy()[0]


def lr_up(lr: np.ndarray) -> np.ndarray:
    """The Euclid input on the SR grid (cubic ×2, flux per SR pixel)."""
    return np.stack([zoom(np.nan_to_num(lr[..., b].astype(np.float64)), 2, order=3, grid_mode=True,
                          mode="grid-mirror") / 4.0 for b in range(lr.shape[-1])], -1).astype(np.float32)


def image_stats(img: np.ndarray, knees, origins, nan_mask: np.ndarray | None = None) -> dict:
    """Every statistic of one ``(H, W, C)`` image."""
    out = {}
    valid = np.ones(len(origins), bool)
    pvalid = np.ones((6, 6), bool)
    if nan_mask is not None:
        dil = ndimage.binary_dilation(nan_mask, structure=np.ones((27, 27), bool))
        valid = np.array([not dil[y:y + W, x:x + W].any() for y, x in origins])
        pvalid = np.array([[not dil[32 + 64 * i:96 + 64 * i, 32 + 64 * k:96 + 64 * k].any() for k in range(6)]
                           for i in range(6)])
        img = np.where(np.isfinite(img), img, 0.0).astype(np.float32)
    e = window_stats(img, origins)
    a = window_stats(np.arcsinh(img / np.asarray(knees, np.float32)).astype(np.float32), origins)
    for k, v in e.items():
        out[k + "_e"] = v
    for k, v in a.items():
        out[k + "_a"] = v
    hp = highpass(img)
    out["mean"] = np.stack([img[y:y + W, x:x + W].mean(axis=(0, 1)) for y, x in origins])
    out["hprms"] = np.stack([hp[y:y + W, x:x + W].std(axis=(0, 1)) for y, x in origins])
    out["pz32"] = protocol_z(img, 32)
    out["pz0"] = protocol_z(img, 0)
    out["valid"] = valid
    out["pvalid32"] = pvalid
    return out


def stack_rows(rows: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


# --------------------------------------------------------------------------- #
# shared inputs
# --------------------------------------------------------------------------- #

def clean_tiles(limit: int | None = None) -> list[str]:
    """NEXUS tiles with no blank (0) LR pixel, VIS < 5e4 and NISP < 5e3 e⁻."""
    out = []
    for f in sorted(glob.glob(os.path.join(RESOLUTION_CACHE, "f200w-*.npz"))):
        with np.load(f) as d:
            lr = d["lr"]
        if not np.isfinite(lr).all() or (lr == 0).any():
            continue
        if lr[..., 0].max() >= 5e4 or lr[..., 1:].max() >= 5e3:
            continue
        out.append(os.path.basename(f)[:-4])
    return out[:limit]


def cache_tile(tid: str) -> tuple[np.ndarray, np.ndarray]:
    """``(lr, jwst F200W on the SR grid)`` of a tile from the resolution cache."""
    with np.load(os.path.join(RESOLUTION_CACHE, f"{tid}.npz")) as d:
        return d["lr"].astype(np.float32), d["jwst"].astype(np.float32)


def window_classes(lr_vis: np.ndarray, sig_s: float, origins) -> np.ndarray:
    """Per window: the footprint's fraction of LR pixels with the smoothed VIS
    at 1–10 σ_s, its max and mean (σ_s), and the mean LR VIS per SR pixel."""
    S = ndimage.gaussian_filter(lr_vis.astype(np.float64), SMOOTH) / sig_s
    out = []
    for y, x in origins:
        win = S[y // 2:(y + W) // 2, x // 2:(x + W) // 2]
        raw = lr_vis[y // 2:(y + W) // 2, x // 2:(x + W) // 2]
        out.append(((win >= 1) & (win <= 10)).mean())
        out.append(win.max())
        out.append(win.mean())
        out.append(raw.mean() / 4.0)
    return np.asarray(out, np.float32).reshape(len(origins), 4)


def prep(bt: Battery) -> tuple[dict, dict]:
    """Tiles, fields, windows, noise levels and window classes (no model)."""
    def compute():
        tiles = clean_tiles(bt.limits.tiles)
        lrs = {t: cache_tile(t)[0] for t in tiles}
        sig_r = np.array([noise_sigma(np.concatenate([lrs[t][..., b].ravel()[::7] for t in tiles]))
                          for b in range(4)])
        sm = np.concatenate([ndimage.gaussian_filter(lrs[t][..., 0].astype(np.float64), SMOOTH).ravel()[::7]
                             for t in tiles])
        sigs_r = noise_sigma(sm)
        origins = stride_origins(510, 510, lo=0)
        cls_r = np.stack([window_classes(lrs[t][..., 0], sigs_r, origins) for t in tiles])
        syn_lr, fields = [], []
        for idx, lr, _rec in test_fields(bt.limits.fields):
            syn_lr.append(lr)
            fields.append(int(idx))
        sig_s = np.array([noise_sigma(np.concatenate([x[..., b].ravel()[::7] for x in syn_lr]))
                          for b in range(4)])
        sm = np.concatenate([ndimage.gaussian_filter(x[..., 0].astype(np.float64), SMOOTH).ravel()[::7]
                             for x in syn_lr])
        sigs_s = noise_sigma(sm)
        cls_s = np.stack([window_classes(x[..., 0], sigs_s, origins) for x in syn_lr])
        inner = np.array([(32 <= y <= 510 - 32 - W) and (32 <= x <= 510 - 32 - W) for y, x in origins])
        out_r = (cls_r[..., 0] >= 0.5) & (cls_r[..., 1] <= 30) & inner[None]
        out_s = (cls_s[..., 0] >= 0.5) & (cls_s[..., 1] <= 30) & inner[None]
        order = np.argsort(-out_r.sum(1), kind="stable")
        cand = [t for t in NAMED if t in tiles] + [tiles[i] for i in order if tiles[i] not in NAMED][:16]
        cand_syn = [fields[i] for i in np.argsort(-out_s.sum(1), kind="stable")[:4]]
        payload = {"tiles": tiles, "fields": fields, "sigma_lr_real": sig_r, "sigma_lr_syn": sig_s,
                   "sigma_smooth_vis_real": sigs_r, "sigma_smooth_vis_syn": sigs_s, "smooth_lr_px": SMOOTH,
                   "fwhm": target_fwhm(), "cand_real": cand, "cand_syn": cand_syn,
                   "n_outskirt_real": int(out_r.sum()), "n_outskirt_syn": int(out_s.sum())}
        arrays = {"origins": np.array(origins), "cls_real": cls_r, "cls_syn": cls_s, "out_real": out_r,
                  "out_syn": out_s, "inner": inner}
        return payload, arrays

    meta = bt.stage(None, "grid_prep", compute)
    return meta, bt.store.arrays(None, "grid_prep")


def _col(t: Table, name: str) -> np.ndarray:
    x = t[name]
    return np.asarray(x.filled(np.nan) if hasattr(x, "filled") else x, np.float64)


def catalogue_masks(bt: Battery, meta: dict, arrays: dict) -> dict[str, np.ndarray]:
    """Per real window: ``star_near`` (a MER point source close enough to light
    it) and ``gal_in`` (its centre inside a galaxy's Kron ellipse)."""
    def compute():
        origins = arrays["origins"]
        half_diag = 32 / np.sqrt(2.0) * np.sqrt(2.0)
        cat = Table.read(MER_CATALOGUE)
        ra, dec = _col(cat, "right_ascension"), _col(cat, "declination")
        plp = _col(cat, "point_like_prob")
        mag = 23.9 - 2.5 * np.log10(np.clip(_col(cat, "flux_detection_total"), 1e-6, None))
        star = np.isfinite(plp) & (plp > 0.5) & np.isfinite(mag)
        gal = ((_col(cat, "vis_det") == 1) & (plp < 0.1) & (_col(cat, "kron_radius") > 0)
               & np.isfinite(_col(cat, "ellipticity")))
        r_star = 10.0 * 10 ** (0.25 * (21.0 - mag))
        kr, ell, pa = _col(cat, "kron_radius"), _col(cat, "ellipticity"), _col(cat, "position_angle")
        entries = {e.id: e for e in rt.list_entries("nexus")}
        cy = (origins[:, 0] + 32) / 2.0 - 0.5
        cx = (origins[:, 1] + 32) / 2.0 - 0.5
        star_near, gal_in, gal_big = [], [], []
        for tid in meta["tiles"]:
            w = WCS(rt.lr_header(entries[tid])).celestial
            x, y = w.world_to_pixel_values(ra, dec)
            near = (x > -400) & (x < 655) & (y > -400) & (y < 655)
            s = np.flatnonzero(near & star)
            d = np.hypot(cx[:, None] - x[s][None], cy[:, None] - y[s][None])
            star_near.append((d < (r_star[s][None] + half_diag)).any(1))
            g = np.flatnonzero(near & gal)
            dx, dy = cx[:, None] - x[g][None], cy[:, None] - y[g][None]
            th = np.deg2rad(pa[g] + float(fc.KRON_ANGLE_OFFSET_DEG))[None]
            u = dx * np.cos(th) + dy * np.sin(th)
            v = -dx * np.sin(th) + dy * np.cos(th)
            a = kr[g][None]
            b = a * (1.0 - ell[g][None])
            inside = np.hypot(u / a, v / np.maximum(b, 1e-3)) <= 1.0
            gal_in.append(inside.any(1))
            gal_big.append((inside & (a >= 20.0)).any(1))
        return {}, {"star_near": np.array(star_near), "gal_in": np.array(gal_in),
                    "gal_big": np.array(gal_big)}

    bt.stage(None, "grid_catalogue", compute)
    return bt.store.arrays(None, "grid_catalogue")


def _real_stats(meta: dict, origins, image_of) -> dict[str, np.ndarray]:
    knees = KNEE_FRAC * np.asarray(meta["sigma_lr_real"]) / 4.0
    rows = []
    for tid in meta["tiles"]:
        img, nan_mask, k = image_of(tid, knees)
        rows.append(image_stats(img, k, origins, nan_mask))
    return stack_rows(rows)


def _syn_stats(bt: Battery, meta: dict, origins, image_of) -> dict[str, np.ndarray]:
    knees = KNEE_FRAC * np.asarray(meta["sigma_lr_syn"]) / 4.0
    rows = []
    for idx, lr, rec in test_fields(bt.limits.fields):
        img, k = image_of(idx, lr, rec, knees)
        rows.append(image_stats(img, k, origins))
    return stack_rows(rows)


def references(bt: Battery, meta: dict, arrays: dict) -> dict[str, dict[str, np.ndarray]]:
    """Statistics of the reference images: ``{"lr_real", "lr_syn", "jwst_real",
    "truth_syn", "white_real", "white_syn"}``."""
    origins = [tuple(o) for o in arrays["origins"]]
    fwhm = float(meta["fwhm"])

    def lr_real(tid, knees):
        return lr_up(cache_tile(tid)[0]), None, knees

    def jwst_real(tid, knees):
        jw = cache_tile(tid)[1]
        return jw[..., None], ~np.isfinite(jw), knees[3:4]

    def white_real_fn():
        rng = np.random.default_rng(1234)
        return lambda tid, knees: (rng.normal(size=(510, 510, 4)).astype(np.float32), None, [0.2] * 4)

    def white_syn_fn():
        rng = np.random.default_rng(4321)
        return lambda idx, lr, rec, knees: (rng.normal(size=(510, 510, 4)).astype(np.float32), [0.2] * 4)

    makers = {
        "lr_real": lambda: _real_stats(meta, origins, lr_real),
        "jwst_real": lambda: _real_stats(meta, origins, jwst_real),
        "white_real": lambda: _real_stats(meta, origins, white_real_fn()),
        "lr_syn": lambda: _syn_stats(bt, meta, origins, lambda i, lr, rec, k: (lr_up(lr), k)),
        "truth_syn": lambda: _syn_stats(bt, meta, origins, lambda i, lr, rec, k: (field_truth(rec, fwhm), k)),
        "white_syn": lambda: _syn_stats(bt, meta, origins, white_syn_fn()),
    }
    out = {}
    for name, make in makers.items():
        bt.stage(None, f"grid_ref_{name}", lambda make=make: ({}, make()))
        out[name] = bt.store.arrays(None, f"grid_ref_{name}")
    return out


def run(bt: Battery, info: ModelInfo) -> dict:
    """The model's window statistics on the clean real tiles and the test fields."""
    meta, arrays = prep(bt)
    origins = [tuple(o) for o in arrays["origins"]]

    def compute():
        real = _real_stats(meta, origins, lambda tid, knees: (
            bt.cache.get(info, nexus_key(tid), cache_tile(tid)[0]), None, knees))
        syn = _syn_stats(bt, meta, origins, lambda idx, lr, rec, knees: (
            bt.cache.get(info, field_key(idx), lr), knees))
        return {"tiles": meta["tiles"], "fields": meta["fields"]}, \
            {**{f"real_{k}": v for k, v in real.items()}, **{f"syn_{k}": v for k, v in syn.items()}}

    return bt.stage(info, "grid", compute)


def model_stats(bt: Battery, info: ModelInfo) -> dict[str, dict[str, np.ndarray]]:
    a = bt.store.arrays(info, "grid")
    return {dom: {k[len(dom) + 1:]: v for k, v in a.items() if k.startswith(dom + "_")}
            for dom in ("real", "syn")}


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #

def per_unit(z: np.ndarray, valid: np.ndarray, thr: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Hits (z ≥ ``thr``) and counts of valid windows per tile / field, bands pooled.
    ``z`` is ``(U, ..., C)``, ``valid`` ``(U, ...)``."""
    U, C = z.shape[0], z.shape[-1]
    zz = z.reshape(U, -1, C)
    vv = np.broadcast_to(valid.reshape(U, -1, 1), zz.shape)
    return ((zz >= thr) & vv).sum((1, 2)).astype(float), vv.sum((1, 2)).astype(float)


def pooled(units: tuple[np.ndarray, np.ndarray]) -> float:
    hits, n = units
    return float(hits.sum() / max(n.sum(), 1))


def share_ci(units: tuple[np.ndarray, np.ndarray], n_boot: int = 2000, seed: int = 0) -> list[float]:
    """95 % CI of a pooled share, resampling tiles / fields."""
    hits, n = units
    idx = np.random.default_rng(seed).integers(0, len(n), size=(n_boot, len(n)))
    shares = hits[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return [float(v) for v in np.percentile(shares, [2.5, 97.5])]


def paired(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], n_boot: int = 4000,
           seed: int = 11) -> dict:
    """Median and 95 % CI of share(a) − share(b), resampling the same tiles / fields."""
    k = len(a[1])
    idx = np.random.default_rng(seed).integers(0, k, size=(n_boot, k))
    da = a[0][idx].sum(1) / np.maximum(a[1][idx].sum(1), 1)
    db = b[0][idx].sum(1) / np.maximum(b[1][idx].sum(1), 1)
    d = da - db
    return {"median": float(np.median(d)), "ci95": [float(v) for v in np.percentile(d, [2.5, 97.5])]}


def outskirt_masks(arrays: dict, catalogue: dict, refs: dict) -> dict[str, np.ndarray]:
    """``{"real": clean, "syn": truth-light}`` outskirt windows ``(U, N)``."""
    real = arrays["out_real"] & ~catalogue["star_near"] & catalogue["gal_in"]
    syn = arrays["out_syn"] & (refs["truth_syn"]["mean"][..., 0] >= 0.5 * refs["lr_syn"]["mean"][..., 0])
    return {"real": real, "syn": syn}


def stats_row(s_real: dict | None, s_syn: dict | None, masks: dict, refs: dict, meta: dict,
              jwst: bool = False, outskirts: bool = True) -> dict:
    """One row of the grid table (final_tables.py), plus per-unit hits for the
    paired bootstraps."""
    r: dict = {}
    for dom, s in (("real", s_real), ("syn", s_syn)):
        if s is None:
            continue
        if dom == "real":
            valid = s["pvalid32"] if jwst else np.ones(s["pz32"].shape[:-1], bool)
            units = per_unit(s["pz32"], valid)
            r["real3"], r["real3_ci"], r["real_units"] = pooled(units), share_ci(units), units
        else:
            v = np.ones(s["pz0"].shape[:-1], bool)
            units = per_unit(s["pz0"], v)
            r["syn3"], r["syn6"] = pooled(units), pooled(per_unit(s["pz0"], v, 6.0))
            r["syn3_ci"], r["syn_units"] = share_ci(units), units
        if not outskirts:
            continue
        m = masks[dom] & s["valid"]
        units = per_unit(s["z_e"], m)
        r[f"{dom}_out3"], r[f"{dom}_out3_ci"], r[f"{dom}_out_units"] = pooled(units), share_ci(units), units
        r[f"{dom}_out_n"] = int(m.sum())
        r[f"{dom}_out_med"] = float(np.median(s["z_e"][m])) if m.any() else None
        zz = s["z_e"]
        if zz.shape[-1] == 4 and m.any():
            vis, h_ = (zz[..., 0] >= 3) & m, (zz[..., 3] >= 3) & m
            r[f"{dom}_vis_and_h"] = float((vis & h_).sum() / max(vis.sum(), 1))
            light = refs[f"lr_{dom}"]["mean"]
            hit = (zz >= 3) & m[..., None]
            r[f"{dom}_amp_hit"] = float(np.median(s["amp_e"][hit] / light[hit])) if hit.any() else None
            sig = np.asarray(meta["sigma_lr_real" if dom == "real" else "sigma_lr_syn"]) / 4.0
            r[f"{dom}_amp_hit_sig"] = float(np.median((s["amp_e"] / sig)[hit])) if hit.any() else None
    return r


def summary(bt: Battery, infos: list[ModelInfo]) -> dict:
    meta, arrays = prep(bt)
    catalogue = catalogue_masks(bt, meta, arrays)
    refs = references(bt, meta, arrays)
    masks = outskirt_masks(arrays, catalogue, refs)
    rows = {}
    for info in infos:
        if bt.store.read(info, "grid") is None:
            continue
        s = model_stats(bt, info)
        rows[info.spec] = stats_row(s["real"], s["syn"], masks, refs, meta)
    ref_rows = {"jwst": stats_row(refs["jwst_real"], None, masks, refs, meta, jwst=True),
                "lr": stats_row(refs["lr_real"], refs["lr_syn"], masks, refs, meta),
                "truth": stats_row(None, refs["truth_syn"], masks, refs, meta),
                "white": stats_row(refs["white_real"], refs["white_syn"], masks, refs, meta, outskirts=False)}
    vs = {}
    if infos and infos[0].spec in rows:
        base = rows[infos[0].spec]
        for info in infos[1:]:
            cur = rows.get(info.spec)
            if cur is None:
                continue
            vs[info.spec] = {key.removesuffix("_units"): paired(cur[key], base[key])
                             for key in ("real_units", "syn_units", "real_out_units", "syn_out_units")
                             if key in cur and key in base}

    def public(r):
        return {k: v for k, v in r.items() if not k.endswith("_units")}

    return {"models": {k: public(v) for k, v in rows.items()},
            "references": {k: public(v) for k, v in ref_rows.items()}, "vs_baseline": vs,
            "n_tiles": len(meta["tiles"]), "n_fields": len(meta["fields"]),
            "n_outskirt_windows": {"real": int(masks["real"].sum()), "syn": int(masks["syn"].sum())}}


def figure_keys(bt: Battery) -> set[str]:
    """SR cache keys the grid figures read (kept when the cache is pruned)."""
    meta, _arrays = prep(bt)
    return ({nexus_key(t) for t in FIGURE_TILES if t in meta["tiles"]}
            | {field_key(i) for i in meta["cand_syn"]})
