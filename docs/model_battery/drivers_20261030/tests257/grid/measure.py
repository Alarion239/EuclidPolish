"""Per-window grid statistics of one model (or reference) on the real NEXUS
clean tiles and the 100 synthetic test fields. One model per process.

    python measure.py SPEC            # a model spec, or lr | jwst | truth | hr | white

Writes stats/<slug>__real.npz and/or stats/<slug>__syn.npz; caches the SR's
VIS + H (float32) of the figure candidates under sr_cache/.
"""
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from euclid_polish.web.helpers import universal_models as um  # noqa: E402

P = np.load(os.path.join(C.HERE, "prep.npz"))
META = json.load(open(os.path.join(C.HERE, "prep.json")))
ORIGINS = [tuple(o) for o in P["origins"]]
KNEE_FRAC = 0.2       # asinh knee of the "a" representation, x the per-SR-pixel noise


def lr_up(lr):
    """The Euclid input on the SR grid (cubic x2, flux per SR pixel), as jwst_resolution's 'lr'."""
    return np.stack([zoom(np.nan_to_num(lr[..., b].astype(np.float64)), 2, order=3, grid_mode=True,
                          mode="grid-mirror") / 4.0 for b in range(lr.shape[-1])], -1).astype(np.float32)


def image_stats(img, knees, nan_mask=None):
    """All statistics of one (H, W, C) image."""
    out = {}
    valid = np.ones(len(ORIGINS), bool)
    pvalid = np.ones((6, 6), bool)
    if nan_mask is not None:
        dil = ndimage.binary_dilation(nan_mask, structure=np.ones((27, 27), bool))
        valid = np.array([not dil[y:y + C.W, x:x + C.W].any() for y, x in ORIGINS])
        pvalid = np.array([[not dil[32 + 64 * i:96 + 64 * i, 32 + 64 * k:96 + 64 * k].any() for k in range(6)]
                           for i in range(6)])
        img = np.where(np.isfinite(img), img, 0.0).astype(np.float32)
    e = C.window_stats(img, ORIGINS)
    a = C.window_stats(np.arcsinh(img / np.asarray(knees, np.float32)).astype(np.float32), ORIGINS)
    for k, v in e.items():
        out[k + "_e"] = v
    for k, v in a.items():
        out[k + "_a"] = v
    hp = C.highpass(img)
    out["mean"] = np.stack([img[y:y + C.W, x:x + C.W].mean(axis=(0, 1)) for y, x in ORIGINS])
    out["hprms"] = np.stack([hp[y:y + C.W, x:x + C.W].std(axis=(0, 1)) for y, x in ORIGINS])
    out["pz32"] = C.protocol_z(img, 32)
    out["pz0"] = C.protocol_z(img, 0)
    out["valid"] = valid
    out["pvalid32"] = pvalid
    return out


def stack(rows):
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def cache_sr(spec, key, sr):
    os.makedirs(C.SRC, exist_ok=True)
    path = os.path.join(C.SRC, f"{C.slug(spec)}__{key}.npy")
    if not os.path.isfile(path):
        np.save(path, np.ascontiguousarray(sr[..., [0, 3]], np.float32))


def run_real(spec):
    tiles = META["tiles"]
    knees = KNEE_FRAC * np.asarray(META["sigma_lr_real"]) / 4.0
    rows = []
    rng = np.random.default_rng(1234)
    t0 = time.time()
    for n, tid in enumerate(tiles):
        lr, jw = C.tile(tid)
        if spec == "lr":
            rows.append(image_stats(lr_up(lr), knees))
        elif spec == "jwst":
            img = jw[..., None]
            rows.append(image_stats(img, knees[3:4], nan_mask=~np.isfinite(jw)))
        elif spec == "white":
            rows.append(image_stats(rng.normal(size=(510, 510, 4)).astype(np.float32), [0.2] * 4))
        else:
            sr = C.run_sr(spec, lr)
            if tid in META["cand_real"]:
                cache_sr(spec, tid, sr)
            rows.append(image_stats(sr, knees))
        if n % 50 == 0:
            print(f"  real {spec} {n}/{len(tiles)} {time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(C.STATS, f"{C.slug(spec)}__real.npz"), tiles=np.array(tiles), **stack(rows))


def run_syn(spec):
    from euclid_polish.web.helpers.ensemble_viz import universal_score_identity
    ident = universal_score_identity()
    fwhm = float(ident["target_psf_fwhm_arcsec"])
    knees = KNEE_FRAC * np.asarray(META["sigma_lr_syn"]) / 4.0
    rows, fields = [], []
    rng = np.random.default_rng(4321)
    t0 = time.time()
    for idx, lr, rec in um._test_pairs(C.RECORDS, ident["indices"]):
        if spec == "truth":
            img = um._field_truth(rec, fwhm)
        elif spec == "hr":
            img = np.asarray(rec.data, np.float32)
        elif spec == "lr":
            img = lr_up(lr)
        elif spec == "white":
            img = rng.normal(size=(510, 510, 4)).astype(np.float32)
        else:
            img = C.run_sr(spec, lr)
        if int(idx) in META["cand_syn"] and spec not in ("white",):
            cache_sr(spec, f"syn{int(idx):03d}", img)
        rows.append(image_stats(img, knees if spec != "white" else [0.2] * 4))
        fields.append(int(idx))
        if len(fields) % 50 == 1:
            print(f"  syn {spec} {len(fields)}/100 {time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(C.STATS, f"{C.slug(spec)}__syn.npz"), fields=np.array(fields), **stack(rows))


def main():
    spec = sys.argv[1]
    os.makedirs(C.STATS, exist_ok=True)
    t0 = time.time()
    if spec not in ("truth", "hr") and not os.path.isfile(os.path.join(C.STATS, f"{C.slug(spec)}__real.npz")):
        run_real(spec)
    if spec != "jwst" and not os.path.isfile(os.path.join(C.STATS, f"{C.slug(spec)}__syn.npz")):
        run_syn(spec)
    print(f"done {spec} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
