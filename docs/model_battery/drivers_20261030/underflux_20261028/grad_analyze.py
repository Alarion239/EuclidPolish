"""Backprop flux gain by pixel type.  python grad_analyze.py OUT_DIR

G[j, b, y, x, c] = ∂ Σ A(SR)[..., b] / ∂ LR[y, x, c].  Correlated with the band's LR PSF it is the
fraction of a faint point source added at (y, x) that survives into the re-observed SR (1 = flux
kept); with PSF ⊛ Gaussian(σ=3 px) the same for a small extended blob."""
import json
import sys

import numpy as np
from scipy import ndimage

from common import BANDS, block_factors, kernels
from features import CLASSES, TRIM, catalogue, tile_features

out_dir = sys.argv[1]
MODELS = ("member_238", "member_249")
BAND = [b.name for b in BANDS]
lr_all = np.load(f"{out_dir}/lr.npy", mmap_mode="r")
tiles = json.load(open(f"{out_dir}/tiles.json"))
pick = np.load(f"{out_dir}/grad_tiles.npy")
cat = catalogue()
factors = block_factors()
kernel, _ = kernels()
k = np.pad(kernel.numpy()[..., 0], ((0, 1), (0, 1), (0, 0)))           # 42×42×4 HR
psf_lr = k.reshape(21, 2, 21, 2, 4).sum((1, 3))
psf_lr /= psf_lr.sum((0, 1))
blob = np.stack([ndimage.gaussian_filter(np.pad(psf_lr[..., c], 12), 3.0) for c in range(4)], -1)
blob /= blob.sum((0, 1))
LO, HI = 40, 215                                                       # LR coords with full support
snr_edges = np.array([-np.inf, 1, 2, 4, 8, 16, 32, 64, 128, 1e3, np.inf])
report = {}
for label in MODELS:
    G = np.load(f"{out_dir}/G_{label}.npy", mmap_mode="r")
    a_all = np.load(f"{out_dir}/A_{label}.npy", mmap_mode="r")
    rec = {key: [] for key in ("pt", "ext", "pix", "z4", "snr", "cls", "band", "rnorm", "mag", "frac")}
    for j, i in enumerate(pick):
        f = tile_features(np.asarray(lr_all[i]), np.asarray(a_all[i]), tiles["headers"][i], cat, factors)
        for b in range(4):
            g = np.asarray(G[j, b, :, :, b], np.float64)
            pt = ndimage.correlate(g, psf_lr[..., b], mode="constant")[LO:HI, LO:HI]
            ext = ndimage.correlate(g, blob[..., b], mode="constant")[LO:HI, LO:HI]
            pix = g[LO:HI, LO:HI]
            sl = (slice(LO - TRIM, HI - TRIM),) * 2
            def grab(name):
                return f[name][:, b].reshape(235, 235)[sl].ravel()
            for key, val in (("pt", pt.ravel()), ("ext", ext.ravel()), ("pix", pix.ravel()),
                             ("z4", grab("z4")), ("snr", grab("snr_smooth")), ("cls", grab("cls")),
                             ("rnorm", grab("rnorm")), ("mag", grab("mag")),
                             ("frac", (grab("diff") / np.maximum(grab("lr"), 1e-9)))):
                rec[key].append(val)
            rec["band"].append(np.full(pt.size, b))
        print(label, j, flush=True)
    d = {key: np.concatenate(v) for key, v in rec.items()}
    np.savez_compressed(f"{out_dir}/gain_{label}.npz", **d)
    rep = {}
    for b in range(4):
        m = (d["band"] == b) & np.isfinite(d["z4"]) & (d["cls"] != CLASSES.index("hole"))
        r = {"by_class": {}, "by_snr": [], "worst": {}}
        for c, name in enumerate(CLASSES):
            s = m & (d["cls"] == c)
            if s.sum() < 50:
                continue
            r["by_class"][name] = {"n": int(s.sum()), "pt_gain_median": float(np.median(d["pt"][s])),
                                   "ext_gain_median": float(np.median(d["ext"][s])),
                                   "pix_gain_median": float(np.median(d["pix"][s]))}
        for lo, hi in zip(snr_edges[:-1], snr_edges[1:]):
            s = m & (d["snr"] >= lo) & (d["snr"] < hi)
            if s.sum() >= 50:
                r["by_snr"].append({"snr_lo": float(lo), "n": int(s.sum()),
                                    "pt_gain_median": float(np.median(d["pt"][s])),
                                    "ext_gain_median": float(np.median(d["ext"][s])),
                                    "z4_median": float(np.median(d["z4"][s]))})
        worst, bright = m & (d["z4"] < -5), m & (d["snr"] > 8)
        for name, s in (("z4<-5", worst), ("all", m), ("snr>8 & |z4|<2", bright & (np.abs(d["z4"]) < 2)),
                        ("snr>8 & z4<-5", bright & worst)):
            if s.sum():
                r["worst"][name] = {"n": int(s.sum()), "pt_gain_median": float(np.median(d["pt"][s])),
                                    "ext_gain_median": float(np.median(d["ext"][s])),
                                    "snr_median": float(np.median(d["snr"][s]))}
        rep[BAND[b]] = r
    report[label] = rep
json.dump(report, open(f"{out_dir}/gain_report.json", "w"), indent=1)
print("done")
