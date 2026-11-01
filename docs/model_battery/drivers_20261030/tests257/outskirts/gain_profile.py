"""Real-sky (NEXUS) outskirt profiles on the 45 gradient tiles: backprop gain and SR/input flux vs r/r_Kron.

python gain_profile.py UNDERFLUX_OUT RUNS_DIR   (run from the repo root: features.py reads the MER catalogue)

gain  = median over non-star pixels of G ⊛ blob (grad_analyze.py's "ext": the fraction of a faint, slightly
        extended blob added at that pixel that survives into the re-observed SR; 1 = kept, 0 = erased).
ratio = 1 + Σdiff / ΣLR over non-star, non-hole pixels (profiles.py's A_fit/LR, flux-weighted), with A_fit =
        K ⊛ rebin(SR) and K a unit-sum 21×21 LR kernel fitted per model on these same 45 tiles (fitpsf.py's
        normal equations). 238/249 use the stored 10-28 G and R arrays, 257@15M the arrays of gain_run.py.
"""
import json
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

from common import block_factors, kernels
from euclid_polish.eval import knee_attribution as ka
from features import CLASSES, TRIM, catalogue, tile_features

out_dir, runs = sys.argv[1], sys.argv[2]
BAND = ["VIS", "Y", "J", "H"]
R_EDGES = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 9.1])
MAG_EDGES = np.array([0, 18, 19.5, 21, 22.5, 24, 99])
STAR = [CLASSES.index("star core"), CLASSES.index("star wing")]
HOLE = CLASSES.index("hole")
LO, HI = 40, 215
H = 10
MODELS = {"238": (f"{out_dir}/G_member_238.npy", f"{out_dir}/R_member_238.npy", True),
          "249": (f"{out_dir}/G_member_249.npy", f"{out_dir}/R_member_249.npy", True),
          "257@15M": (f"{runs}/gain_257_15M_G.npy", f"{runs}/gain_257_15M_R.npy", False)}

lr_all = np.load(f"{out_dir}/lr.npy", mmap_mode="r")
tiles = json.load(open(f"{out_dir}/tiles.json"))
pick = np.load(f"{out_dir}/grad_tiles.npy")
cat = catalogue()
factors = block_factors()
kernel, _ = kernels()
k = np.pad(kernel.numpy()[..., 0], ((0, 1), (0, 1), (0, 0)))           # 42×42×4 HR  (grad_analyze.py)
psf_lr = k.reshape(21, 2, 21, 2, 4).sum((1, 3))
psf_lr /= psf_lr.sum((0, 1))
blob = np.stack([ndimage.gaussian_filter(np.pad(psf_lr[..., c], 12), 3.0) for c in range(4)], -1)
blob /= blob.sum((0, 1))
nm, nr = len(MAG_EDGES) - 1, len(R_EDGES) - 1


def fit_kernel(R, tiles_idx):
    p = (2 * H + 1) ** 2
    c = p // 2
    XtX = np.zeros((4, p - 1, p - 1))
    Xty = np.zeros((4, p - 1))
    for j, i in enumerate(tiles_idx):
        lr = np.asarray(lr_all[i], np.float64)
        r = np.asarray(R[j], np.float64)
        sky = ka.noise_sigma(lr) / factors[:, 0]
        for b in range(4):
            win = sliding_window_view(r[..., b], (2 * H + 1, 2 * H + 1)).reshape(-1, p)
            y = lr[H:-H, H:-H, b].ravel()
            ok = np.isfinite(y) & (y != 0)
            w = 1.0 / (sky[b] ** 2 + np.clip(y, 0, None))
            X = (np.delete(win, c, 1) - win[:, c:c + 1])[ok]
            t = (y - win[:, c])[ok]
            XtX[b] += X.T @ (X * w[ok][:, None])
            Xty[b] += X.T @ (t * w[ok])
    K = np.zeros((4, 2 * H + 1, 2 * H + 1))
    for b in range(4):
        kk = np.linalg.solve(XtX[b], Xty[b])
        K[b] = np.insert(kk, c, 1.0 - kk.sum()).reshape(2 * H + 1, 2 * H + 1)
    return K


res = {"r_edges": R_EDGES.tolist(), "mag_edges": MAG_EDGES.tolist(), "models": {}}
for label, (gpath, rpath, stored_full) in MODELS.items():
    G = np.load(gpath, mmap_mode="r")
    Rall = np.load(rpath, mmap_mode="r")
    R = Rall[pick] if stored_full else Rall                               # (45, 255, 255, 4)
    K = fit_kernel(R, pick)
    gain_vals = [[[[] for _ in range(nr)] for _ in range(nm)] for _ in range(4)]
    fsum = np.zeros((4, nm, nr, 2))                                       # Σdiff, ΣLR
    for j, i in enumerate(pick):
        r = np.asarray(R[j], np.float64)
        a_fit = np.stack([ndimage.correlate(r[..., b], K[b], mode="constant") for b in range(4)], -1)[10:-10, 10:-10]
        lr = np.asarray(lr_all[i])
        f = tile_features(lr, a_fit, tiles["headers"][i], cat, factors)
        rn, mag, hole = f["rnorm"][:, 0], f["mag"][:, 0], f["hole"]
        mi = np.where(np.isfinite(mag), np.clip(np.digitize(np.nan_to_num(mag), MAG_EDGES) - 1, 0, nm - 1), -1)
        ri = np.clip(np.digitize(rn, R_EDGES) - 1, 0, nr - 1)
        for b in range(4):
            nonstar = ~np.isin(f["cls"][:, b], STAR) & (mi >= 0)       # profiles.py: NaN host mag in no bin
            m = nonstar & ~hole[:, b] & np.isfinite(f["diff"][:, b]) & np.isfinite(f["lr"][:, b])
            np.add.at(fsum[b, :, :, 0], (mi[m], ri[m]), f["diff"][m, b])
            np.add.at(fsum[b, :, :, 1], (mi[m], ri[m]), f["lr"][m, b])
            g = np.asarray(G[j, b, :, :, b], np.float64)
            ext = ndimage.correlate(g, blob[..., b], mode="constant")[LO:HI, LO:HI].ravel()
            sl = (slice(LO - TRIM, HI - TRIM),) * 2
            grab = lambda v: v.reshape(235, 235)[sl].ravel()
            gm, gr, gs = grab(mi), grab(ri), grab(nonstar)
            for a_ in range(nm):
                for r_ in range(nr):
                    s = gs & (gm == a_) & (gr == r_)
                    if s.any():
                        gain_vals[b][a_][r_].append(ext[s].astype(np.float32))
        print(label, j, flush=True)
    gmed = np.full((4, nm, nr), np.nan)
    gn = np.zeros((4, nm, nr), int)
    gmed_faint = np.full((4, nr), np.nan)          # host m > 19.5 pooled (the injection test's selection)
    for b in range(4):
        for r_ in range(nr):
            pooled = []
            for a_ in range(nm):
                v = np.concatenate(gain_vals[b][a_][r_]) if gain_vals[b][a_][r_] else np.zeros(0)
                gn[b, a_, r_] = v.size
                if v.size >= 30:
                    gmed[b, a_, r_] = np.median(v)
                if a_ >= 2:
                    pooled.append(v)
            v = np.concatenate(pooled) if pooled else np.zeros(0)
            if v.size >= 30:
                gmed_faint[b, r_] = np.median(v)
    ratio = 1 + fsum[..., 0] / np.where(fsum[..., 1] > 0, fsum[..., 1], np.nan)
    ff = fsum[:, 2:].sum(1)
    ratio_faint = 1 + ff[..., 0] / np.where(ff[..., 1] > 0, ff[..., 1], np.nan)
    res["models"][label] = {"gain_median": gmed.tolist(), "gain_n": gn.tolist(), "ratio": ratio.tolist(),
                            "gain_median_m_gt_19.5": gmed_faint.tolist(), "ratio_m_gt_19.5": ratio_faint.tolist(),
                            "K_centre": K[:, H, H].tolist()}
    print(label, "K centre", K[:, H, H].round(3), flush=True)

json.dump(res, open(f"{runs}/gain_profile.json", "w"), indent=1)
rings = [f"{R_EDGES[i]:g}-{R_EDGES[i + 1]:g}" for i in range(nr)]
print("\n=== NEXUS, host m > 19.5, non-star pixels: blob gain median | A_fit/LR   (r / r_Kron)")
for b in range(4):
    print(f"-- {BAND[b]:<9}" + "".join(f"{r:>13s}" for r in rings))
    for label in MODELS:
        d = res["models"][label]
        print(f"{label:<11}" + "".join(f"  {d['gain_median_m_gt_19.5'][b][r]:5.2f}|{d['ratio_m_gt_19.5'][b][r]:5.2f}"
                                       for r in range(nr)))
print("\n=== 4-band mean, host m > 19.5: gain | ratio")
for label in MODELS:
    d = res["models"][label]
    g = np.nanmean(np.array(d["gain_median_m_gt_19.5"]), 0)
    q = np.nanmean(np.array(d["ratio_m_gt_19.5"]), 0)
    print(f"{label:<11}" + "".join(f"  {g[r]:5.2f}|{q[r]:5.2f}" for r in range(nr)))
print("\n=== per host-mag bin, 4-band mean gain median")
for label in MODELS:
    gm = np.nanmean(np.array(res["models"][label]["gain_median"]), 0)
    for a_ in range(nm):
        print(f"{label:<9} m{MAG_EDGES[a_]:>4g}-{MAG_EDGES[a_ + 1]:<3g}" + "".join(f"{gm[a_, r]:7.2f}" for r in range(nr)))
print("wrote", f"{runs}/gain_profile.json")
