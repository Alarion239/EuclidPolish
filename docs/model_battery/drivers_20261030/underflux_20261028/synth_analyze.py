"""Real vs synthetic: where is flux lost, does the truth have envelopes, and does the net respond there?

python synth_analyze.py OUT_DIR  → prints tables, writes synth_report.npz
Both datasets use the photutils sources of detect.py (galaxies only), the same r/r_Kron and host-mag bins.
"""
import json
import sys

import numpy as np
from scipy import ndimage

from common import block_factors, kernels
from euclid_polish.eval import knee_attribution as ka

out_dir = sys.argv[1]
MODELS = ("member_238", "member_249")
BAND = ["VIS", "Y", "J", "H"]
TRIM = 10
R_EDGES = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 9.1])
MAG_EDGES = np.array([0, 18, 19.5, 21, 22.5, 24, 99])
SNR_EDGES = np.array([-np.inf, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 1e3, np.inf])
LO, HI = 40, 215
factors = block_factors()


def maps(s, i, shape=(235, 235)):
    sel = np.flatnonzero(s["tile"] == i)
    h, w = shape
    best = np.full(shape, np.inf)
    star = np.zeros(shape, bool)
    mag = np.full(shape, np.nan)
    yy, xx = np.mgrid[0:h, 0:w]
    for j in sel[np.argsort(s["mag"][sel])]:
        kron = np.clip(s["kron_px"][j], 1.0, 25.0)
        reach = int(np.ceil(4 * kron + 2))
        x, y = s["x"][j] - TRIM, s["y"][j] - TRIM
        x0, x1 = max(0, int(x) - reach), min(w, int(x) + reach + 2)
        y0, y1 = max(0, int(y) - reach), min(h, int(y) + reach + 2)
        if x0 >= x1 or y0 >= y1:
            continue
        dx, dy = xx[y0:y1, x0:x1] - x, yy[y0:y1, x0:x1] - y
        th = s["theta"][j]
        u = dx * np.cos(th) + dy * np.sin(th)
        v = -dx * np.sin(th) + dy * np.cos(th)
        rn = np.sqrt(u ** 2 + (v / max(1 - s["ellipticity"][j], 0.05)) ** 2) / kron
        better = rn < best[y0:y1, x0:x1]
        best[y0:y1, x0:x1][better] = rn[better]
        star[y0:y1, x0:x1][better] = s["star"][j]
        mag[y0:y1, x0:x1][better] = s["mag"][j]
    return np.minimum(best, 9.0), star, mag


def snr_of(img, lr_full):
    """3×3-mean S/N in the white-noise convention used for the real tables (snr_smooth)."""
    sky = ka.noise_sigma(lr_full) / factors[:, 0]
    level = np.sqrt(sky ** 2 + np.clip(img, 0, None))
    return ndimage.uniform_filter(img, (3, 3, 1)) / (factors[:, 0] * level / 3.0), level


def z4_of(diff, level2):
    hb = 232
    b = lambda v: v[:hb, :hb].reshape(58, 4, 58, 4, 4).mean((1, 3))
    z = np.full(diff.shape, np.nan)
    z[:hb, :hb] = np.repeat(np.repeat(b(diff) / (factors[:, 2] * np.sqrt(b(level2))), 4, 0), 4, 1)
    return z


def bins(rn, mag):
    return (np.clip(np.digitize(mag, MAG_EDGES) - 1, 0, len(MAG_EDGES) - 2),
            np.clip(np.digitize(rn, R_EDGES) - 1, 0, len(R_EDGES) - 2))


kernel, _ = kernels()
k = np.pad(kernel.numpy()[..., 0], ((0, 1), (0, 1), (0, 0)))
psf_lr = k.reshape(21, 2, 21, 2, 4).sum((1, 3))
psf_lr /= psf_lr.sum((0, 1))
blob = np.stack([ndimage.gaussian_filter(np.pad(psf_lr[..., c], 12), 3.0) for c in range(4)], -1)
blob /= blob.sum((0, 1))

nm, nr, ns = len(MAG_EDGES) - 1, len(R_EDGES) - 1, len(SNR_EDGES) - 1
out = {}
for ds in ("real", "syn"):
    src = dict(np.load(f"{out_dir}/sources_{ds}.npz"))
    lr_all = np.load(f"{out_dir}/{'lr' if ds == 'real' else 'lr_syn'}.npy", mmap_mode="r")
    A = {m: np.load(f"{out_dir}/{'A_fit_' if ds == 'real' else 'A_syn_'}{m}.npy", mmap_mode="r") for m in MODELS}
    if ds == "syn":
        AT = np.load(f"{out_dir}/AT_syn.npy", mmap_mode="r")
        RT = np.load(f"{out_dir}/RT_syn.npy", mmap_mode="r")
        R = {m: np.load(f"{out_dir}/R_syn_{m}.npy", mmap_mode="r") for m in MODELS}
    # radial sums [mag, r, band]: LR, A_m, AT, RT, R_m, #px, #z4<-5 (per model)
    prof = {key: np.zeros((nm, nr, 4)) for key in ("lr", "at", "rt", "n")}
    for m in MODELS:
        for key in ("a", "r", "z4neg"):
            prof[f"{key}_{m}"] = np.zeros((nm, nr, 4))
    # S/N-binned sums (galaxy pixels): by LR snr_smooth (both) and by noise-free truth S/N (syn)
    snr = {key: np.zeros((ns, 4)) for key in ("lr", "n", "t_n", "t_at", "t_rt")}
    for m in MODELS:
        for key in ("a", "t_a", "t_r"):
            snr[f"{key}_{m}"] = np.zeros((ns, 4))
    for i in range(lr_all.shape[0]):
        lr_full = np.asarray(lr_all[i], np.float64)
        lr = lr_full[TRIM:-TRIM, TRIM:-TRIM]
        rn, star, mag = maps(src, i)
        hole = (lr == 0) | ~np.isfinite(lr)
        mi, ri = bins(rn, np.nan_to_num(mag, nan=99))
        gal = ~star & (rn < 9)
        s_lr, level = snr_of(lr, lr_full)
        sb = np.clip(np.digitize(s_lr, SNR_EDGES) - 1, 0, ns - 1)
        if ds == "syn":
            at = np.asarray(AT[i], np.float64)
            rt = np.asarray(RT[i], np.float64)[TRIM:-TRIM, TRIM:-TRIM]
            s_t, _ = snr_of(at, lr_full)
            tb = np.clip(np.digitize(s_t, SNR_EDGES) - 1, 0, ns - 1)
        for b in range(4):
            ok = gal & ~hole[..., b]
            idx = (mi[ok], ri[ok])
            np.add.at(prof["lr"][..., b], idx, lr[..., b][ok])
            np.add.at(prof["n"][..., b], idx, 1)
            np.add.at(snr["lr"][:, b], sb[..., b][ok], lr[..., b][ok])
            np.add.at(snr["n"][:, b], sb[..., b][ok], 1)
            if ds == "syn":
                np.add.at(prof["at"][..., b], idx, at[..., b][ok])
                np.add.at(prof["rt"][..., b], idx, rt[..., b][ok])
                np.add.at(snr["t_n"][:, b], tb[..., b][ok], 1)
                np.add.at(snr["t_at"][:, b], tb[..., b][ok], at[..., b][ok])
                np.add.at(snr["t_rt"][:, b], tb[..., b][ok], rt[..., b][ok])
        for m in MODELS:
            a = np.asarray(A[m][i], np.float64)
            z4 = z4_of(a - lr, (level / 1.0) ** 2)
            if ds == "syn":
                r = np.asarray(R[m][i], np.float64)[TRIM:-TRIM, TRIM:-TRIM]
            for b in range(4):
                ok = gal & ~hole[..., b]
                idx = (mi[ok], ri[ok])
                np.add.at(prof[f"a_{m}"][..., b], idx, a[..., b][ok])
                np.add.at(prof[f"z4neg_{m}"][..., b], idx, (z4[..., b][ok] < -5))
                np.add.at(snr[f"a_{m}"][:, b], sb[..., b][ok], a[..., b][ok])
                if ds == "syn":
                    np.add.at(prof[f"r_{m}"][..., b], idx, r[..., b][ok])
                    np.add.at(snr[f"t_a_{m}"][:, b], tb[..., b][ok], a[..., b][ok])
                    np.add.at(snr[f"t_r_{m}"][:, b], tb[..., b][ok], r[..., b][ok])
    # gains: ext-blob gain median per (mag, r) and per S/N bin (LR S/N; truth S/N for syn)
    pick = np.load(f"{out_dir}/grad_tiles.npy") if ds == "real" else np.arange(30)
    for m in MODELS:
        G = np.load(f"{out_dir}/{'G_' if ds == 'real' else 'G_syn_'}{m}.npy", mmap_mode="r")
        rec = {key: [] for key in ("ext", "mi", "ri", "sb", "tb", "band")}
        for j, i in enumerate(pick):
            lr_full = np.asarray(lr_all[i], np.float64)
            lr = lr_full[TRIM:-TRIM, TRIM:-TRIM]
            rn, star, mag = maps(src, i)
            mi, ri = bins(rn, np.nan_to_num(mag, nan=99))
            gal = ~star & (rn < 9)
            s_lr, _ = snr_of(lr, lr_full)
            if ds == "syn":
                s_t, _ = snr_of(np.asarray(AT[i], np.float64), lr_full)
            sl = (slice(LO - TRIM, HI - TRIM),) * 2
            for b in range(4):
                g = np.asarray(G[j, b, :, :, b], np.float64)
                ext = ndimage.correlate(g, blob[..., b], mode="constant")[LO:HI, LO:HI]
                ok = gal[sl] & (lr[sl][..., b] != 0)
                rec["ext"].append(ext[ok])
                rec["mi"].append(mi[sl][ok])
                rec["ri"].append(ri[sl][ok])
                rec["sb"].append(np.clip(np.digitize(s_lr[sl][..., b][ok], SNR_EDGES) - 1, 0, ns - 1))
                rec["tb"].append(np.clip(np.digitize(s_t[sl][..., b][ok], SNR_EDGES) - 1, 0, ns - 1)
                                 if ds == "syn" else np.zeros(ok.sum(), int))
                rec["band"].append(np.full(ok.sum(), b))
        d = {key: np.concatenate(v) for key, v in rec.items()}
        gp = np.full((nm, nr, 4), np.nan)
        gs = np.full((ns, 4), np.nan)
        gt = np.full((ns, 4), np.nan)
        for b in range(4):
            mb = d["band"] == b
            for x in range(nm):
                for y in range(nr):
                    s = mb & (d["mi"] == x) & (d["ri"] == y)
                    if s.sum() >= 30:
                        gp[x, y, b] = np.median(d["ext"][s])
            for x in range(ns):
                s = mb & (d["sb"] == x)
                if s.sum() >= 30:
                    gs[x, b] = np.median(d["ext"][s])
                s = mb & (d["tb"] == x)
                if ds == "syn" and s.sum() >= 30:
                    gt[x, b] = np.median(d["ext"][s])
        prof[f"gain_{m}"], snr[f"gain_{m}"], snr[f"tgain_{m}"] = gp, gs, gt
    out[ds] = (prof, snr)
    print(ds, "done", flush=True)

np.savez(f"{out_dir}/synth_report.npz",
         **{f"{ds}_{kind}_{k}": v for ds, (p, s) in out.items() for kind, dd in (("prof", p), ("snr", s))
            for k, v in dd.items()})


def ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > 0, a / b, np.nan)


def table(title, cells):
    """cells: (mag, r, band) array of strings or floats."""
    print(f"\n===== {title}   rows host VIS mag, cols r/r_Kron")
    for b in range(4):
        print(f"-- {BAND[b]:<4}       " + " ".join(f"{R_EDGES[j]:>4}-{R_EDGES[j+1]:<4}" for j in range(nr)))
        for x in range(nm):
            print(f"m{MAG_EDGES[x]:>4}-{MAG_EDGES[x+1]:<4} " + " ".join(f"{cells[x, y, b]:>9}" for y in range(nr)))


def fmt(v, n, k=30):
    return np.where((n >= k) & np.isfinite(v), np.char.mod("%.2f", np.nan_to_num(v)), ".")


rp, rs = out["real"]
sp, ss = out["syn"]
# envelope content: light in each r bin relative to the r<1 core, same host-mag bin
core = lambda p, key: p[key][:, :2].sum(1, keepdims=True)
table("ENVELOPE  LR(r)/LR(r<1): real | synthetic LR | synthetic truth RT",
      np.char.add(np.char.add(np.char.add(fmt(ratio(rp["lr"], core(rp, "lr")), rp["n"]), "|"),
                               np.char.add(fmt(ratio(sp["lr"], core(sp, "lr")), sp["n"]), "|")),
                  fmt(ratio(sp["rt"], core(sp, "rt")), sp["n"])))
for m in MODELS:
    table(f"{m}  A/LR: real | synthetic",
          np.char.add(np.char.add(fmt(ratio(rp[f"a_{m}"], rp["lr"]), rp["n"]), "|"),
                      fmt(ratio(sp[f"a_{m}"], sp["lr"]), sp["n"])))
    table(f"{m}  synthetic SR/truth (2×2 summed) | AT/LR sanity",
          np.char.add(np.char.add(fmt(ratio(sp[f"r_{m}"], sp["rt"]), sp["n"]), "|"),
                      fmt(ratio(sp["at"], sp["lr"]), sp["n"])))
    table(f"{m}  ext-blob gain median: real | synthetic",
          np.char.add(np.char.add(np.char.mod("%.2f", rp[f"gain_{m}"]), "|"),
                      np.char.mod("%.2f", sp[f"gain_{m}"])))
    table(f"{m}  % px with z4<-5: real | synthetic",
          np.char.add(np.char.add(fmt(100 * ratio(rp[f"z4neg_{m}"], rp["n"]), rp["n"]), "|"),
                      fmt(100 * ratio(sp[f"z4neg_{m}"], sp["n"]), sp["n"])))

print("\n===== by S/N per 3×3 (white-noise units): galaxy-pixel light share (% of ΣLR) real | syn ; A/LR real | syn")
for b in range(4):
    print("--", BAND[b])
    for x in range(ns):
        sh_r = 100 * rs["lr"][x, b] / rs["lr"][:, b].sum()
        sh_s = 100 * ss["lr"][x, b] / ss["lr"][:, b].sum()
        print(f"  snr>{SNR_EDGES[x]:>6}  share {sh_r:5.1f}|{sh_s:5.1f}   " + "  ".join(
            f"{m[-3:]} A/LR {ratio(rs[f'a_{m}'][x, b], rs['lr'][x, b]):5.2f}|{ratio(ss[f'a_{m}'][x, b], ss['lr'][x, b]):5.2f}"
            f" gain {rs[f'gain_{m}'][x, b]:4.2f}|{ss[f'gain_{m}'][x, b]:4.2f}" for m in MODELS))
print("\n===== synthetic by NOISE-FREE truth S/N: truth light share, SR/truth, A/AT, gain")
for b in range(4):
    print("--", BAND[b])
    for x in range(ns):
        sh = 100 * ss["t_rt"][x, b] / ss["t_rt"][:, b].sum()
        print(f"  snr>{SNR_EDGES[x]:>6}  n {int(ss['t_n'][x, b]):>8}  truth share {sh:5.1f}  " + "  ".join(
            f"{m[-3:]} SR/T {ratio(ss[f't_r_{m}'][x, b], ss['t_rt'][x, b]):5.2f} A/AT "
            f"{ratio(ss[f't_a_{m}'][x, b], ss['t_at'][x, b]):5.2f} gain {ss[f'tgain_{m}'][x, b]:4.2f}"
            for m in MODELS))
print("done")
