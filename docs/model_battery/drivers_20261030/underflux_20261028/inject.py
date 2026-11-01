"""Injection: noise-free synthetic galaxies pasted into real tiles vs into other synthetic fields.

python inject.py OUT_DIR
inj_j = K_syn ⊛ RC_j (synthetic field j's starless truth seen by Euclid, no noise).  Recovered fraction
= Σ [rebin SR(host + inj_j) − rebin SR(host)] / Σ RC_j, per r/r_Kron × host-mag bin of field j's own
(galaxy) sources, and per noise-free S/N.  Hosts: real tile k (real noise, real neighbours) or
synthetic field j+50 (training-like noise and neighbours).
"""
import sys

import numpy as np
import tensorflow as tf
from scipy import ndimage

from common import block_factors, kernels, load, make_fns
from syn_helpers import MAG_EDGES, R_EDGES, SNR_EDGES, TRIM, bins, maps, snr_of

out_dir = sys.argv[1]
MODELS = ("member_238", "member_249")
BAND = ["VIS", "Y", "J", "H"]
N = 40
lr_real = np.load(f"{out_dir}/lr.npy", mmap_mode="r")
lr_syn = np.load(f"{out_dir}/lr_syn.npy", mmap_mode="r")
RC = np.load(f"{out_dir}/RC_syn.npy", mmap_mode="r")
K = np.load(f"{out_dir}/K_syn.npy")
src = dict(np.load(f"{out_dir}/sources_syn.npz"))
real_hosts = np.linspace(0, lr_real.shape[0] - 1, N).astype(int)
kernel, _ = kernels()
nm, nr, ns = len(MAG_EDGES) - 1, len(R_EDGES) - 1, len(SNR_EDGES) - 1


def rebin(x):
    return x.reshape(255, 2, 255, 2, 4).sum((1, 3))


for label in MODELS:
    net, _, _ = load(label)
    fwd, _ = make_fns(net, kernel)
    sr = lambda x: rebin(fwd(tf.constant(x[None].astype(np.float32)))[0].numpy()[0])
    acc = {h: {k: np.zeros(s) for k, s in (("rec", (nm, nr, 4)), ("tru", (nm, nr, 4)),
                                             ("srec", (ns, 4)), ("stru", (ns, 4)))}
           for h in ("real", "syn")}
    for n, j in enumerate(range(N)):
        rc = np.asarray(RC[j], np.float64)
        inj = np.stack([ndimage.correlate(rc[..., b], K[b], mode="constant") for b in range(4)], -1)
        rn, star, mag = maps(src, j)
        mi, ri = bins(rn, np.nan_to_num(mag, nan=99))
        gal = ~star & (rn < 9)
        tru = rc[TRIM:-TRIM, TRIM:-TRIM]
        for host, base in (("real", np.asarray(lr_real[real_hosts[n]], np.float64)),
                           ("syn", np.asarray(lr_syn[j + 50], np.float64))):
            ok_host = base != 0
            rec = (sr(np.where(ok_host, base + inj, 0.0)) - sr(base))[TRIM:-TRIM, TRIM:-TRIM]
            s_t, _ = snr_of(inj[TRIM:-TRIM, TRIM:-TRIM], base)
            hole = ~ok_host[TRIM:-TRIM, TRIM:-TRIM]
            for b in range(4):
                ok = gal & ~hole[..., b]
                idx = (mi[ok], ri[ok])
                np.add.at(acc[host]["rec"][..., b], idx, rec[..., b][ok])
                np.add.at(acc[host]["tru"][..., b], idx, tru[..., b][ok])
                tb = np.clip(np.digitize(s_t[..., b][ok], SNR_EDGES) - 1, 0, ns - 1)
                np.add.at(acc[host]["srec"][:, b], tb, rec[..., b][ok])
                np.add.at(acc[host]["stru"][:, b], tb, tru[..., b][ok])
        print(label, n, flush=True)
    np.savez(f"{out_dir}/inject_{label}.npz", **{f"{h}_{k}": v for h, d in acc.items() for k, v in d.items()})
    with np.errstate(divide="ignore", invalid="ignore"):
        print(f"\n===== {label}: injected-galaxy flux recovered  real host | synthetic host   rows inj. host mag")
        for b in range(4):
            print(f"-- {BAND[b]:<4}       " + " ".join(f"{R_EDGES[y]:>4}-{R_EDGES[y+1]:<4}" for y in range(nr)))
            for x in range(nm):
                cells = []
                for y in range(nr):
                    t = acc["real"]["tru"][x, y, b]
                    if t <= 0 or acc["syn"]["tru"][x, y, b] <= 0:
                        cells.append("      .  ")
                        continue
                    cells.append(f"{acc['real']['rec'][x, y, b] / t:4.2f}|"
                                 f"{acc['syn']['rec'][x, y, b] / acc['syn']['tru'][x, y, b]:4.2f}")
                print(f"m{MAG_EDGES[x]:>4}-{MAG_EDGES[x+1]:<4} " + " ".join(f"{c:>9}" for c in cells))
        print(f"-- by noise-free S/N per 3×3 of the injected light: recovered real | syn  (truth share)")
        for b in range(4):
            tot = acc["syn"]["stru"][:, b].sum()
            print(BAND[b], "  ".join(
                f"{SNR_EDGES[x]:g}:{acc['real']['srec'][x, b] / acc['real']['stru'][x, b]:4.2f}|"
                f"{acc['syn']['srec'][x, b] / acc['syn']['stru'][x, b]:4.2f}({100 * acc['syn']['stru'][x, b] / tot:.0f}%)"
                for x in range(1, ns - 1)))
print("done")
