"""σ vs pixel-correlation scan with Gaussian hosts (member_238, same injected galaxies)."""
import sys

import numpy as np
import tensorflow as tf
from scipy import ndimage

from common import kernels, load, make_fns
from syn_helpers import MAG_EDGES, R_EDGES, TRIM, bins, maps

out_dir = sys.argv[1]
model = sys.argv[2] if len(sys.argv) > 2 else "member_238"
BAND = ["VIS", "Y", "J", "H"]
N = 40
RC = np.load(f"{out_dir}/RC_syn.npy", mmap_mode="r")
K = np.load(f"{out_dir}/K_syn.npy")
src = dict(np.load(f"{out_dir}/sources_syn.npz"))
SIG = {"real": np.array([18.44, 1.972, 2.124, 2.095]), "syn": np.array([20.89, 2.101, 2.25, 2.165])}
LAG1 = {"real": np.array([0.196, 0.76, 0.762, 0.762]), "syn": np.array([0.243, 0.829, 0.832, 0.826])}
nm, nr = len(MAG_EDGES) - 1, len(R_EDGES) - 1
HOSTS = {"σreal corr_real": ("real", "real", 1.0), "σsyn corr_real": ("syn", "real", 1.0),
         "σreal corr_syn": ("real", "syn", 1.0), "σsyn corr_syn": ("syn", "syn", 1.0),
         "σsyn×0.8 corr_syn": ("syn", "syn", 0.8), "σsyn×1.25 corr_syn": ("syn", "syn", 1.25),
         "σsyn×1.5 corr_syn": ("syn", "syn", 1.5)}


def rebin(x):
    return x.reshape(255, 2, 255, 2, 4).sum((1, 3))


net, _, _ = load(model)
fwd, _ = make_fns(net, kernels()[0])
sr = lambda x: rebin(fwd(tf.constant(x[None].astype(np.float32)))[0].numpy()[0])
acc = {h: np.zeros((2, nr, 4)) for h in HOSTS}
for j in range(N):
    rc = np.asarray(RC[j], np.float64)
    inj = np.stack([ndimage.correlate(rc[..., b], K[b], mode="constant") for b in range(4)], -1)
    rn, star, mag = maps(src, j)
    mi, ri = bins(rn, np.nan_to_num(mag, nan=99))
    sel = ~star & (rn < 9) & (mi >= 2)
    tru = rc[TRIM:-TRIM, TRIM:-TRIM]
    white = np.random.default_rng(j).standard_normal((255, 255, 4))   # same draw for every host
    for h, (sk, ck, scale) in HOSTS.items():
        base = np.empty((255, 255, 4))
        for b in range(4):
            n = ndimage.gaussian_filter(white[..., b], np.sqrt(1 / (4 * -np.log(LAG1[ck][b]))), mode="wrap")
            base[..., b] = n / n.std() * SIG[sk][b] * scale
        rec = (sr(base + inj) - sr(base))[TRIM:-TRIM, TRIM:-TRIM]
        for b in range(4):
            np.add.at(acc[h][0, :, b], ri[sel], rec[..., b][sel])
            np.add.at(acc[h][1, :, b], ri[sel], tru[..., b][sel])
    print(j, flush=True)
print(model, "recovered fraction, injected hosts m>19.5, by r/r_Kron")
for b in range(4):
    print(f"-- {BAND[b]:<17}" + " ".join(f"{R_EDGES[y]:>4}-{R_EDGES[y+1]:<4}" for y in range(nr)))
    for h in HOSTS:
        print(f"  {h:<17}" + " ".join(f"{acc[h][0, y, b] / acc[h][1, y, b]:9.2f}" for y in range(nr)))
