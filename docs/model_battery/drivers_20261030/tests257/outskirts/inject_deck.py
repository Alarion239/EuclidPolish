"""inject3.py reduced to two hosts (real σ+corr, training σ+corr); saves per-ring sums and stamps."""
import sys

import numpy as np
import tensorflow as tf
from scipy import ndimage

from common import kernels, load, make_fns
from syn_helpers import R_EDGES, TRIM, bins, maps

out_dir, model, dest = sys.argv[1], sys.argv[2], sys.argv[3]
N, NSTAMP = 40, 10
RC = np.load(f"{out_dir}/RC_syn.npy", mmap_mode="r")
K = np.load(f"{out_dir}/K_syn.npy")
src = dict(np.load(f"{out_dir}/sources_syn.npz"))
SIG = {"real": np.array([18.44, 1.972, 2.124, 2.095]), "syn": np.array([20.89, 2.101, 2.25, 2.165])}
LAG1 = {"real": np.array([0.196, 0.76, 0.762, 0.762]), "syn": np.array([0.243, 0.829, 0.832, 0.826])}
HOSTS = ["real", "syn"]
nr = len(R_EDGES) - 1


def rebin(x):
    return x.reshape(255, 2, 255, 2, 4).sum((1, 3))


net, _, _ = load(model)
fwd, _ = make_fns(net, kernels()[0])
sr = lambda x: rebin(fwd(tf.constant(x[None].astype(np.float32)))[0].numpy()[0])
acc = np.zeros((2, 2, nr, 4))
stamps = np.zeros((NSTAMP, 2, 255 - 2 * TRIM, 255 - 2 * TRIM, 4), np.float32)
truth = np.zeros((NSTAMP, 255 - 2 * TRIM, 255 - 2 * TRIM, 4), np.float32)
for j in range(N):
    rc = np.asarray(RC[j], np.float64)
    inj = np.stack([ndimage.correlate(rc[..., b], K[b], mode="constant") for b in range(4)], -1)
    rn, star, mag = maps(src, j)
    mi, ri = bins(rn, np.nan_to_num(mag, nan=99))
    sel = ~star & (rn < 9) & (mi >= 2)
    tru = rc[TRIM:-TRIM, TRIM:-TRIM]
    white = np.random.default_rng(j).standard_normal((255, 255, 4))
    for h, key in enumerate(HOSTS):
        base = np.empty((255, 255, 4))
        for b in range(4):
            n = ndimage.gaussian_filter(white[..., b], np.sqrt(1 / (4 * -np.log(LAG1[key][b]))), mode="wrap")
            base[..., b] = n / n.std() * SIG[key][b]
        rec = (sr(base + inj) - sr(base))[TRIM:-TRIM, TRIM:-TRIM]
        if j < NSTAMP:
            stamps[j, h] = rec
        for b in range(4):
            np.add.at(acc[h, 0, :, b], ri[sel], rec[..., b][sel])
            np.add.at(acc[h, 1, :, b], ri[sel], tru[..., b][sel])
    if j < NSTAMP:
        truth[j] = tru
    print(j, flush=True)
np.savez(dest, acc=acc, stamps=stamps, truth=truth, r_edges=np.array(R_EDGES))
print("done", model)
