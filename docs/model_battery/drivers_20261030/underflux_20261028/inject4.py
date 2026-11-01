"""Real hosts as-is vs lightly smoothed so their noise lag-1 correlation matches the training noise."""
import sys

import numpy as np
import tensorflow as tf
from scipy import ndimage

from common import kernels, load, make_fns
from syn_helpers import MAG_EDGES, R_EDGES, TRIM, bins, maps

out_dir = sys.argv[1]
model = sys.argv[2]
BAND = ["VIS", "Y", "J", "H"]
N = 40
lr_real = np.load(f"{out_dir}/lr.npy", mmap_mode="r")
RC = np.load(f"{out_dir}/RC_syn.npy", mmap_mode="r")
K = np.load(f"{out_dir}/K_syn.npy")
src = dict(np.load(f"{out_dir}/sources_syn.npz"))
hosts_i = np.linspace(0, lr_real.shape[0] - 1, N).astype(int)
gs = lambda r: 1 / (4 * -np.log(np.array(r)))
SIG_ADD = np.sqrt(np.clip(gs([0.243, 0.829, 0.832, 0.826]) - gs([0.196, 0.76, 0.762, 0.762]), 0, None))
print("added smoothing σ (LR px)", SIG_ADD.round(3))
nr = len(R_EDGES) - 1


def rebin(x):
    return x.reshape(255, 2, 255, 2, 4).sum((1, 3))


def smooth(x):
    hole = x == 0
    out = np.stack([ndimage.gaussian_filter(x[..., b], SIG_ADD[b]) for b in range(4)], -1)
    return np.where(hole, 0.0, out)


net, _, _ = load(model)
fwd, _ = make_fns(net, kernels()[0])
sr = lambda x: rebin(fwd(tf.constant(x[None].astype(np.float32)))[0].numpy()[0])
acc = {h: np.zeros((2, nr, 4)) for h in ("real", "real smoothed")}
for j in range(N):
    rc = np.asarray(RC[j], np.float64)
    inj = np.stack([ndimage.correlate(rc[..., b], K[b], mode="constant") for b in range(4)], -1)
    rn, star, mag = maps(src, j)
    mi, ri = bins(rn, np.nan_to_num(mag, nan=99))
    sel = ~star & (rn < 9) & (mi >= 2)
    tru = rc[TRIM:-TRIM, TRIM:-TRIM]
    raw = np.asarray(lr_real[hosts_i[j]], np.float64)
    for h, base in (("real", raw), ("real smoothed", smooth(raw))):
        ok_host = base != 0
        rec = (sr(np.where(ok_host, base + inj, 0.0)) - sr(base))[TRIM:-TRIM, TRIM:-TRIM]
        hole = ~ok_host[TRIM:-TRIM, TRIM:-TRIM]
        for b in range(4):
            ok = sel & ~hole[..., b]
            np.add.at(acc[h][0, :, b], ri[ok], rec[..., b][ok])
            np.add.at(acc[h][1, :, b], ri[ok], tru[..., b][ok])
print(model, "recovered fraction of injected galaxies (m>19.5) by r/r_Kron")
for b in range(4):
    print(f"-- {BAND[b]:<14}" + " ".join(f"{R_EDGES[y]:>4}-{R_EDGES[y+1]:<4}" for y in range(nr)))
    for h in acc:
        print(f"  {h:<14}" + " ".join(f"{acc[h][0, y, b] / acc[h][1, y, b]:9.2f}" for y in range(nr)))
