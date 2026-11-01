"""Which part of the real background kills injected outskirts?  member_238, same 40 injected galaxies.

Hosts: real tile | synthetic field | synthetic noise only (LR − K⊛truth) | Gaussian noise matched to
real σ + pixel correlation | Gaussian matched to synthetic | real tile with every pixel's sign... (none).
"""
import sys

import numpy as np
import tensorflow as tf
from scipy import ndimage

from common import kernels, load, make_fns
from syn_helpers import MAG_EDGES, R_EDGES, TRIM, bins, maps

out_dir = sys.argv[1]
BAND = ["VIS", "Y", "J", "H"]
N = 40
lr_real = np.load(f"{out_dir}/lr.npy", mmap_mode="r")
lr_syn = np.load(f"{out_dir}/lr_syn.npy", mmap_mode="r")
RC = np.load(f"{out_dir}/RC_syn.npy", mmap_mode="r")
RT = np.load(f"{out_dir}/RT_syn.npy", mmap_mode="r")
K = np.load(f"{out_dir}/K_syn.npy")
src = dict(np.load(f"{out_dir}/sources_syn.npz"))
real_hosts = np.linspace(0, lr_real.shape[0] - 1, N).astype(int)
SIG = {"real": np.array([18.44, 1.972, 2.124, 2.095]), "syn": np.array([20.89, 2.101, 2.25, 2.165])}
LAG1 = {"real": np.array([0.196, 0.76, 0.762, 0.762]), "syn": np.array([0.243, 0.829, 0.832, 0.826])}
rng = np.random.default_rng(1)
nm, nr = len(MAG_EDGES) - 1, len(R_EDGES) - 1


def rebin(x):
    return x.reshape(255, 2, 255, 2, 4).sum((1, 3))


def observe(r):
    return np.stack([ndimage.correlate(r[..., b], K[b], mode="constant") for b in range(4)], -1)


def gauss(kind):
    out = np.empty((255, 255, 4))
    for b in range(4):
        sk = np.sqrt(1 / (4 * -np.log(LAG1[kind][b])))
        n = ndimage.gaussian_filter(rng.standard_normal((255, 255)), sk, mode="wrap")
        out[..., b] = n / n.std() * SIG[kind][b]
    return out


net, _, _ = load("member_238")
fwd, _ = make_fns(net, kernels()[0])
sr = lambda x: rebin(fwd(tf.constant(x[None].astype(np.float32)))[0].numpy()[0])
HOSTS = ("real", "syn", "synnoise", "gauss_real", "gauss_syn")
acc = {h: np.zeros((2, nm, nr, 4)) for h in HOSTS}
for j in range(N):
    rc = np.asarray(RC[j], np.float64)
    inj = observe(rc)
    rn, star, mag = maps(src, j)
    mi, ri = bins(rn, np.nan_to_num(mag, nan=99))
    gal = ~star & (rn < 9)
    tru = rc[TRIM:-TRIM, TRIM:-TRIM]
    k = j + 50
    syn_lr = np.asarray(lr_syn[k], np.float64)
    hosts = {"real": np.asarray(lr_real[real_hosts[j]], np.float64), "syn": syn_lr,
             "synnoise": np.where(syn_lr != 0, syn_lr - observe(np.asarray(RT[k], np.float64)), 0.0),
             "gauss_real": gauss("real"), "gauss_syn": gauss("syn")}
    for h, base in hosts.items():
        ok_host = base != 0
        rec = (sr(np.where(ok_host, base + inj, 0.0)) - sr(base))[TRIM:-TRIM, TRIM:-TRIM]
        hole = ~ok_host[TRIM:-TRIM, TRIM:-TRIM]
        for b in range(4):
            ok = gal & ~hole[..., b]
            np.add.at(acc[h][0, ..., b], (mi[ok], ri[ok]), rec[..., b][ok])
            np.add.at(acc[h][1, ..., b], (mi[ok], ri[ok]), tru[..., b][ok])
    print(j, flush=True)
np.savez(f"{out_dir}/inject2.npz", **acc)
# collapse hosts mag 19.5–99 (the injected galaxies with outskirts inside the frame)
print("recovered fraction, injected hosts m>19.5, by r/r_Kron")
for b in range(4):
    print(f"-- {BAND[b]:<10}" + " ".join(f"{R_EDGES[y]:>4}-{R_EDGES[y+1]:<4}" for y in range(nr)))
    for h in HOSTS:
        a = acc[h][:, 2:].sum(1)
        print(f"  {h:<10}" + " ".join(f"{a[0, y, b] / a[1, y, b]:9.2f}" for y in range(nr)))
