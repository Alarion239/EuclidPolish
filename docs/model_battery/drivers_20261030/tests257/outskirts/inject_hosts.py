"""inject.py (10-28) for one model: the same 40 noise-free synthetic galaxies (K_syn ⊛ RC_j) pasted into
REAL NEXUS tiles and into other SYNTHETIC test fields (j+50: true v5 training noise + neighbours).

python inject_hosts.py UNDERFLUX_OUT SPEC DEST.npz
Saves inject.py's arrays (real_rec, real_tru, syn_rec, syn_tru: (mag bin, ring, band)) and per-field
sums for host mag > 19.5 (acc_f: (N, 2 hosts, 2 [rec, truth], ring, band)); S/N tables dropped.
"""
import sys

import numpy as np
import tensorflow as tf
from scipy import ndimage

from common import kernels, load, make_fns
from syn_helpers import MAG_EDGES, R_EDGES, TRIM, bins, maps

out_dir, spec, dest = sys.argv[1], sys.argv[2], sys.argv[3]
N = 40
lr_real = np.load(f"{out_dir}/lr.npy", mmap_mode="r")
lr_syn = np.load(f"{out_dir}/lr_syn.npy", mmap_mode="r")
RC = np.load(f"{out_dir}/RC_syn.npy", mmap_mode="r")
K = np.load(f"{out_dir}/K_syn.npy")
src = dict(np.load(f"{out_dir}/sources_syn.npz"))
real_hosts = np.linspace(0, lr_real.shape[0] - 1, N).astype(int)
kernel, _ = kernels()
nm, nr = len(MAG_EDGES) - 1, len(R_EDGES) - 1


def rebin(x):
    return x.reshape(255, 2, 255, 2, 4).sum((1, 3))


net, _, _ = load(spec)
fwd, _ = make_fns(net, kernel)
sr = lambda x: rebin(fwd(tf.constant(x[None].astype(np.float32)))[0].numpy()[0])
acc = {h: {k: np.zeros((nm, nr, 4)) for k in ("rec", "tru")} for h in ("real", "syn")}
acc_f = np.zeros((N, 2, 2, nr, 4))
for n, j in enumerate(range(N)):
    rc = np.asarray(RC[j], np.float64)
    inj = np.stack([ndimage.correlate(rc[..., b], K[b], mode="constant") for b in range(4)], -1)
    rn, star, mag = maps(src, j)
    mi, ri = bins(rn, np.nan_to_num(mag, nan=99))
    gal = ~star & (rn < 9)
    tru = rc[TRIM:-TRIM, TRIM:-TRIM]
    for h, (host, base) in enumerate((("real", np.asarray(lr_real[real_hosts[n]], np.float64)),
                                      ("syn", np.asarray(lr_syn[j + 50], np.float64)))):
        ok_host = base != 0
        rec = (sr(np.where(ok_host, base + inj, 0.0)) - sr(base))[TRIM:-TRIM, TRIM:-TRIM]
        hole = ~ok_host[TRIM:-TRIM, TRIM:-TRIM]
        for b in range(4):
            ok = gal & ~hole[..., b]
            idx = (mi[ok], ri[ok])
            np.add.at(acc[host]["rec"][..., b], idx, rec[..., b][ok])
            np.add.at(acc[host]["tru"][..., b], idx, tru[..., b][ok])
            f = ok & (mi >= 2)
            np.add.at(acc_f[n, h, 0, :, b], ri[f], rec[..., b][f])
            np.add.at(acc_f[n, h, 1, :, b], ri[f], tru[..., b][f])
    print(spec, n, flush=True)
np.savez(dest, acc_f=acc_f, r_edges=np.asarray(R_EDGES),
         **{f"{h}_{k}": v for h, d in acc.items() for k, v in d.items()})
print("done", spec, flush=True)
