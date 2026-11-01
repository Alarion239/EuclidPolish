"""Empty-sky statistics, real vs synthetic: σ level, pedestal, autocorrelation, block-scatter excess."""
import sys

import numpy as np
from scipy import ndimage

from common import block_factors
from euclid_polish.eval import knee_attribution as ka

out_dir = sys.argv[1]
BAND = ["VIS", "Y", "J", "H"]
fac = block_factors()
for name, path in (("real", "lr.npy"), ("syn", "lr_syn.npy")):
    lr_all = np.load(f"{out_dir}/{path}", mmap_mode="r")
    sig, ped, ac1, ac2, blk4, blk8, mad = [], [], [], [], [], [], []
    for i in range(0, lr_all.shape[0], max(1, lr_all.shape[0] // 100)):
        lr = np.asarray(lr_all[i], np.float64)
        s = ka.noise_sigma(lr)
        sig.append(s)
        row = [[], [], [], [], [], [], []]
        for b in range(4):
            x = lr[..., b]
            # empty sky: far from anything above 2σ after 5-px smoothing, no holes
            src = ndimage.binary_dilation(ndimage.uniform_filter(x, 5) > 2 * s[b] / 5 * 1.5, iterations=6)
            sky = ~src & (x != 0)
            v = np.where(sky, x, np.nan)
            row[0].append(np.nanmedian(v) / s[b])
            row[6].append(1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v))) / s[b])
            vz = np.where(sky, x - np.nanmedian(v), 0.0)
            m = sky.astype(float)
            def ac(dy, dx):
                a = vz[:x.shape[0] - dy, :x.shape[1] - dx] * vz[dy:, dx:]
                w = m[:x.shape[0] - dy, :x.shape[1] - dx] * m[dy:, dx:]
                return a.sum() / w.sum() / np.nanvar(v)
            row[1].append((ac(0, 1) + ac(1, 0)) / 2)
            row[2].append((ac(0, 2) + ac(2, 0)) / 2)
            for k, (sc, fi) in ((3, (4, 2)), (4, (8, 3))):
                h = (x.shape[0] // sc) * sc
                bs = vz[:h, :h].reshape(h // sc, sc, h // sc, sc).sum((1, 3))
                bw = m[:h, :h].reshape(h // sc, sc, h // sc, sc).sum((1, 3))
                full = bw == sc * sc
                row[k].append(np.std(bs[full] / sc ** 2) / (fac[b, fi] * s[b] / fac[b, 0]) if full.sum() > 20 else np.nan)
        for k in range(7):
            [ped, ac1, ac2, blk4, blk8, None, mad][k] is not None and [ped, ac1, ac2, blk4, blk8, None, mad][k].append(row[k])
    sig = np.array(sig)
    print(f"== {name}  (median over tiles)")
    for lab, arr in (("pixel σ e-", sig), ("pedestal/σ", ped), ("MAD/σ", mad), ("autocorr lag1", ac1),
                     ("autocorr lag2", ac2), ("4×4 block σ / model", blk4), ("8×8 block σ / model", blk8)):
        a = np.array(arr, float)
        print(f"  {lab:<22}" + "  ".join(f"{BAND[b]} {np.nanmedian(a[:, b]):7.3f} [{np.nanpercentile(a[:, b], 10):.3f},{np.nanpercentile(a[:, b], 90):.3f}]" for b in range(4)))
