"""Re-observe the SR with a fitted, unit-sum, per-band LR-grid PSF instead of the mean ePSF.

R = rebin2(SR) (LR grid).  Per model and band, K (21×21, ΣK = 1) minimises
Σ (LR − K ⊛ R)² / level² over every tile pixel (normal equations).  Writes A_fit_<label>.npy
in the same (n, 235, 235, 4) layout as A_<label>.npy, plus the kernels.
"""
import sys
import time

import numpy as np
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

from common import block_factors, kernels, load, make_fns
from euclid_polish.eval import knee_attribution as ka

out_dir = sys.argv[1]
MODELS = ("member_238", "member_249")
H = 10                                     # kernel half-size (LR px)
lr_all = np.load(f"{out_dir}/lr.npy", mmap_mode="r")
n = lr_all.shape[0]
factors = block_factors()
kernel, _ = kernels()

for label in MODELS:
    path = f"{out_dir}/R_{label}.npy"
    try:
        R = np.load(path, mmap_mode="r")
    except FileNotFoundError:
        net, _, _ = load(label)
        fwd, _ = make_fns(net, kernel)
        R = np.lib.format.open_memmap(path, "w+", np.float32, lr_all.shape)
        for i in range(n):
            sr, _ = fwd(tf.constant(lr_all[i][None]))
            s = sr.numpy()[0]
            R[i] = s.reshape(255, 2, 255, 2, 4).sum((1, 3))
        R.flush()
        print(label, "SR rebinned", flush=True)
    # normal equations, centre-substituted for ΣK = 1
    p = (2 * H + 1) ** 2
    c = p // 2
    XtX = np.zeros((4, p - 1, p - 1))
    Xty = np.zeros((4, p - 1))
    t0 = time.time()
    for i in range(0, n, 2):
        lr = np.asarray(lr_all[i], np.float64)
        r = np.asarray(R[i], np.float64)
        sky = ka.noise_sigma(lr) / factors[:, 0]
        for b in range(4):
            win = sliding_window_view(r[..., b], (2 * H + 1, 2 * H + 1)).reshape(-1, p)
            y = lr[H:-H, H:-H, b].ravel()
            ok = np.isfinite(y) & (y != 0)
            w = 1.0 / (sky[b] ** 2 + np.clip(y, 0, None))
            X = (np.delete(win, c, 1) - win[:, c:c + 1])[ok]
            t = (y - win[:, c])[ok]
            sw = w[ok]
            XtX[b] += X.T @ (X * sw[:, None])
            Xty[b] += X.T @ (t * sw)
        if i % 50 == 0:
            print(label, "fit", i, f"{time.time() - t0:.0f}s", flush=True)
    K = np.zeros((4, 2 * H + 1, 2 * H + 1))
    for b in range(4):
        k = np.linalg.solve(XtX[b], Xty[b])
        full = np.insert(k, c, 1.0 - k.sum())
        K[b] = full.reshape(2 * H + 1, 2 * H + 1)
    np.save(f"{out_dir}/K_{label}.npy", K)
    print(label, "kernel centre", K[:, H, H].round(3), "sum", K.sum((1, 2)).round(4),
          "frac in r<=2", [round(float(K[b, H - 2:H + 3, H - 2:H + 3].sum()), 3) for b in range(4)], flush=True)
    A = np.lib.format.open_memmap(f"{out_dir}/A_fit_{label}.npy", "w+", np.float32, (n, 235, 235, 4))
    for i in range(n):
        r = np.asarray(R[i], np.float64)
        A[i] = np.stack([ndimage.correlate(r[..., b], K[b], mode="constant") for b in range(4)],
                        -1)[10:-10, 10:-10]
    A.flush()
    del A
print("done", flush=True)
