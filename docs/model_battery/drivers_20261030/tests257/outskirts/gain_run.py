"""Real-sky (NEXUS) backprop flux gain for one model on the 10-28 gradient tiles (run.py phase 2, one model).

python gain_run.py UNDERFLUX_OUT SPEC DEST_PREFIX [LIMIT]
Writes DEST_PREFIX_G.npy (n, 4, 255, 255, 4) = ∂ Σ A(SR)_b / ∂ LR  (exactly run.py's gain()),
DEST_PREFIX_R.npy (n, 255, 255, 4) = 2×2-summed SR and DEST_PREFIX_A.npy (n, 235, 235, 4) = mean-ePSF
re-observation (run.py phase 1's A) for the same tiles; LIMIT keeps the first LIMIT gradient tiles.
"""
import sys
import time

import numpy as np
import tensorflow as tf

from common import kernels, load, make_fns

out_dir, spec, dest = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
lr_all = np.load(f"{out_dir}/lr.npy", mmap_mode="r")
pick = np.load(f"{out_dir}/grad_tiles.npy")[:limit]
kernel, _ = kernels()
net, _, _ = load(spec)
fwd, gain = make_fns(net, kernel)
G = np.lib.format.open_memmap(f"{dest}_G.npy", "w+", np.float32, (len(pick), 4, 255, 255, 4))
R = np.lib.format.open_memmap(f"{dest}_R.npy", "w+", np.float32, (len(pick), 255, 255, 4))
A = np.lib.format.open_memmap(f"{dest}_A.npy", "w+", np.float32, (len(pick), 235, 235, 4))
t0 = time.time()
for j, i in enumerate(pick):
    x = tf.constant(lr_all[i][None])
    sr, a = fwd(x)
    R[j] = sr.numpy()[0].reshape(255, 2, 255, 2, 4).sum((1, 3))
    A[j] = a.numpy()[0]
    for b in range(4):
        G[j, b] = gain(x, tf.constant(b)).numpy()[0]   # ∂ΣA_b / ∂x[..., c]
    print(spec, "grad", j, len(pick), f"{time.time() - t0:.0f}s", flush=True)
for arr in (G, R, A):
    arr.flush()
print("done", spec, flush=True)
