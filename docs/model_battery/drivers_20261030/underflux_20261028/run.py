"""Phase 1: re-observed SR on every NEXUS tile. Phase 2: flux-gain maps (backprop) on a subset.

    python run.py OUT_DIR [--grad-every 10]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from astropy.wcs import WCS

from common import kernels, load, make_fns
from euclid_polish.web.helpers import real_tiles as rt

MODELS = ("member_238", "member_249")

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--grad-every", type=int, default=10)
p.add_argument("--phase", default="12")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

entries = [e for e in rt.list_entries("nexus") if e.extras.get("jwst_file")]
n = len(entries)
kernel, _ = kernels()

if "1" in args.phase:
    lr_all = np.lib.format.open_memmap(f"{args.out}/lr.npy", "w+", np.float32, (n, 255, 255, 4))
    headers = []
    for i, e in enumerate(entries):
        lr_all[i] = rt.get_tile("nexus", e.id, entry=e).lr_e
        headers.append(WCS(rt.lr_header(e)).celestial.to_header_string())
    lr_all.flush()
    json.dump({"ids": [e.id for e in entries], "headers": headers}, open(f"{args.out}/tiles.json", "w"))
    for label in MODELS:
        net, _, _ = load(label)
        fwd, _ = make_fns(net, kernel)
        a_all = np.lib.format.open_memmap(f"{args.out}/A_{label}.npy", "w+", np.float32, (n, 235, 235, 4))
        t0 = time.time()
        for i in range(n):
            _, a = fwd(tf.constant(lr_all[i][None]))
            a_all[i] = a.numpy()[0]
            if i % 50 == 0:
                print(label, i, f"{time.time() - t0:.0f}s", flush=True)
        a_all.flush()
        del a_all

if "2" in args.phase:
    lr_all = np.load(f"{args.out}/lr.npy", mmap_mode="r")
    pick = np.arange(0, n, args.grad_every)
    np.save(f"{args.out}/grad_tiles.npy", pick)
    for label in MODELS:
        net, _, _ = load(label)
        _, gain = make_fns(net, kernel)
        g_all = np.lib.format.open_memmap(f"{args.out}/G_{label}.npy", "w+", np.float32,
                                          (len(pick), 4, 255, 255, 4))
        t0 = time.time()
        for j, i in enumerate(pick):
            x = tf.constant(lr_all[i][None])
            for b in range(4):
                g_all[j, b] = gain(x, tf.constant(b)).numpy()[0]   # ∂ΣA_b / ∂x[..., c]
            print(label, "grad", j, len(pick), f"{time.time() - t0:.0f}s", flush=True)
        g_all.flush()
        del g_all
print("done", flush=True)
sys.exit(0)
