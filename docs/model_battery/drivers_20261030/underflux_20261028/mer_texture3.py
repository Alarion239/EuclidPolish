import math
import sys

import numpy as np

sys.argv = ["x"]
from mer_texture2 import sim, texture   # noqa: E402  (re-runs the real table; fine)

rng = np.random.default_rng(5)
print("\nsim: lag(0,1) (1,0) (1,1) (1,-1) (0,2) (2,0) | ap4 ap8 | σpix  σ8(=σpix·√ap8) in units of the MEAN stack's large-scale σ")
for name, factor in (("VIS", 1), ("NISP", 3)):
    for n in (3, 4, 5, 6, 7, 8, 10, 12):
        for combine in ("mean", "median"):
            res, sp = [], []
            for _ in range(5):
                x = sim(factor, 0, n, combine, rng) * math.sqrt(n) / factor
                t, s = texture(x, np.ones(x.shape, bool))
                res.append(t)
                sp.append(s)
            t = np.mean(res, 0)
            s = np.mean(sp)
            print(f"sim {name:<4} n={n:>2} {combine:<6} " + " ".join(f"{v:6.3f}" for v in t) + f" | σpix {s:.3f} σ8 {s * math.sqrt(t[-1]):.3f}")
