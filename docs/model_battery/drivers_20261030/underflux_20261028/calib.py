import math

import numpy as np

from euclid_polish.config import Config
from mer_fair import stack


def lags(u):
    z = u - u.mean()
    v = (z * z).mean()
    a = lambda dy, dx: (z[:z.shape[0] - dy, :z.shape[1] - dx] * z[dy:, dx:]).mean() / v
    return (a(0, 1) + a(1, 0)) / 2, a(1, 1), (a(0, 2) + a(2, 0)) / 2


def large(u, box=48):
    h = u.shape[0] // box * box
    return math.sqrt(u[:h, :h].reshape(h // box, box, h // box, box).sum((1, 3)).var() / box ** 2)


print("real Q1 (fair): VIS lag1 .205 diag .051 lag2 .001 σpix/RMS .666 | NISP .80 .673 .497 σpix/RMS .231")
for name in ("VIS", "Y_E"):
    band = Config.get_band(name)
    for combine, ns in (("mean", (4,)), ("median", (2, 3, 4, 5, 6, 8))):
        for n in ns:
            rng = np.random.default_rng(11)
            fs = [stack(band, rng, n, combine, (576, 576)) for _ in range(6)]
            L = np.mean([lags(f) for f in fs], 0)
            sp = np.mean([f.std() for f in fs])
            sl = math.sqrt(np.mean([large(f) ** 2 for f in fs]))
            print(f"{name:<4} {combine:<6} n={n}: lag1 {L[0]:.3f} diag {L[1]:.3f} lag2 {L[2]:.3f} | σpix {sp:.3f} σlarge {sl:.3f} "
                  f"σpix/σlarge {sp / sl:.3f}")
