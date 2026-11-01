import math
import sys

import numpy as np

from euclid_polish.config import Config
from mer_fair import stack   # noqa: E402

band = Config.get_band("Y_E")
rng = np.random.default_rng(7)


def plaid(u):
    v = np.array([[u[i::3, j::3].var() for j in range(3)] for i in range(3)])
    return v.std() / v.mean()


for combine, ns in (("mean", (4, 6, 8)), ("median", (3, 4, 5, 6, 8))):
    for n in ns:
        p = [plaid(stack(band, rng, n, combine, (510, 510))) for _ in range(20)]
        print(f"{combine:<6} n={n}: plaid {100 * np.median(p):5.1f}%  [{100 * np.percentile(p, 16):.1f},{100 * np.percentile(p, 84):.1f}]")
