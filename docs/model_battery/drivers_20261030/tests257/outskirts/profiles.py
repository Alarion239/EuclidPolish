"""A/LR and backprop gain vs r/r_Kron, split by host magnitude and star/galaxy.  python profiles.py OUT_DIR"""
import sys

import numpy as np

from features import CLASSES

out_dir = sys.argv[1]
MODELS = ("member_238", "member_249")
BAND = ["VIS", "Y", "J", "H"]
R_EDGES = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 9.1]
MAG_EDGES = [0, 18, 19.5, 21, 22.5, 24, 99]
STAR = {CLASSES.index("star core"), CLASSES.index("star wing")}
for label in MODELS:
    d = dict(np.load(f"{out_dir}/sample_A_fit_{label}.npz"))
    g = dict(np.load(f"{out_dir}/gain_{label}.npz"))
    star = np.isin(d["cls"], list(STAR))
    gstar = np.isin(g["cls"], list(STAR))
    for kind, sel, gsel in (("galaxies", ~star, ~gstar), ("stars", star, gstar)):
        print(f"===== {label} {kind}: A_fit/LR (flux-weighted) | gain_ext median  — rows host VIS mag, cols r/r_Kron")
        for b in range(4):
            print(f"-- {BAND[b]}   " + " ".join(f"{R_EDGES[j]:>4}-{R_EDGES[j+1]:<4}" for j in range(len(R_EDGES) - 1)))
            for lo, hi in zip(MAG_EDGES[:-1], MAG_EDGES[1:]):
                cells = []
                for rlo, rhi in zip(R_EDGES[:-1], R_EDGES[1:]):
                    m = (sel & (d["band"] == b) & ~d["hole"] & (d["mag"] >= lo) & (d["mag"] < hi)
                         & (d["rnorm"] >= rlo) & (d["rnorm"] < rhi))
                    gm = (gsel & (g["band"] == b) & (g["mag"] >= lo) & (g["mag"] < hi)
                          & (g["rnorm"] >= rlo) & (g["rnorm"] < rhi))
                    w = d["w"][m]
                    if m.sum() < 30 or (w * d["lr"][m]).sum() <= 0:
                        cells.append("    .    ")
                        continue
                    ratio = 1 + (w * d["diff"][m]).sum() / (w * d["lr"][m]).sum()
                    gain = np.median(g["ext"][gm]) if gm.sum() >= 30 else np.nan
                    cells.append(f"{ratio:4.2f}|{gain:4.2f}")
                print(f"m{lo:>4}-{hi:<3} " + " ".join(cells))
