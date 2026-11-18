import sys
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish/output/jwst_triples_257/code")
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from render import triple_images
specs = [a.split(":") for a in sys.argv[2:]]   # tile:cx:cy:hw:q:mode
fig, axs = plt.subplots(len(specs), 3, figsize=(9, 3 * len(specs)), facecolor="k", squeeze=False)
for i, (t, cx, cy, hw, q, mode) in enumerate(specs):
    imgs, m = triple_images(dict(tile=t, cx=int(cx), cy=int(cy), hw=int(hw), q=float(q), mode=mode))
    for k in range(3):
        axs[i, k].imshow(imgs[k], origin="lower", interpolation="nearest" if k == 0 else "lanczos"); axs[i, k].axis("off")
    axs[i, 0].set_title(f"{t} {cx},{cy} hw{hw} q{q} {mode}", color="w", fontsize=8)
fig.savefig(sys.argv[1], dpi=120, facecolor="k")
