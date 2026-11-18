"""Preview whole NEXUS tiles (Euclid | POLISHed | JWST) to pick the cold-open tile."""
import sys
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish/output/jwst_triples_257/code")
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from render import triple_images
tiles = sys.argv[1:]
fig, axs = plt.subplots(len(tiles), 3, figsize=(9, 3 * len(tiles)), facecolor="k")
for i, t in enumerate(tiles):
    for mode in ("full", "vy"):
        try:
            imgs, m = triple_images(dict(tile=t, cx=255, cy=255, hw=255, mode=mode, q=60))
            if mode == "full" and np.mean(imgs[2] == 0.12) > 0.02:
                continue
            break
        except Exception as e:
            print(t, mode, e)
    for k in range(3):
        axs[i, k].imshow(imgs[k], origin="lower"); axs[i, k].axis("off")
    axs[i, 0].set_title(f"{t} ({m})", color="w", fontsize=8)
fig.savefig(sys.argv[0].replace(".py", ".png"), dpi=110, facecolor="k")
