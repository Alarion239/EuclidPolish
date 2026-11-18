"""Single-band Euclid / POLISHed / JWST previews (k: 0 VIS..3 H), shared asinh stretch."""
import sys
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish/output/jwst_triples_257/code")
sys.path.insert(0, "/Users/alarion239/Desktop/EuclidPolish")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from render import cubes

def band_imgs(tile, cx, cy, hw, k, q=30.0, pct=99.7):
    lr, sr, jw = cubes(tile)
    sl = (slice(cy - hw, cy + hw), slice(cx - hw, cx + hw))
    ims = [c[sl][..., k] for c in (lr, sr, jw)]
    white = np.nanpercentile(ims[2], pct); soft = white / q
    f = lambda x: np.clip(np.arcsinh(np.maximum(np.nan_to_num(x), 0) / soft) / np.arcsinh(white / soft), 0, 1)
    return [f(x) for x in ims]

if __name__ == "__main__":
    specs = [a.split(":") for a in sys.argv[2:]]
    fig, axs = plt.subplots(len(specs), 3, figsize=(9, 3 * len(specs)), facecolor="k", squeeze=False)
    for i, (t, cx, cy, hw, k) in enumerate(specs):
        ims = band_imgs(t, int(cx), int(cy), int(hw), int(k))
        for j in range(3):
            axs[i, j].imshow(ims[j], origin="lower", cmap="gray", interpolation="nearest" if j == 0 else "lanczos"); axs[i, j].axis("off")
        axs[i, 0].set_title(f"{t} {cx},{cy} hw{hw} band{k}", color="w", fontsize=8)
    fig.savefig(sys.argv[1], dpi=110, facecolor="k")
