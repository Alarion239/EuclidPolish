"""F6: Euclid's PSF is sharp and stable. Mean empirical ePSFs (data/euclid_psf/psf_fig_full_mean.npz, 1023^2 at
0.05"/px, from 17,905 Q1 stars) per band, log stretch, as a 10" core and a 40" view with the spikes; plus
the VIS ePSF of 9 of the 44 sky regions, to show how alike they are."""
import numpy as np
import matplotlib
from matplotlib import cm

from style import save_panel

d = np.load("/Users/alarion239/Desktop/EuclidPolish/data/euclid_psf/psf_fig_full_mean.npz")
st = np.load("/Users/alarion239/Desktop/EuclidPolish/data/euclid_psf/psf_fig_stack.npz")
cmap = matplotlib.colormaps["magma"]


def logimg(p, half, floor=1e-6):
    c = p.shape[0] // 2
    a = p[c - half:c + half, c - half:c + half] / p.max()
    v = (np.log10(np.clip(a, floor, 1)) - np.log10(floor)) / -np.log10(floor)
    return cmap(v)[..., :3]


names = {"VIS": "VIS", "Y": "NISP Y", "J": "NISP J", "H": "NISP H"}
for b, lab in names.items():
    save_panel(logimg(d[b], 100), f"PSF_{b}_core10.png", label=f"{lab} PSF", bar_arcsec=1.0, interp="lanczos")
    save_panel(logimg(d[b], 400), f"PSF_{b}_wide40.png", label=f"{lab} PSF", bar_arcsec=5.0, interp="lanczos")
idx = np.argsort(-st["VIS_nstars"])[:9]
for k, i in enumerate(idx):
    save_panel(logimg(st["VIS"][i], 120), f"PSF_VIS_region{k + 1}.png",
               label=f"VIS · region {k + 1}", interp="lanczos")
print("ok", st["VIS_nstars"][idx])
