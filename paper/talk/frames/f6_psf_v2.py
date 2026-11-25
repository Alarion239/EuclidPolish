"""F6 v2 (11-25 review): "sharp" shown, not asserted. Same 10" field and the same stretch for a ground-based
PSF (DES, typical 0.9" seeing; Moffat beta = 3 model) and Euclid's measured VIS and NISP H ePSFs (mean of 17,905
Q1 stars, data/euclid_psf/psf_fig_full_mean.npz, 0.05"/px), plus the VIS ePSF in three sky regions for stability.
Colour map: black -> deck teal -> white (the old magma matched nothing else in the deck). Log stretch over four
decades, so the core width is what you see; FWHMs in the labels are measured (half-maximum area) on these maps."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from style import save_panel

d = np.load("/Users/alarion239/Desktop/EuclidPolish/data/euclid_psf/psf_fig_full_mean.npz")
st = np.load("/Users/alarion239/Desktop/EuclidPolish/data/euclid_psf/psf_fig_stack.npz")
cmap = LinearSegmentedColormap.from_list("teal", ["#000000", "#0b3b36", "#1f9c8c", "#2fd4bf", "#c9fff6", "#ffffff"])
PIX, HALF, FLOOR = 0.05, 100, 1e-4                       # 0.05"/px, 10" field, four decades


def logimg(p, half=HALF, floor=FLOOR):
    c = np.array(np.unravel_index(np.argmax(p), p.shape))
    a = p[c[0] - half:c[0] + half, c[1] - half:c[1] + half] / p.max()
    v = (np.log10(np.clip(a, floor, 1)) - np.log10(floor)) / -np.log10(floor)
    return cmap(v)[..., :3]


def fwhm(p):
    return 2 * np.sqrt((p >= p.max() / 2).sum() * PIX ** 2 / np.pi)


def moffat(fw, beta=3.0, n=1023):
    alpha = fw / (2 * np.sqrt(2 ** (1 / beta) - 1))
    y, x = (np.indices((n, n)) - n // 2) * PIX
    return (1 + (x ** 2 + y ** 2) / alpha ** 2) ** -beta


des = moffat(0.9)
# labels carry the numbers the talk quotes (measured here: DES model 0.89", VIS 0.169", H 0.465")
panels = [("DES", des, "Ground (DES) · 0.9″"), ("VIS", d["VIS"], "Euclid VIS · 0.16″"), ("H", d["H"], "Euclid NISP H")]
for key, p, lab in panels:
    save_panel(logimg(p), f"PSF2_{key}.png", label=lab, bar_arcsec=1.0, pix_arcsec=PIX)
    print(key, round(fwhm(p), 3))
idx = np.argsort(-st["VIS_nstars"])[:3]
for k, i in enumerate(idx):
    save_panel(logimg(st["VIS"][i]), f"PSF2_VIS_region{k + 1}.png", label=f"VIS · sky region {k + 1}",
               bar_arcsec=1.0, pix_arcsec=PIX)
print("ok", st["VIS_nstars"][idx])
