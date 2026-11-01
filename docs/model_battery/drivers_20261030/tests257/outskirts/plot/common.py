"""Shared helpers for the weekly-review deck figures."""
import os
import numpy as np
from astropy.io import fits

REPO = "/Users/alarion239/Desktop/EuclidPolish"
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

POSTER = f"{REPO}/data/euclid_inference/experiments/outputs/poster/target_181255"
NEXUS = f"{REPO}/data/euclid_inference/experiments/outputs/nexus"
CACHE = f"{REPO}/data/jwst_euclid_overlap/resolution_cache"
JW_BANDS = {0: "VIS-wide_ep1_01_f090w", 1: "Y_E-wide_ep1_01_f115w", 2: "J_E-wide_ep1_01_f150w"}

# vivid console colour: per-band calibration factors
FAC = np.array([1.0, 0.2088, 0.2929, 0.3799])

# deck palette (dark surface)
BG = "#0f1319"
INK = "#e9e6df"
INK2 = "#a9adb5"
GRID = "#262c35"
C = dict(lr="#8a8f98", m249="#3987e5", prod="#d95926", m218="#9085e9",
         m227="#d55181", m237="#199e70", jwst="#c98500", m257="#2fd1c4")


def rgb(cube, knee, white=30.0, sat=1.15):
    """cube: (4, H, W) electrons per SR px (VIS, Y, J, H) -> (H, W, 3) floats."""
    x = cube / FAC[:, None, None]
    r = x[3]
    g = 0.5 * (x[1] + x[2])
    b = x[0]
    out = []
    for ch in (r, g, b):
        v = np.arcsinh(np.clip(ch, -knee, None) / knee) / np.arcsinh(white)
        out.append(v)
    im = np.stack(out, -1)
    # gentle saturation boost around the luminance
    lum = im.mean(-1, keepdims=True)
    im = lum + sat * (im - lum)
    return np.clip(im, 0, 1)


def lr_to_sr(lr):
    """(4, h, w) per-LR-px electrons -> (4, 2h, 2w) per-SR-px (nearest, flux /4)."""
    return np.repeat(np.repeat(lr, 2, axis=1), 2, axis=2) / 4.0


def read_sr(path):
    return fits.getdata(path).astype(np.float32)


def read_poster_lr():
    with fits.open(f"{REPO}/poster/target_181255_lr.fits") as h:
        return np.stack([h[i].data.astype(np.float32) for i in range(1, 5)])


def nexus_lr_jwst(tile):
    z = np.load(f"{CACHE}/{tile}.npz")
    lr = np.moveaxis(z["lr"], -1, 0).astype(np.float32)
    jw = np.zeros((4,) + z["jwst"].shape, np.float32)
    jw[3] = z["jwst"]
    have = [3]
    for i, d in JW_BANDS.items():
        p = f"{CACHE}/{d}/{tile}.npz"
        if os.path.exists(p):
            jw[i] = np.load(p)["jwst"]
            have.append(i)
    return lr, jw, sorted(have)


def save_img(arr, name):
    from PIL import Image
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(os.path.join(OUT, name))
    return os.path.join(OUT, name)


def mpl_dark():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 1.0, "axes.spines.top": False,
        "axes.spines.right": False, "font.family": "DejaVu Sans", "font.size": 15,
        "axes.titlesize": 17, "axes.titlecolor": INK, "axes.titleweight": "bold",
        "legend.frameon": False, "legend.fontsize": 14, "lines.linewidth": 2.4,
        "lines.solid_capstyle": "round", "axes.titlepad": 12,
    })
