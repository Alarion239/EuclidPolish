"""Hard-stretch figure of faint galaxy outskirts (no inference: reads sr_cache).

    python figure.py real VIS tiles(comma) OUT.png
    python figure.py syn VIS fields(comma) OUT.png
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "sr_cache")
STATS = os.path.join(HERE, "stats")
CACHE = "data/jwst_euclid_overlap/resolution_cache"
META = json.load(open(os.path.join(HERE, "prep.json")))
P = np.load(os.path.join(HERE, "prep.npz"))
ORIGINS = [tuple(o) for o in P["origins"]]
SPECS = ["universal_member_238", "milestone_member_257_15000000", "universal_member_227"]
TITLES = {"universal_member_238": "238 (production, 1.9M)", "milestone_member_257_15000000": "257 @ 15M",
          "universal_member_227": "227 (15.5M, no isotropy rule)"}
CROP = int(os.environ.get("CROP", 192))
KNEE = float(os.environ.get("KNEE", 0.2))          # x the per-SR-pixel noise
BIDX = {"VIS": 0, "H": 3}


def lr_up(band_img):
    return zoom(np.nan_to_num(band_img.astype(np.float64)), 2, order=3, grid_mode=True, mode="grid-mirror") / 4.0


def crop_origin(lr_vis, sig_s):
    """192-px crop (origin on the 32-px grid, inside the 32-px border) holding the
    most LR VIS light in [1, 30] sigma_s (the galaxy and its faint outskirts)."""
    S = ndimage.gaussian_filter(lr_vis.astype(np.float64), 2.0) / sig_s
    faint = ((S >= 1) & (S <= 30)).astype(float)
    best = None
    for y in range(32, 510 - 32 - CROP + 1, 32):
        for x in range(32, 510 - 32 - CROP + 1, 32):
            sc = faint[y // 2:(y + CROP) // 2, x // 2:(x + CROP) // 2].sum()
            if best is None or sc > best[0]:
                best = (sc, y, x)
    return best[1], best[2]


_CAT = {}


def galaxy_crop(tid):
    """Crop origin (SR px, on the 32-px grid) centred on the tile's brightest MER
    galaxy (point-like prob < 0.1) whose centre lies >= 48 SR px inside the tile."""
    if not _CAT:
        from astropy.table import Table
        from astropy.wcs import WCS
        from euclid_polish.web.helpers import real_tiles as rt
        cat = Table.read("data/jwst_euclid_overlap/flux_conservation/mer_nexus.fits")
        _CAT["cat"] = cat
        _CAT["entries"] = {e.id: e for e in rt.list_entries("nexus")}
        _CAT["WCS"], _CAT["rt"] = WCS, rt
    cat, WCS, rt = _CAT["cat"], _CAT["WCS"], _CAT["rt"]
    w = WCS(rt.lr_header(_CAT["entries"][tid])).celestial
    x, y = w.world_to_pixel_values(np.asarray(cat["right_ascension"], float), np.asarray(cat["declination"], float))
    flux = np.asarray(cat["flux_detection_total"], float)
    ok = ((np.asarray(cat["point_like_prob"], float) < 0.1) & (x > 24) & (x < 230) & (y > 24) & (y < 230)
          & np.isfinite(flux))
    i = np.flatnonzero(ok)[np.argmax(flux[ok])]
    cy, cx = 2 * y[i] + 0.5, 2 * x[i] + 0.5
    snap = lambda c: int(np.clip(32 * round((c - CROP / 2) / 32), 0, 510 - CROP))  # noqa: E731
    return snap(cy), snap(cx)


def window_z(slug, dom, key_index, y0, x0, band):
    s = np.load(os.path.join(STATS, f"{slug}__{dom}.npz"))
    z = s["z_e"][key_index, :, BIDX[band]]
    sel = [i for i, (y, x) in enumerate(ORIGINS) if y0 <= y <= y0 + CROP - 64 and x0 <= x <= x0 + CROP - 64]
    return z[sel]


def main():
    dom, band, keys, out_png = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
    b = BIDX[band]
    sig_sr = np.asarray(META["sigma_lr_real" if dom == "real" else "sigma_lr_syn"])[b] / 4.0
    knee = KNEE * sig_sr
    sig_s = META["sigma_smooth_vis_real" if dom == "real" else "sigma_smooth_vis_syn"]
    rows = []
    for key in keys:
        if dom == "real":
            with np.load(os.path.join(CACHE, f"{key}.npz")) as d:
                lr = d["lr"]
            ui = META["tiles"].index(key)
            ckey = key
            truth = None
        else:
            ui = META["fields"].index(int(key))
            ckey = f"syn{int(key):03d}"
            lr = None
            truth = np.load(os.path.join(SRC, f"truth__{ckey}.npy"))[..., 0 if band == "VIS" else 1]
        if lr is None:
            lrv = None
        srs = {s: np.load(os.path.join(SRC, f"{s}__{ckey}.npy"))[..., 0 if band == "VIS" else 1] for s in SPECS}
        inp = (lr_up(lr[..., b]) if lr is not None
               else np.load(os.path.join(SRC, f"lr__{ckey}.npy"))[..., 0 if band == "VIS" else 1])
        lrv = lr[..., 0] if lr is not None else None
        if lrv is None:                                      # synthetic: the LR VIS from the cubic copy
            lrv = np.load(os.path.join(SRC, f"lr__{ckey}.npy"))[..., 0].reshape(255, 2, 255, 2).sum((1, 3))
        if dom == "real" and os.environ.get("CROPMODE", "galaxy") == "galaxy":
            y0, x0 = galaxy_crop(key)
        elif dom == "syn":                                   # the truth's faint light, same thresholds
            tv = np.load(os.path.join(SRC, f"truth__{ckey}.npy"))[..., 0]
            tv = tv.reshape(255, 2, 255, 2).sum((1, 3))          # on the LR grid, e- per LR px
            y0, x0 = crop_origin(tv + 0.0, sig_s)
        else:
            y0, x0 = crop_origin(lrv, sig_s)
        cut = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
        panels = ([("truth (target PSF)", truth[cut], None)] if truth is not None else [])
        panels.append(("Euclid input (cubic x2)", inp[cut], None))
        for s in SPECS:
            panels.append((TITLES[s], srs[s][cut], window_z(s, dom, ui, y0, x0, band)))
        rows.append((key, y0, x0, panels))
    ncol = len(rows[0][3])
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.9 * ncol, 2.62 * len(rows) + 0.75), squeeze=False)
    for r, (key, y0, x0, panels) in enumerate(rows):
        ref = np.arcsinh(panels[-3][1] / knee)                 # the 238 panel sets the top
        top = float(np.percentile(ref, 99.0))
        if os.environ.get("TOP_SIGMA"):                        # saturate above TOP_SIGMA x the noise
            top = float(np.arcsinh(float(os.environ["TOP_SIGMA"]) * sig_sr / knee))
        for c, (title, img, z) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(np.arcsinh(img / knee), cmap="gray", vmin=-1.0, vmax=top, origin="lower",
                      interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{key}\n({y0},{x0})", fontsize=8)
            if z is not None:
                ax.text(0.02, 0.02, f"median z {np.median(z):.1f}, z≥3 in {int((z >= 3).sum())}/{len(z)}",
                        transform=ax.transAxes, fontsize=7, color="yellow", va="bottom",
                        bbox=dict(facecolor="black", alpha=0.5, pad=1, lw=0))
    what = "real NEXUS tiles" if dom == "real" else "synthetic test fields"
    topdesc = (f"white at {os.environ['TOP_SIGMA']} × the noise" if os.environ.get("TOP_SIGMA")
               else "white at the 99th percentile of 238")
    fig.suptitle(f"{band}, {what}, {CROP * 0.05:.1f}″ crops: asinh(x / k), k = {KNEE:g} × per-SR-pixel noise "
                 f"σ ({knee:.3g} e⁻), black at asinh = −1, {topdesc}.\n"
                 f"Labels: AxisExcess z of the crop's 64-px windows (stride 32), {band}, electrons "
                 f"(isotropic: median 0, z≥3 in ~0.4%).", fontsize=9, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / (2.62 * len(rows) + 0.75)))
    fig.savefig(out_png, dpi=int(os.environ.get("DPI", 110)))
    print("wrote", out_png)


if __name__ == "__main__":
    main()
