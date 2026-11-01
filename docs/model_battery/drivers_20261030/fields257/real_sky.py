"""Real-sky look at faint galaxy outskirts: Euclid input, models, JWST on NEXUS tiles.

    python blur_visual.py OUT.png spec1,spec2,... [--n 4]

Picks the tiles whose JWST image has the most extended faint light (pixels
between 1 and 30 x the Euclid per-SR-pixel noise, star-free), crops 192 SR px
around the biggest galaxy, and shows VIS (vs F090W) and H (vs F200W) in an
asinh stretch with the knee at the Euclid noise (faint light visible).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import jwst_resolution as J  # noqa: E402
from euclid_polish.eval import jwst_resolution as jr  # noqa: E402
from euclid_polish.ensemble import EnsembleModel  # noqa: E402
from euclid_polish.web.helpers import jwst_euclid, milestones, model_catalog  # noqa: E402

CACHE = "data/jwst_euclid_overlap/resolution_cache"
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr_cache")
RGB = ("H_E", "J_E", "VIS")
CROP = 192
FIG = float(os.environ.get("FIG", 2.6))
_MODELS = {}


def raw_band(entries, band):
    if band == "H_E":
        raw = J._tile_jwst(CACHE, entries, band)
    else:
        mosaic = os.path.join(str(jwst_euclid.overlap_root()),
                              J.EDR_MOSAIC.format(filter=jr.JWST_FILTER[band].lower()))
        raw = J._mosaic_jwst(CACHE, entries, band, mosaic)
    raw = {t: v for t, v in raw.items() if np.isfinite(v[1]).any()}
    shifts = np.array([s for _l, _j, s in raw.values() if s is not None])
    shift = tuple(float(v) for v in np.median(shifts, axis=0))
    b = J.BANDS.index(band)
    tiles = {t: (lr, jr.register(j, shift)) for t, (lr, j, _s) in raw.items()}
    sigma_e = J._noise_sigma([lr[..., b] for lr, _j in tiles.values()]) / 4.0
    return tiles, sigma_e


def sr(spec, tid, lr):
    os.makedirs(SRC, exist_ok=True)
    path = os.path.join(SRC, f"{spec.replace(':', '_').replace('@', '_')}__{tid}.npy")
    if os.path.isfile(path):
        return np.load(path)
    if spec not in _MODELS and spec.startswith("milestone:"):
        _MODELS[spec] = milestones.load(*milestones.parse_spec(spec))
    elif spec not in _MODELS:
        item = model_catalog.resolve_spec(spec)
        _MODELS[spec] = EnsembleModel(model_catalog.ensemble_dir(), labels=[item.reads[0]],
                                      include_universal=True, starless=False)
    out = np.asarray(_MODELS[spec].member_arrays(np.ascontiguousarray(lr, np.float32))[0], np.float32)
    np.save(path, out)
    return out


def rgb(chans, sigs, tops):
    out = np.stack([np.clip((np.arcsinh(c / s) + 0.3) / (t + 0.3), 0, 1) for c, s, t in zip(chans, sigs, tops)], -1)
    return np.nan_to_num(out)


def main():
    out_png, specs = sys.argv[1], [s for s in sys.argv[2].split(",") if s]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 6
    crop = int(os.environ.get("CROP", CROP))
    entries = J._entries(None)
    by_band = {band: raw_band(entries, band) for band in RGB}
    tiles_h, sig_h = by_band["H_E"]
    picks = []
    for tid, (lr, jw) in tiles_h.items():
        if not all(tid in by_band[b][0] for b in RGB):
            continue
        jv = by_band["VIS"][0][tid][1]
        jj = by_band["J_E"][0][tid][1]
        sm = ndimage.gaussian_filter(np.nan_to_num(jw), 2)
        faint = (sm > 2 * sig_h) & (sm < 50 * sig_h)
        h, w = jw.shape
        best = None
        for y0 in range(0, h - crop + 1, 32):
            for x0 in range(0, w - crop + 1, 32):
                cut = (slice(y0, y0 + crop), slice(x0, x0 + crop))
                lc = lr[y0 // 2:(y0 + crop) // 2, x0 // 2:(x0 + crop) // 2]
                if (not all(np.isfinite(a[cut]).all() for a in (jw, jv, jj)) or np.nanmax(jw[cut]) > 1500
                        or not np.isfinite(lc).all() or (lc == 0).mean() > 0.001):
                    continue
                score = int(faint[cut].sum())
                if best is None or score > best[0]:
                    best = (score, y0 + crop // 2, x0 + crop // 2)
        if best:
            picks.append((best[0], tid, best[1], best[2]))
    picks = sorted(picks, reverse=True)
    if "--tiles" in sys.argv:
        want = sys.argv[sys.argv.index("--tiles") + 1].split(",")
        picks = sorted((p for p in picks if p[1] in want), key=lambda p: want.index(p[1]))
    picks = picks[:n]
    srs = {spec: {tid: sr(spec, tid, tiles_h[tid][0]) for _s, tid, _y, _x in picks} for spec in specs}
    names = {"universal:member_238": "production 238", "milestone:member_257@15000000": "257 @ 15M"}
    cols = ["Euclid input", *[names.get(s, s) for s in specs], "JWST (F200W/F150W/F090W)"]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(FIG * len(cols), FIG * len(picks) + 0.4), squeeze=False)
    for r, (_s, tid, cy, cx) in enumerate(picks):
        h, w = tiles_h[tid][1].shape
        y0 = int(np.clip(cy - crop // 2, 0, h - crop))
        x0 = int(np.clip(cx - crop // 2, 0, w - crop))
        cut = (slice(y0, y0 + crop), slice(x0, x0 + crop))
        sigs = [by_band[b][1] for b in RGB]
        jws = [by_band[b][0][tid][1][cut] for b in RGB]
        tops = [np.nanpercentile(np.arcsinh(j / s), 99.7) for j, s in zip(jws, sigs)]
        panels = [[J._runner("lr", b)(by_band[b][0][tid][0])[cut] for b in RGB]]
        panels += [[srs[s][tid][..., J.BANDS.index(b)][cut] for b in RGB] for s in specs]
        panels += [jws]
        for c, chans in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(rgb(chans, sigs, tops), origin="lower", interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(tid, fontsize=8)
    fig.suptitle("257 @ 15M on real sky (NEXUS tiles, 9.6\u2033 crops), next to JWST\nRGB = H / J / VIS, asinh knee at the Euclid noise per SR pixel", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print("wrote", out_png, [p[1] for p in picks])


if __name__ == "__main__":
    main()
