"""Poster galaxy cores, input | 238 | 257@15M, from the SRs saved by holes_run.py.

Two figures with the same panels (rows = band x nucleus crop):
* plain: an asinh stretch shared by the three panels of a row (knee = the crop's
  median LR flux per SR pixel, white = the input's 99.9th percentile), so a hole —
  SR below half the LR flux per SR pixel — shows as a dark patch in the bright core;
* overlay: the same with the real_metrics hole pixels painted red (input panel: the
  bright-1 % LR pixels that are judged, outlined cyan).
Hole definition exactly as euclid_polish/web/helpers/real_metrics._holes (whole-tile
99th percentile of the LR band, LR > 0, SR < 0.5 * LR / 4).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation

BANDS = ("VIS", "Y_E", "J_E", "H_E")
STEM = "poster-target_181255"
MODELS = (("Universal 238 (production)", "production", "production"),
          ("257 @ 15M", "milestone-member_257-15000000", "milestone:member_257@15000000"))
#: The two nuclei (LR px, y, x): the brighter southern one and the central spiral's.
CROPS = {"A": (168, 279), "B": (512, 528)}


def hole_maps(lr: np.ndarray, sr: np.ndarray, f: int = 2):
    finite = np.isfinite(lr)
    thr = np.percentile(lr[finite], 99.0)
    bright = finite & (lr >= thr) & (lr > 0)
    block = np.ones((f, f))
    mask = np.kron(bright, block) > 0
    per = np.kron(np.where(bright, lr, 0.0), block) / float(f * f)
    holes = mask & ~(sr >= 0.5 * per)
    return mask, holes


def render(lr, srs, tile_holes, bands, half, overlay, out):
    rows = [(band, name, centre) for band in bands for name, centre in CROPS.items()]
    fig, axes = plt.subplots(len(rows), 3, figsize=(3 * 4.4, len(rows) * 4.6), squeeze=False)
    maps = {}
    for band in bands:
        b = BANDS.index(band)
        for _n, slug, _s in MODELS:
            maps[band, slug] = hole_maps(lr[..., b].astype(np.float64), srs[slug][..., b].astype(np.float64))
    for r, (band, cname, (cy, cx)) in enumerate(rows):
        b = BANDS.index(band)
        y0, y1, x0, x1 = cy - half, cy + half, cx - half, cx + half
        sl = (slice(2 * y0, 2 * y1), slice(2 * x0, 2 * x1))
        ref = np.kron(np.nan_to_num(lr[y0:y1, x0:x1, b].astype(np.float64)), np.ones((2, 2))) / 4.0
        k = max(float(np.median(ref)), 1e-3)
        vmax = np.arcsinh(float(np.percentile(ref, 99.9)) / k)
        mask_in = maps[band, MODELS[0][1]][0][sl]
        panels = [("Input (LR/4 per SR px)", ref, None)]
        panels += [(name, srs[slug][..., b][sl].astype(np.float64), maps[band, slug][1][sl])
                   for name, slug, _s in MODELS]
        for c, (name, img, holes) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(np.arcsinh(img / k), origin="lower", cmap="gray", vmin=0.0, vmax=vmax,
                      interpolation="nearest")
            title = f"{band} · nucleus {cname} · {name}"
            if overlay:
                rgba = np.zeros(img.shape + (4,))
                if holes is None:
                    rgba[binary_dilation(mask_in) & ~mask_in] = (0.2, 0.9, 1.0, 0.9)
                else:
                    rgba[holes] = (1.0, 0.1, 0.1, 0.85)
                ax.imshow(rgba, origin="lower", interpolation="nearest")
            if holes is not None:
                slug = MODELS[c - 1][1]
                crop = 100.0 * holes.sum() / max(int(mask_in.sum()), 1)
                whole = tile_holes.get(slug, {}).get(band)
                title += f"\nhole px {int(holes.sum())} = {crop:.1f}% of judged px here"
                if whole is not None:
                    title += f" · tile {whole:.2f}%"
            else:
                title += f"\njudged (bright-1 %) px here: {int(mask_in.sum())}"
            ax.set_title(title, fontsize=8.5)
            ax.set_xticks([])
            ax.set_yticks([])
    what = "red = hole: SR < ½·LR/4 in the tile's bright-1 % LR pixels; cyan = judged pixels" if overlay \
        else "dark patches inside the bright core = holes (SR < ½·LR/4)"
    fig.suptitle(f"poster/target_181255: nucleus A (LR y,x {CROPS['A']}) and B {CROPS['B']}, "
                 f"{2 * half / 10:.1f}″ boxes, 0.05″ SR px, north up; asinh stretch per row "
                 f"(knee = crop median, white = input p99.9)\n{what}; black blobs in the NISP input = "
                 f"zero-valued (saturated) LR pixels, which the hole metric does not judge", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sr-dir", required=True)
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--half", type=int, default=64, help="half side of each crop, LR px")
    p.add_argument("--bands", default="VIS,H_E")
    p.add_argument("--metrics", default=None, help="record JSON (holes_run.py) for whole-tile hole %")
    args = p.parse_args()
    lr = np.load(os.path.join(args.sr_dir, f"{STEM}__lr.npy"))
    srs = {slug: np.load(os.path.join(args.sr_dir, f"{STEM}__{slug}.npy")) for _n, slug, _s in MODELS}
    tile_holes = {}
    if args.metrics:
        res = json.load(open(args.metrics))["results"]["poster/target_181255"]
        tile_holes = {slug: {b: v["hole_pct"] for b, v in res[spec]["metrics"]["per_band"].items()}
                      for _n, slug, spec in MODELS}
    bands = [b.strip() for b in args.bands.split(",")]
    render(lr, srs, tile_holes, bands, args.half, False, args.out_prefix + "_plain.png")
    render(lr, srs, tile_holes, bands, args.half, True, args.out_prefix + "_holes_overlay.png")


if __name__ == "__main__":
    main()
