#!/usr/bin/env python
"""Paper collages of the training data, in the style of the LR/POLISHed/JWST
triples (output/jwst_triples_257): solar-referenced colour, one Lupton asinh
stretch shared inside a figure, white titles, band key, scale bar.

* ``euclid_vs_synthetic``: real Euclid Q1 MER fields (archive cutouts) above
  synthetic LR fields from the v6 test records, all 25.6" / 25.5" wide and on
  one stretch, so the reader can compare populations, colours and noise.
* ``hr_lr_pairs``: synthetic HR truth (0.05", target PSF) next to its
  forward-modelled Euclid LR (0.1", Euclid PSFs + Q1 noise), crops centred on
  large galaxies; one stretch per pair, LR shown with its native pixels.

    python paper_figures/generation_collages/build.py sheets   # contact sheets for picking
    python paper_figures/generation_collages/build.py final    # the paper figures (PNG + PDF)
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.stats import sigma_clipped_stats  # noqa: E402

from euclid_polish.visualization import color  # noqa: E402
from euclid_polish.web.helpers import real_tiles, sky_records  # noqa: E402

OUT = Path(__file__).resolve().parent
RECORDS = ROOT / "data" / "records_v6_test"
BANDS = ("VIS", "Y_E", "J_E", "H_E")
#: Euclid white balance of the triples figures (output/jwst_triples_257/code/
#: balance.json "sr"): makes the median galaxy of the NEXUS field neutral.
BALANCE = np.array([0.25984329, 0.82713020, 1.24706697, 1.65685165])
KEY_COLOURS = ("#7fb2ff", "#8fe08f", "#ff8a7a")
STROKE = [pe.withStroke(linewidth=1.6, foreground="black")]
FIG_W = 7.13           # AASTeX full width (in)


# ----------------------------------------------------------------- data
def sky_subtract(cube: np.ndarray) -> np.ndarray:
    out = cube.astype(np.float64).copy()
    for k in range(cube.shape[-1]):
        plane = out[..., k]
        _m, med, _s = sigma_clipped_stats(plane[np.isfinite(plane)], sigma=2.5, maxiters=8)
        out[..., k] = plane - med
    return out


def calibrated(cube: np.ndarray) -> np.ndarray:
    """Sky-subtracted, solar-referenced, white-balanced 4-band cube."""
    cal = color.calibrate(np.nan_to_num(sky_subtract(cube)), band_names=BANDS, reference="solar")
    return cal / BALANCE


def real_field(identifier: str) -> np.ndarray:
    entry = real_tiles.get_entry("archive", identifier)
    return real_tiles.get_tile("archive", identifier, entry=entry).lr_e


def syn(kind: str, index: int) -> np.ndarray:
    return np.asarray(sky_records.read_record(str(RECORDS / f"{kind}_test.tfrecord"), index).data,
                      np.float64)


# ----------------------------------------------------------------- colour
def lupton(cal: np.ndarray, *, white: float, q: float) -> np.ndarray:
    """R = H, G = (Y+J)/2, B = VIS; Lupton+04 asinh (hue kept), knee white/q,
    x1.25 chroma: the triples' mapping."""
    r, g, b = cal[..., 3], 0.5 * (cal[..., 1] + cal[..., 2]), cal[..., 0]
    i = (r + g + b) / 3.0
    soft = white / q
    f = np.arcsinh(np.maximum(i, 0) / soft) / np.arcsinh(white / soft)
    scale = np.where(i > 0, f / np.maximum(i, 1e-30), 0.0)
    rgb = np.clip(np.stack([r, g, b], -1) * scale[..., None], 0, None)
    peak = rgb.max(-1, keepdims=True)
    rgb = np.where(peak > 1, rgb / np.maximum(peak, 1e-30), rgb)
    lum = rgb.mean(-1, keepdims=True)
    return np.clip(lum + 1.25 * (rgb - lum), 0, 1)


def intensity(cal: np.ndarray) -> np.ndarray:
    return (cal[..., 3] + 0.5 * (cal[..., 1] + cal[..., 2]) + cal[..., 0]) / 3.0


# ----------------------------------------------------------------- drawing
def panel(ax, img, *, title=None, key=True, bar_arcsec=None, pix=0.1, fs=8.0, interp="nearest",
          corner=None):
    ax.imshow(img, origin="lower", interpolation=interp)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.text(0.04, 0.955, title, transform=ax.transAxes, color="white", fontsize=fs,
                weight="bold", va="top", path_effects=STROKE)
    if key:
        x = 0.04
        for name, colr in zip(("VIS", "Y+J", "H"), KEY_COLOURS):
            t = ax.text(x, 0.04, name, transform=ax.transAxes, color=colr, fontsize=fs * 0.72,
                        va="bottom", path_effects=STROKE)
            ax.figure.canvas.draw()
            bb = t.get_window_extent().transformed(ax.transAxes.inverted())
            x = bb.x1 + 0.03
    if bar_arcsec:
        n = img.shape[0]
        length = bar_arcsec / pix
        x1 = n * 0.955; x0 = x1 - length; y = n * 0.065
        ax.plot([x0, x1], [y, y], color="white", lw=1.5, solid_capstyle="butt",
                path_effects=[pe.withStroke(linewidth=3, foreground="black")])
        ax.text((x0 + x1) / 2, y + n * 0.025, f"{bar_arcsec:g}″", color="white", ha="center",
                va="bottom", fontsize=fs * 0.75, path_effects=STROKE)
    if corner:
        ax.text(0.96, 0.955, corner, transform=ax.transAxes, color="0.85", fontsize=fs * 0.68,
                ha="right", va="top", path_effects=STROKE)


# ----------------------------------------------------------------- picking sheets
def contact(items, path, ncol=8, size=1.6):
    nrow = (len(items) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * size, nrow * size * 1.08))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (label, img) in zip(np.ravel(axes), items):
        ax.imshow(img, origin="lower", interpolation="nearest")
        ax.set_title(label, fontsize=6)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def big_galaxies(n=60):
    d = pd.read_csv(RECORDS / "sources_test.csv")
    g = d[(d["type"] == "galaxy") & (d["off_field"] == 0)
          & d.x_pix.between(70, 440) & d.y_pix.between(70, 440)].copy()
    g["size"] = g["re_arcsec"]
    g = g[g.flux_vis_e > 3e4].sort_values("size", ascending=False)
    return g.head(n)


def sheets():
    rng = np.random.default_rng(2026)
    ids = [e.id for e in real_tiles.list_entries("archive")]
    pick = sorted(rng.choice(len(ids), 48, replace=False))
    cals = [(f"real {ids[i]}", calibrated(real_field(ids[i]))) for i in pick]
    syns = [(f"syn {i}", calibrated(syn("dirty", i))) for i in range(48)]
    white = np.percentile(np.concatenate([intensity(c).ravel() for _l, c in cals]), 99.7)
    meta = {"white": float(white)}
    for q in (30, 100):
        contact([(l, lupton(c, white=white, q=q)) for l, c in cals], OUT / f"sheet_real_q{q}.png")
        contact([(l, lupton(c, white=white, q=q)) for l, c in syns], OUT / f"sheet_syn_q{q}.png")
    gal = big_galaxies()
    items = []
    for _i, row in gal.iterrows():
        f, cx, cy = int(row.field_index), int(row.x_pix), int(row.y_pix)
        hr = calibrated(syn("hr", f))
        hw = 64
        c = hr[max(cy - hw, 0):cy + hw, max(cx - hw, 0):cx + hw]
        if c.shape[:2] != (2 * hw, 2 * hw):
            continue
        w = np.percentile(intensity(c), 99.8)
        items.append((f"f{f} ({cx},{cy}) re{row['size']:.2f}", lupton(c, white=w, q=30)))
    contact(items, OUT / "sheet_hr_galaxies.png")
    (OUT / "sheet_meta.json").write_text(json.dumps(meta, indent=1))
    print("sheets written to", OUT)


# ----------------------------------------------------------------- final figures
def final():
    sel = json.loads((OUT / "selection.json").read_text())
    # --- Euclid vs synthetic: 2 rows x 4 full fields, one stretch for all
    reals = [calibrated(real_field(i)) for i in sel["real"]]
    syns = [calibrated(syn("dirty", i)) for i in sel["synthetic"]]
    white = np.percentile(np.concatenate([intensity(c).ravel() for c in reals]), sel["white_pct"])
    ncol, gap = 4, 0.006
    w = (1 - (ncol - 1) * gap) / ncol
    h = w                                     # square panels; figure aspect follows
    fig = plt.figure(figsize=(FIG_W, FIG_W * (2 * h + gap)))
    hfrac = h / (2 * h + gap); gfrac = gap / (2 * h + gap)
    for row, (cubes, title) in enumerate(((reals, "Euclid Q1"), (syns, "Synthetic"))):
        for k, cal in enumerate(cubes):
            ax = fig.add_axes([k * (w + gap), (1 - row) * (hfrac + gfrac), w, hfrac])
            n = cal.shape[0]
            cal = cal[(n - 255) // 2:(n - 255) // 2 + 255, (n - 255) // 2:(n - 255) // 2 + 255]
            panel(ax, lupton(cal, white=white, q=sel["q_fields"]), title=title,
                  key=(k == 0), bar_arcsec=5.0 if k == ncol - 1 else None, pix=0.1)
    fig.savefig(OUT / "euclid_vs_synthetic.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "euclid_vs_synthetic.pdf", facecolor="white")
    plt.close(fig)

    # --- HR / LR pairs: rows of two pairs (HR | LR), one stretch per pair
    pairs = sel["pairs"]
    npair_row = 2
    nrow = (len(pairs) + npair_row - 1) // npair_row
    ncol = 2 * npair_row
    gap, pgap = 0.004, 0.018                  # inside a pair / between pairs
    w = (1 - (npair_row - 1) * pgap - npair_row * gap) / ncol
    fig_h = FIG_W * (nrow * w + (nrow - 1) * pgap)
    fig = plt.figure(figsize=(FIG_W, fig_h))
    hf = w * FIG_W / fig_h; vgap = pgap * FIG_W / fig_h
    for p, rec in enumerate(pairs):
        r, c = divmod(p, npair_row)
        hr = calibrated(syn("hr", rec["field"]))
        lr = calibrated(syn("dirty", rec["field"]))
        hw = rec["hw"]
        cx, cy = rec["cx"], rec["cy"]
        hc = hr[cy - hw:cy + hw, cx - hw:cx + hw]
        lc = lr[cy // 2 - hw // 2:cy // 2 + hw // 2, cx // 2 - hw // 2:cx // 2 + hw // 2]
        lc = np.repeat(np.repeat(lc, 2, 0), 2, 1) / 4.0      # native 0.1" pixels on the HR grid
        white = rec.get("white_scale", 1.0) * np.percentile(intensity(hc), 99.8)
        x0 = c * (2 * w + gap + pgap)
        y0 = 1 - (r + 1) * hf - r * vgap
        for k, (img, title, interp) in enumerate(((hc, "HR truth", "lanczos"),
                                                  (lc, "Simulated Euclid", "nearest"))):
            ax = fig.add_axes([x0 + k * (w + gap), y0, w, hf])
            panel(ax, lupton(img, white=white, q=rec.get("q", 30.0)), title=title,
                  key=(k == 1), bar_arcsec=(rec.get("bar", 1.0) if k == 0 else None), pix=0.05,
                  interp=interp)
    fig.savefig(OUT / "hr_lr_pairs.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "hr_lr_pairs.pdf", facecolor="white")
    plt.close(fig)
    print("final figures written to", OUT)


if __name__ == "__main__":
    {"sheets": sheets, "final": final}[sys.argv[1]]()
