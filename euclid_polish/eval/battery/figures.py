"""Test 8 and the tests' figures, one column per model.

* ``grid_<sample>_<band>.png`` — hard-stretched crops of faint galaxy
  outskirts with each 64-px window's AxisExcess z (2026-10-29 grid test);
* ``poster_cores_plain.png`` / ``poster_cores_holes.png`` — the poster
  galaxy's two nuclei, hole pixels painted red in the second;
* ``kron_vis.png`` — SR/input Kron VIS flux against magnitude and the pull
  against MER (``scripts/plot_flux_conservation.py``'s layout);
* ``outskirt_recovery.png`` — injected light recovered per r / r_Kron ring;
* ``curve_of_growth.png`` — real galaxies' SR/input F(<r) against r (all
  galaxies and VIS > 24; raw, sky-matched and through the real ePSF), the
  median VIS curve of growth and the synthetic twin (SR/truth, SR ⊛ ePSF/input);
* ``resolution_knee_scan.png`` — ½-coherence resolution against JWST at each
  surface brightness of the knee scan;
* ``fields_synthetic_zooms.png`` / ``fields_synthetic_full.png`` — test
  fields picked by the last model's ∫PSNR gain over the baseline, with the truth;
* ``fields_real_sky.png`` — NEXUS crops with the most faint extended JWST
  light, beside JWST.

Only cached SRs are read (the tiles the figures need are kept when the cache is
pruned); RGB panels are H / J / VIS in asinh stretches.
"""
from __future__ import annotations

import csv
import os

import numpy as np
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy import ndimage

from euclid_polish.eval import jwst_resolution as jr
from euclid_polish.eval.battery import grid, growth, holes, outskirts, resolution, synthetic
from euclid_polish.eval.battery.common import (
    BANDS,
    MER_CATALOGUE,
    RESOLUTION_CACHE,
    Battery,
    ModelInfo,
    field_key,
    field_truth,
    nexus_entries,
    nexus_key,
    nexus_lr,
    real_tile_input,
    records_dir,
    target_fwhm,
    test_fields,
    tile_key,
)
from euclid_polish.eval.battery.flux import catalogue_columns, geometry, isolated
from euclid_polish.web.helpers import real_tiles as rt
from scripts import jwst_resolution as script
from scripts import plot_flux_conservation as pfc

RGB = (3, 2, 0)                                  # H, J, VIS channels of a 4-band cube
RGB_BANDS = ("H_E", "J_E", "VIS")
GRID_CROP = 192
GRID_KNEE = 0.2                                  # × the per-SR-pixel noise
GRID_TOP_SIGMA = 4.0
POSTER_CROPS = {"A": (168, 279), "B": (512, 528)}  # the two nuclei (LR y, x)
POSTER_HALF = 64
FIELD_CROP = 160
REAL_CROP = 192
N_REAL = 6


def _figure(width: float, height: float) -> Figure:
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)
    return fig


def _save(fig: Figure, path: str, dpi: int = 110) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return path


def _bare(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def rgb(chans, sigs, tops) -> np.ndarray:
    out = np.stack([np.clip((np.arcsinh(c / s) + 0.3) / (t + 0.3), 0, 1) for c, s, t in zip(chans, sigs, tops,
                                                                                           strict=True)], -1)
    return np.nan_to_num(out)


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #

def _faint_crop(lr_vis: np.ndarray, sig_s: float) -> tuple[int, int]:
    """Crop origin holding the most LR VIS light at 1–30 σ_s (32-px grid, inside the border)."""
    S = ndimage.gaussian_filter(lr_vis.astype(np.float64), 2.0) / sig_s
    faint = ((S >= 1) & (S <= 30)).astype(float)
    best = None
    for y in range(32, 510 - 32 - GRID_CROP + 1, 32):
        for x in range(32, 510 - 32 - GRID_CROP + 1, 32):
            sc = faint[y // 2:(y + GRID_CROP) // 2, x // 2:(x + GRID_CROP) // 2].sum()
            if best is None or sc > best[0]:
                best = (sc, y, x)
    return best[1], best[2]


def _galaxy_crop(tid: str, cat: Table, entries: dict) -> tuple[int, int]:
    """Crop origin centred on the tile's brightest MER galaxy ≥ 48 SR px inside."""
    w = WCS(rt.lr_header(entries[tid])).celestial
    x, y = w.world_to_pixel_values(np.asarray(cat["right_ascension"], float),
                                   np.asarray(cat["declination"], float))
    flux = np.asarray(cat["flux_detection_total"], float)
    ok = ((np.asarray(cat["point_like_prob"], float) < 0.1) & (x > 24) & (x < 230) & (y > 24) & (y < 230)
          & np.isfinite(flux))
    i = np.flatnonzero(ok)[np.argmax(flux[ok])]
    cy, cx = 2 * y[i] + 0.5, 2 * x[i] + 0.5

    def snap(c):
        return int(np.clip(32 * round((c - GRID_CROP / 2) / 32), 0, 510 - GRID_CROP))

    return snap(cy), snap(cx)


def grid_figure(bt: Battery, infos: list[ModelInfo], dom: str, band: str, path: str) -> str | None:
    if not any(bt.store.read(i, "grid") for i in infos):
        return None
    meta, arrays = grid.prep(bt)
    origins = [tuple(o) for o in arrays["origins"]]
    b = {"VIS": 0, "H": 3}[band]
    sig_sr = np.asarray(meta["sigma_lr_real" if dom == "real" else "sigma_lr_syn"])[b] / 4.0
    knee = GRID_KNEE * sig_sr
    sig_s = meta["sigma_smooth_vis_real" if dom == "real" else "sigma_smooth_vis_syn"]
    stats = {info.spec: grid.model_stats(bt, info)[dom] for info in infos if bt.store.read(info, "grid")}
    if not stats:
        return None
    rows = []
    if dom == "real":
        cat = Table.read(MER_CATALOGUE)
        entries = {e.id: e for e in rt.list_entries("nexus")}
        for tid in [t for t in grid.FIGURE_TILES if t in meta["tiles"]]:
            lr = grid.cache_tile(tid)[0]
            y0, x0 = _galaxy_crop(tid, cat, entries)
            srs = {i.spec: bt.cache.get(i, nexus_key(tid), lr)[..., b] for i in infos if i.spec in stats}
            rows.append((tid, meta["tiles"].index(tid), y0, x0, None, grid.lr_up(lr)[..., b], srs))
    else:
        fwhm = target_fwhm()
        want = set(meta["cand_syn"])
        for idx, lr, rec in test_fields(bt.limits.fields):
            if idx not in want:
                continue
            truth = field_truth(rec, fwhm)
            tv = truth[..., 0].reshape(255, 2, 255, 2).sum((1, 3))
            y0, x0 = _faint_crop(tv + 0.0, sig_s)
            srs = {i.spec: bt.cache.get(i, field_key(idx), lr)[..., b] for i in infos if i.spec in stats}
            rows.append((f"field {idx}", meta["fields"].index(int(idx)), y0, x0, truth[..., b],
                         grid.lr_up(lr)[..., b], srs))
    if not rows:
        return None
    ncol = 1 + len(stats) + (1 if dom == "syn" else 0)
    fig = _figure(2.9 * ncol, 2.62 * len(rows) + 0.75)
    axes = fig.subplots(len(rows), ncol, squeeze=False)
    top = float(np.arcsinh(GRID_TOP_SIGMA * sig_sr / knee))
    for r, (name, ui, y0, x0, truth, inp, srs) in enumerate(rows):
        cut = (slice(y0, y0 + GRID_CROP), slice(x0, x0 + GRID_CROP))
        sel = [i for i, (y, x) in enumerate(origins)
               if y0 <= y <= y0 + GRID_CROP - 64 and x0 <= x <= x0 + GRID_CROP - 64]
        panels = ([("truth (target PSF)", truth[cut], None)] if truth is not None else [])
        panels.append(("Euclid input (cubic ×2)", inp[cut], None))
        for info in infos:
            if info.spec in srs:
                panels.append((info.name, srs[info.spec][cut], stats[info.spec]["z_e"][ui, sel, b]))
        for c, (title, img, z) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(np.arcsinh(img / knee), cmap="gray", vmin=-1.0, vmax=top, origin="lower",
                      interpolation="nearest")
            _bare(ax)
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{name}\n({y0},{x0})", fontsize=8)
            if z is not None and len(z):
                ax.text(0.02, 0.02, f"median z {np.median(z):.1f}, z≥3 in {int((z >= 3).sum())}/{len(z)}",
                        transform=ax.transAxes, fontsize=7, color="yellow", va="bottom",
                        bbox={"facecolor": "black", "alpha": 0.5, "pad": 1, "lw": 0})
    what = "real NEXUS tiles" if dom == "real" else "synthetic test fields"
    fig.suptitle(f"{band}, {what}, {GRID_CROP * 0.05:.1f}″ crops: asinh(x / k), k = {GRID_KNEE:g} × "
                 f"per-SR-pixel noise σ ({knee:.3g} e⁻), black at asinh = −1, white at {GRID_TOP_SIGMA:g} × "
                 f"the noise.\nLabels: AxisExcess z of the crop's 64-px windows (stride 32), {band} "
                 "(isotropic: median 0, z≥3 in ~0.4%).", fontsize=9, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / (2.62 * len(rows) + 0.75)))
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# poster cores
# --------------------------------------------------------------------------- #

def hole_maps(lr: np.ndarray, sr: np.ndarray, f: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """``(judged SR pixels, hole SR pixels)`` of one band (``real_metrics._holes``)."""
    finite = np.isfinite(lr)
    thr = np.percentile(lr[finite], 99.0)
    bright = finite & (lr >= thr) & (lr > 0)
    block = np.ones((f, f))
    mask = np.kron(bright, block) > 0
    per = np.kron(np.where(bright, lr, 0.0), block) / float(f * f)
    return mask, mask & ~(sr >= 0.5 * per)


def poster_figures(bt: Battery, infos: list[ModelInfo], prefix: str, bands=("VIS", "H_E")) -> list[str]:
    if not any(bt.store.read(i, holes.STAGE) for i in infos):
        return []
    lr, lr_input = real_tile_input(holes.POSTER)
    srs, tile_holes = {}, {}
    for info in infos:
        res = bt.store.read(info, holes.STAGE)
        if res is None:
            continue
        srs[info.spec] = bt.cache.get(info, tile_key(holes.POSTER), lr_input)
        per_band = res["poster"]["tiles"][holes.POSTER]["per_band"]
        tile_holes[info.spec] = {b: v.get("hole_pct") for b, v in per_band.items()}
    if not srs:
        return []
    named = [i for i in infos if i.spec in srs]
    out = []
    for overlay in (False, True):
        rows = [(band, name, centre) for band in bands for name, centre in POSTER_CROPS.items()]
        ncol = 1 + len(named)
        fig = _figure(4.4 * ncol, 4.6 * len(rows))
        axes = fig.subplots(len(rows), ncol, squeeze=False)
        maps = {(band, i.spec): hole_maps(lr[..., BANDS.index(band)].astype(np.float64),
                                          srs[i.spec][..., BANDS.index(band)].astype(np.float64))
                for band in bands for i in named}
        for r, (band, cname, (cy, cx)) in enumerate(rows):
            b = BANDS.index(band)
            y0, y1, x0, x1 = cy - POSTER_HALF, cy + POSTER_HALF, cx - POSTER_HALF, cx + POSTER_HALF
            sl = (slice(2 * y0, 2 * y1), slice(2 * x0, 2 * x1))
            ref = np.kron(np.nan_to_num(lr[y0:y1, x0:x1, b].astype(np.float64)), np.ones((2, 2))) / 4.0
            k = max(float(np.median(ref)), 1e-3)
            vmax = np.arcsinh(float(np.percentile(ref, 99.9)) / k)
            mask_in = maps[band, named[0].spec][0][sl]
            panels = [("Input (LR/4 per SR px)", ref, None, None)]
            panels += [(i.name, srs[i.spec][..., b][sl].astype(np.float64), maps[band, i.spec][1][sl], i.spec)
                       for i in named]
            for c, (name, img, hole, spec) in enumerate(panels):
                ax = axes[r][c]
                ax.imshow(np.arcsinh(img / k), origin="lower", cmap="gray", vmin=0.0, vmax=vmax,
                          interpolation="nearest")
                title = f"{band} · nucleus {cname} · {name}"
                if overlay:
                    rgba = np.zeros(img.shape + (4,))
                    if hole is None:
                        rgba[ndimage.binary_dilation(mask_in) & ~mask_in] = (0.2, 0.9, 1.0, 0.9)
                    else:
                        rgba[hole] = (1.0, 0.1, 0.1, 0.85)
                    ax.imshow(rgba, origin="lower", interpolation="nearest")
                if hole is not None:
                    crop = 100.0 * hole.sum() / max(int(mask_in.sum()), 1)
                    title += f"\nhole px {int(hole.sum())} = {crop:.1f}% of judged px here"
                    whole = tile_holes.get(spec, {}).get(band)
                    if whole is not None:
                        title += f" · tile {whole:.2f}%"
                else:
                    title += f"\njudged (bright-1 %) px here: {int(mask_in.sum())}"
                ax.set_title(title, fontsize=8.5)
                _bare(ax)
        what = ("red = hole: SR < ½·LR/4 in the tile's bright-1 % LR pixels; cyan = judged pixels"
                if overlay else "dark patches inside the bright core = holes (SR < ½·LR/4)")
        fig.suptitle(f"{holes.POSTER}: nucleus A (LR y,x {POSTER_CROPS['A']}) and B {POSTER_CROPS['B']}, "
                     f"{2 * POSTER_HALF / 10:.1f}″ boxes; asinh per row (knee = crop median, "
                     "white = input p99.9)\n"
                     f"{what}; black blobs in the NISP input = zero-valued (saturated) LR pixels, not judged",
                     fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        out.append(_save(fig, f"{prefix}_{'holes' if overlay else 'plain'}.png", dpi=100))
    return out


# --------------------------------------------------------------------------- #
# flux, outskirts, resolution
# --------------------------------------------------------------------------- #

def flux_figure(bt: Battery, infos: list[ModelInfo], path: str) -> str | None:
    if not any(bt.store.read(i, "flux") for i in infos):
        return None
    geo = geometry(bt)
    cols = catalogue_columns(bt, geo)
    lr, mag = geo["lr_kron_vis"], cols["vis_mag"]
    cat, err = cols["cat:kron_vis"], cols["err:kron_vis"]
    keep = isolated(lr, cat, err)
    models = {i: bt.store.arrays(i, "flux")["kron_vis"] for i in infos if bt.store.read(i, "flux")}
    fig = _figure(13.2, 4.6)
    left, right = fig.subplots(1, 2)
    for info, sr in models.items():
        ok = keep & np.isfinite(sr)
        centres, medians = pfc.binned_median(sr[ok] / lr[ok], mag[ok])
        left.plot(centres, medians, "o-", label=info.name)
        right.hist((sr[ok] - cat[ok]) / err[ok], pfc.PULL_EDGES, density=True, histtype="step", lw=1.3,
                   label=info.name)
    right.hist((lr[keep] - cat[keep]) / err[keep], pfc.PULL_EDGES, density=True, color="0.8",
               label="Euclid input")
    left.axhline(1.0, color="k", lw=1)
    left.set(title=f"Isolated NEXUS galaxies (n={int(keep.sum())})", xlabel="VIS Kron magnitude (AB)",
             ylabel="SR / input, Kron VIS flux (median)")
    right.set(title="Kron VIS pull in catalogue σ", xlabel="(SR − MER catalogue) / MER error")
    left.legend(fontsize=8, loc="lower left")
    right.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path, dpi=100)


def growth_figure(bt: Battery, infos: list[ModelInfo], path: str) -> str | None:
    """Solid: the SR; dashed: sky-matched; dash-dot: the SR through the real
    ePSF and rebinned. The last panel is the synthetic twin."""
    if not any(bt.store.read(i, "growth") for i in infos):
        return None
    geo = growth.geometry(bt)
    r = growth.RADII
    panels = [(growth.median_curves(bt, infos), "all NEXUS galaxies"),
              (growth.median_curves(bt, infos, geo["mag"] >= 24.0), "VIS > 24 mag")]
    panels.append((growth.median_curves(bt, infos, band=3), "H, all NEXUS galaxies"))
    fig = _figure(25.0, 4.6)
    axes = fig.subplots(1, 5)
    styles = [("raw", "-", 1.8, "SR"), ("sky", "--", 1.2, "SR, sky-matched"),
              ("epsf", "-.", 1.5, "SR ⊛ Euclid ePSF")]
    for ax, (curves, title) in zip(axes[[0, 1, 3]], panels, strict=True):
        for n, info in enumerate(infos):
            cell = curves["models"].get(info.spec) or {}
            for key, ls, lw, _label in styles:
                if key in cell:
                    ax.plot(r, cell[key], ls, color=f"C{n}", lw=lw)
        ax.axhline(1.0, color="k", lw=1)
        band = "" if title.startswith("H") else "VIS "
        ax.set(title=f"{band}F_SR(<r) / F_input(<r), {title}", xlabel="aperture radius (″)",
               ylabel="median per galaxy", ylim=(0.6, 1.6))
    models = [Line2D([], [], color=f"C{n}", lw=1.8, label=info.name) for n, info in enumerate(infos)]
    kinds = [Line2D([], [], color="0.3", ls=ls, lw=lw, label=label) for _k, ls, lw, label in styles]
    axes[0].legend(handles=models, fontsize=8, loc="upper right", title="model", title_fontsize=8)
    axes[1].legend(handles=kinds, fontsize=8, loc="upper right")
    curves = panels[0][0]
    axes[2].plot(r, curves["input"], color="k", lw=2.2, label="Euclid input")
    for n, info in enumerate(infos):
        cell = curves["models"].get(info.spec) or {}
        if "norm" in cell:
            axes[2].plot(r, cell["norm"], color=f"C{n}", lw=1.6, label=info.name)
        if "epsf_norm" in cell:
            axes[2].plot(r, cell["epsf_norm"], "-.", color=f"C{n}", lw=1.5, label=f"{info.name} ⊛ ePSF")
    axes[2].axhline(1.0, color="k", lw=1)
    axes[2].set(title=f"VIS curve of growth (n = {len(geo['mag'])})", xlabel="aperture radius (″)",
                ylabel="median F(<r) / F_input(<3″)")
    axes[2].legend(fontsize=7, loc="lower right")
    syn = growth.syn_median_curves(bt, infos)
    if syn["models"]:
        ax = axes[4]
        ax.plot(r, syn["reference"], "-.", color="k", lw=1.5, label="truth ⊛ ePSF / input")
        for n, info in enumerate(infos):
            cell = syn["models"].get(info.spec)
            if cell:
                ax.plot(r, cell["truth"], "-", color=f"C{n}", lw=1.8, label=f"{info.name} SR / truth")
                ax.plot(r, cell["epsf"], "-.", color=f"C{n}", lw=1.5, label=f"{info.name} SR ⊛ ePSF / input")
        ax.axhline(1.0, color="k", lw=1)
        ax.set(title="VIS, synthetic test fields", xlabel="aperture radius (″)", ylabel="median per source",
               ylim=(0.6, 1.6))
        ax.legend(fontsize=7, loc="upper right")
    else:
        axes[4].set_axis_off()
    fig.tight_layout()
    return _save(fig, path, dpi=100)


def outskirt_figure(summary: dict, infos: list[ModelInfo], path: str) -> str | None:
    panels = [("deck", "real-like noise", "Gaussian host · real Euclid σ + lag-1"),
              ("hosts", "real NEXUS tiles", "real NEXUS tiles as hosts"),
              ("deck", "training-like noise", "Gaussian host · training (v5) σ + lag-1"),
              ("hosts", "synthetic test fields", "synthetic test fields as hosts")]
    if not any((summary.get(t) or {}).get("models") for t, _h, _l in panels):
        return None
    r = outskirts.R_EDGES
    x = 0.5 * (r[:-1] + r[1:])
    x[-1] = 5.0
    names = outskirts.ring_names()
    fig = _figure(13.0, 9.0)
    axes = fig.subplots(2, 2, sharex=True, sharey=True)
    for ax, (test, host, title) in zip(axes.ravel(), panels, strict=True):
        for n, info in enumerate(infos):
            cell = ((summary.get(test) or {}).get("models") or {}).get(info.spec, {}).get(host)
            if not cell:
                continue
            colour = f"C{n}"
            y = np.array(cell["mean4"], np.float64)
            lo = np.array([cell["boot68"][k][0] for k in names])
            hi = np.array([cell["boot68"][k][2] for k in names])
            ax.fill_between(x, lo, hi, color=colour, alpha=0.18, lw=0)
            ax.plot(x, y, "o-", color=colour, lw=2.2, ms=5,
                    label=f"{info.name}: {y[4]:.2f} | {y[5]:.2f}")
        ax.axhline(1, color="0.5", lw=1, ls=":")
        ax.axvspan(2, 2.5, color="0.5", alpha=0.08)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlim(0, 5.3)
        ax.set_ylim(0, 1.25)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left", title="at 2–2.5 | 2.5–3 r_K", title_fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("distance from centre / Kron radius")
    for ax in axes[:, 0]:
        ax.set_ylabel("injected light recovered (4-band mean)")
    fig.suptitle("Outskirt injection: 40 synthetic galaxies, host m > 19.5; "
                 "shading: 16–84 % field bootstrap",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, path)


def resolution_figure(bt: Battery, infos: list[ModelInfo], path: str) -> str | None:
    results = {i: bt.store.read(i, "resolution") for i in infos}
    results = {i: r for i, r in results.items() if r}
    if not results:
        return None
    fig = _figure(16.0, 4.2)
    axes = fig.subplots(1, 4, sharey=True)
    for ax, band in zip(axes, BANDS, strict=True):
        for n, (info, res) in enumerate(results.items()):
            pts = [(e["sb"], (e["resolution_coherence"] or {}).get("arcsec"))
                   for e in res["bands"][band]["knee_scan"]]
            pts = [(sb, a) for sb, a in pts if a is not None and sb is not None]
            if pts:
                sb, a = zip(*pts, strict=True)
                ax.plot(sb, a, "o-", color=f"C{n}", label=info.name)
        meta = next(iter(results.values()))["meta"][band]
        ax.axvline(meta["euclid_noise_sb"], color="0.6", lw=1, ls=":")
        ax.set_title(f"{band} vs JWST {meta['jwst_filter']}", fontsize=10)
        ax.set_xlabel("knee surface brightness (mag arcsec⁻²)")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("½-coherence resolution (″)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Resolution against JWST at each knee of the scan "
                 "(dotted: the Euclid noise; lower = sharper)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# synthetic fields
# --------------------------------------------------------------------------- #

def _brightest_galaxy(index: int, shape) -> tuple[int, int]:
    rows = []
    with open(os.path.join(records_dir(), "sources_test.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if int(r["field_index"]) == index and r["type"] == "galaxy" and r["off_field"] == "0":
                rows.append((float(r["flux_vis_e"]), float(r["x_pix"]), float(r["y_pix"])))
    _f, x, y = max(rows)
    h, w = shape
    return (int(np.clip(round(y) - FIELD_CROP // 2, 0, h - FIELD_CROP)),
            int(np.clip(round(x) - FIELD_CROP // 2, 0, w - FIELD_CROP)))


def field_picks(bt: Battery, infos: list[ModelInfo]) -> tuple[list[int], list[str], dict]:
    """Five test fields by the last model's per-field ∫PSNR gain over the
    baseline (one model: by its own ∫PSNR): largest, 75th, median, 25th
    percentile, smallest. ``(fields, tags, {spec: {field: dB}})``."""
    per = {}
    for info in infos:
        if bt.store.read(info, synthetic.STAGE):
            a = bt.store.arrays(info, synthetic.STAGE)
            per[info.spec] = dict(zip(a["fields"].tolist(),
                                      synthetic.per_field_integrated(a["curves"]).mean(axis=1).tolist(),
                                      strict=True))
    if not per:
        return [], [], {}
    specs = [i.spec for i in infos if i.spec in per]
    fields = sorted(per[specs[-1]])
    score = np.array([per[specs[-1]][f] - (per[specs[0]][f] if len(specs) > 1 else 0.0) for f in fields])
    order = np.argsort(score)
    n = len(order) - 1
    picks = [int(fields[order[i]]) for i in (n, round(0.75 * n), round(0.5 * n), round(0.25 * n), 0)]
    word = "gain" if len(specs) > 1 else "∫PSNR"
    tags = [f"largest {word}", f"75th pct {word}", f"median {word}", f"25th pct {word}", f"smallest {word}"]
    return picks, tags, per


def field_figures(bt: Battery, infos: list[ModelInfo], prefix: str) -> list[str]:
    picks, tags, per = field_picks(bt, infos)
    if not picks:
        return []
    named = [i for i in infos if i.spec in per]
    fwhm = target_fwhm()
    cols = ["Euclid input", *[i.name for i in named], "truth"]
    rows = []
    pairs = {i: (lr, t) for i, lr, t in test_fields(bt.limits.fields) if i in picks}
    for index in picks:
        lr, target = pairs[index]
        truth = field_truth(target, fwhm)
        lr_up = np.repeat(np.repeat(np.nan_to_num(lr), 2, 0), 2, 1) / 4.0
        sigs = [script._noise_sigma([lr[..., b]]) / 4.0 for b in RGB]
        imgs = [lr_up, *[bt.cache.get(i, field_key(index), lr) for i in named], truth]
        rows.append((index, imgs, sigs))
    out = []
    fig = _figure(2.7 * len(cols), 2.7 * len(rows) + 0.5)
    axes = fig.subplots(len(rows), len(cols), squeeze=False)
    for r, ((index, imgs, sigs), tag) in enumerate(zip(rows, tags, strict=True)):
        y0, x0 = _brightest_galaxy(index, imgs[-1].shape[:2])
        cut = (slice(y0, y0 + FIELD_CROP), slice(x0, x0 + FIELD_CROP))
        tops = [np.percentile(np.arcsinh(imgs[-1][cut][..., b] / s), 99.7)
                for b, s in zip(RGB, sigs, strict=True)]
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(rgb([img[cut][..., b] for b in RGB], sigs, tops), origin="lower",
                      interpolation="nearest")
            _bare(ax)
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
        scores = " → ".join(f"{i.name} {per[i.spec][index]:.2f}" for i in named)
        axes[r, 0].set_ylabel(f"field {index} · {tag}\n{scores} dB", fontsize=8)
    fig.suptitle(f"Synthetic test fields: {FIELD_CROP * 0.05:.0f}″ crops on each field's brightest galaxy\n"
                 "RGB = H / J / VIS, asinh knee at the input noise per SR pixel; dB = knee-integrated PSNR, "
                 "mean of 4 bands", fontsize=10)
    fig.tight_layout()
    out.append(_save(fig, f"{prefix}_zooms.png", dpi=130))
    index, imgs, sigs = rows[2]
    tops = [np.percentile(np.arcsinh(imgs[-1][..., b] / s), 99.7) for b, s in zip(RGB, sigs, strict=True)]
    h = imgs[-1].shape[0]
    fig = _figure(len(cols) * h / 128, h / 128 + 0.6)
    axes = fig.subplots(1, len(cols), squeeze=False)[0]
    for c, img in enumerate(imgs):
        axes[c].imshow(rgb([img[..., b] for b in RGB], sigs, tops), origin="lower", interpolation="nearest")
        _bare(axes[c])
        axes[c].set_title(cols[c], fontsize=12)
    fig.suptitle(f"Whole test field {index} ({h * 0.05:.1f}″, {tags[2]})", fontsize=12)
    fig.tight_layout()
    out.append(_save(fig, f"{prefix}_full.png", dpi=128))
    return out


# --------------------------------------------------------------------------- #
# real sky
# --------------------------------------------------------------------------- #

def _band_jwst(band: str, entries) -> dict:
    """``{id: (lr, JWST on the SR grid, shift)}`` of tiles with JWST in ``band``."""
    mosaic = resolution.mosaic_path(band)
    raw = (script._tile_jwst(RESOLUTION_CACHE, entries, band) if mosaic is None
           else script._mosaic_jwst(RESOLUTION_CACHE, entries, band, mosaic))
    return {t: v for t, v in raw.items() if np.isfinite(v[1]).any()}


def real_sky_picks(bt: Battery) -> dict:
    """The NEXUS crops with the most faint extended JWST F200W light (2–50 ×
    the Euclid H noise), fully covered in F200W/F150W/F090W, star-free and
    without blank input pixels; with each band's field shift and noise."""
    def compute():
        entries = nexus_entries(bt.limits.tiles)
        shift, sigma, ok = {}, {}, {}
        for band in ("VIS", "J_E", "H_E"):
            raw = _band_jwst(band, entries)
            shifts = np.array([s for _l, _j, s in raw.values() if s is not None])
            if not len(shifts):
                shifts = resolution.cached_shifts(band)
            shift[band] = [float(v) for v in np.median(shifts, axis=0)]
            b = BANDS.index(band)
            tiles = {t: (lr, jr.register(j, tuple(shift[band]))) for t, (lr, j, _s) in raw.items()}
            del raw
            sigma[band] = script._noise_sigma([lr[..., b] for lr, _j in tiles.values()]) / 4.0
            grids = {}
            for tid, (lr, jw) in tiles.items():
                h, w = jw.shape
                rows = []
                for y0 in range(0, h - REAL_CROP + 1, 32):
                    row = []
                    for x0 in range(0, w - REAL_CROP + 1, 32):
                        cut = (slice(y0, y0 + REAL_CROP), slice(x0, x0 + REAL_CROP))
                        cell = bool(np.isfinite(jw[cut]).all())
                        if band == "H_E":
                            lc = lr[y0 // 2:(y0 + REAL_CROP) // 2, x0 // 2:(x0 + REAL_CROP) // 2]
                            cell = (cell and np.nanmax(jw[cut]) <= 1500 and bool(np.isfinite(lc).all())
                                    and (lc == 0).mean() <= 0.001)
                        row.append(cell)
                    rows.append(row)
                grids[tid] = np.array(rows, bool)
                if band == "H_E":
                    sm = ndimage.gaussian_filter(np.nan_to_num(jw), 2)
                    faint = (sm > 2 * sigma[band]) & (sm < 50 * sigma[band])
                    grids[tid + ":score"] = np.array([[int(faint[y0:y0 + REAL_CROP, x0:x0 + REAL_CROP].sum())
                                                       for x0 in range(0, w - REAL_CROP + 1, 32)]
                                                      for y0 in range(0, h - REAL_CROP + 1, 32)])
            ok[band] = grids
        picks = []
        for tid in [t for t in ok["H_E"] if not t.endswith(":score")]:
            if not all(tid in ok[b] for b in ("VIS", "J_E")):
                continue
            good = ok["H_E"][tid] & ok["VIS"][tid] & ok["J_E"][tid]
            if not good.any():
                continue
            score = np.where(good, ok["H_E"][tid + ":score"], -1)
            best = None
            for iy in range(score.shape[0]):
                for ix in range(score.shape[1]):
                    if good[iy, ix] and (best is None or score[iy, ix] > best[0]):
                        best = (int(score[iy, ix]), 32 * iy + REAL_CROP // 2, 32 * ix + REAL_CROP // 2)
            picks.append((best[0], tid, best[1], best[2]))
        picks = sorted(picks, reverse=True)[:N_REAL]
        return {"picks": [list(p) for p in picks], "shift": shift, "sigma": sigma}, None

    return bt.stage(None, "real_sky_picks", compute)


def real_sky_figure(bt: Battery, infos: list[ModelInfo], path: str) -> str | None:
    info_picks = real_sky_picks(bt)
    picks = info_picks["picks"]
    if not picks:
        return None
    entries = {e.id: e for e in nexus_entries()}
    chosen = [entries[p[1]] for p in picks]
    jwst = {}
    for band in RGB_BANDS:
        raw = _band_jwst(band, chosen)
        jwst[band] = {t: jr.register(j, tuple(info_picks["shift"][band])) for t, (_l, j, _s) in raw.items()}
    sigs = [info_picks["sigma"][b] for b in RGB_BANDS]
    cols = ["Euclid input", *[i.name for i in infos], "JWST (F200W/F150W/F090W)"]
    fig = _figure(2.6 * len(cols), 2.6 * len(picks) + 0.4)
    axes = fig.subplots(len(picks), len(cols), squeeze=False)
    for r, (_score, tid, cy, cx) in enumerate(picks):
        lr = nexus_lr(entries[tid])
        h, w = jwst["H_E"][tid].shape
        y0 = int(np.clip(cy - REAL_CROP // 2, 0, h - REAL_CROP))
        x0 = int(np.clip(cx - REAL_CROP // 2, 0, w - REAL_CROP))
        cut = (slice(y0, y0 + REAL_CROP), slice(x0, x0 + REAL_CROP))
        jws = [jwst[b][tid][cut] for b in RGB_BANDS]
        tops = [np.nanpercentile(np.arcsinh(j / s), 99.7) for j, s in zip(jws, sigs, strict=True)]
        panels = [[script._runner("lr", b)(lr)[cut] for b in RGB_BANDS]]
        for info in infos:
            sr = bt.cache.get(info, nexus_key(tid), lr)
            panels.append([sr[..., BANDS.index(b)][cut] for b in RGB_BANDS])
        panels.append(jws)
        for c, chans in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(rgb(chans, sigs, tops), origin="lower", interpolation="nearest")
            _bare(ax)
            if r == 0:
                ax.set_title(cols[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(tid, fontsize=8)
    fig.suptitle(f"Real sky (NEXUS tiles, {REAL_CROP * 0.05:.1f}″ crops with the most faint extended "
                 "JWST light) "
                 "next to JWST\nRGB = H / J / VIS, asinh knee at the Euclid noise per SR pixel", fontsize=10)
    fig.tight_layout()
    return _save(fig, path, dpi=120)


#: Tests whose runs cache the NEXUS SR outputs the real-sky figure reads.
REAL_SKY_TESTS = ("resolution", "grid", "flux")


def keep_keys(bt: Battery, tests) -> set[str]:
    """NEXUS SR cache keys the figures of these tests read."""
    keys = set()
    if "grid" in tests:
        keys |= grid.figure_keys(bt)
    if "holes" in tests:
        keys |= holes.figure_keys()
    if any(t in tests for t in REAL_SKY_TESTS):
        keys |= {nexus_key(p[1]) for p in real_sky_picks(bt)["picks"]}
    return keys


def jobs(bt: Battery, infos: list[ModelInfo], summaries: dict, out_dir: str,
         tests) -> list[tuple[str, object]]:
    """``(name, job)`` per figure of the tests that ran; a job returns the
    paths it wrote (``None`` entries for a figure with nothing to draw)."""
    fig_dir = os.path.join(out_dir, "figures")

    def path(name):
        return os.path.join(fig_dir, name)

    out = [
        ("grid", "grid real VIS", lambda: [grid_figure(bt, infos, "real", "VIS", path("grid_real_VIS.png"))]),
        ("grid", "grid real H", lambda: [grid_figure(bt, infos, "real", "H", path("grid_real_H.png"))]),
        ("grid", "grid synthetic VIS",
         lambda: [grid_figure(bt, infos, "syn", "VIS", path("grid_synthetic_VIS.png"))]),
        ("holes", "poster cores", lambda: poster_figures(bt, infos, path("poster_cores"))),
        ("flux", "Kron VIS flux", lambda: [flux_figure(bt, infos, path("kron_vis.png"))]),
        ("outskirts", "outskirt recovery", lambda: [outskirt_figure(summaries.get("outskirts") or {}, infos,
                                                                    path("outskirt_recovery.png"))]),
        ("resolution", "resolution knee scan",
         lambda: [resolution_figure(bt, infos, path("resolution_knee_scan.png"))]),
        ("synthetic", "synthetic fields", lambda: field_figures(bt, infos, path("fields_synthetic"))),
        ("growth", "curve of growth", lambda: [growth_figure(bt, infos, path("curve_of_growth.png"))]),
    ]
    out = [(name, job) for test, name, job in out if test in tests]
    if any(t in tests for t in REAL_SKY_TESTS):
        out.append(("real sky", lambda: [real_sky_figure(bt, infos, path("fields_real_sky.png"))]))
    return out
