"""Pixel types by knee use (numerics: :mod:`euclid_polish.eval.knee_pixel_types`;
spec ``docs/superpowers/specs/2026-10-15-knee-attribution-design.md`` › Pixel
types): analyse a knee-attribution run into effective-knee maps,
brightness-controlled groups and knee types, each with its median stamps and
where sources sit around it, and compare two models on the same pixels. The
console's Models › Knees › Pixel types runs these as local jobs and shows the
figures; ``scripts/knee_pixel_types.py`` is the command line.

Output: ``<knee_attribution root>/pixel_types/<run id>/<model slug>/`` for an
analysis and ``pixel_types/compare_<A>_vs_<B>[_<tag>]/`` for a comparison
(PNGs and ``summary.json``).

Units: everything is in the band's LR noise σ per pixel (the field's own,
recomputed from the stored LR with lag-4 differences: resampled NISP noise is
correlated over neighbours, so records ≤ 4 kept a lag-1 σ 2.6× too small). The
"scene" is the truth on synthetic fields and, on real tiles, the LR above the
tile's clipped sky level, smoothed 3×3 (noise σ/3; the SR is not used there: a
model's holes in bright cores would pass for faint pixels). A pixel's
brightness is the scene under it in the electrons an LR pixel of that surface
brightness holds (the truth times ``scale²``, the units of the knees) over σ;
the effective knee is over σ too. Stamps (41 HR px) are in the same units,
each turned so the scene's brighter side is up; the stamp measures and the
occupancy (fraction of stamps above 1σ at each position) are taken on the
scene. A sample's knee profile is its own band's local (3×3 LR block) split.

Chain-rule baseline: the share a network with the same gradient on every knee
would get from the pixel's own LR block (``mechanical_share``). The groups come
twice: by the effective knee itself (``groups_*``) and by its excess over the
baseline (``groups_learned_*``: what the network itself prefers).

Samples whose LR block holds a blanked pixel (exactly 0) are left out of the
groups and types (``1/√(q² + 0)`` puts the share on the lowest knee whatever
the network does); their share of the pixels is reported (``blank_fraction``).

Read the sub-noise knees with care: for ``q ≪ |x|`` every ``asinh(x/q)`` is
the same log stretch up to an offset, so which knee below the noise a model
leans on is arbitrary — two seeds of one recipe differ there by ±0.6 dex.
Figures are drawn with matplotlib's object API (no pyplot), so a job thread is
safe whatever the backend.
"""
from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib.colors import AsinhNorm, LogNorm
from matplotlib.figure import Figure
from scipy.ndimage import uniform_filter

from euclid_polish.config import Config
from euclid_polish.eval import knee_attribution as ka
from euclid_polish.eval import knee_pixel_types as kt
from euclid_polish.web.helpers import atomic_files
from euclid_polish.web.helpers import knee_attribution_runs as runs

BANDS = tuple(Config.LR_INPUT_BAND_NAMES)
SCALE = 2
#: HR stamp side (odd).
STAMP = 41
#: Fewest samples a group needs to be stacked and plotted.
MIN_GROUP = 12
STAR_CLASSES = ("saturated", "star core", "star wing")
GROUP_COLOURS = {"lower": "#3b6fb6", "usual": "#7a7a7a", "higher": "#c8502d"}
EDGES = kt.SIGMA_EDGES
#: On real tiles the scene is the LR smoothed 3×3 (noise σ/3): fainter than
#: this it is noise, so it all goes to the first bin ("sky").
REAL_SKY_SIGMA = 0.1


_VIRIDIS = matplotlib.colormaps["viridis"]
#: k of the knee types (k-means).
DEFAULT_TYPES = 8
#: A progress hook: called with a short label before each step.
Progress = Callable[[str], None]


def _quiet(_label: str) -> None:
    return None


def _legend(ax, **kwargs) -> None:
    """``ax.legend`` when something carries a label (a tiny run may draw none)."""
    if ax.get_legend_handles_labels()[0]:
        ax.legend(**kwargs)


def _subplots(nrows: int = 1, ncols: int = 1, *, figsize, constrained_layout: bool = True,
              squeeze: bool = True):
    """pyplot's ``subplots`` on a bare :class:`Figure` (no pyplot state)."""
    fig = Figure(figsize=figsize, layout="constrained" if constrained_layout else None)
    return fig, fig.subplots(nrows, ncols, squeeze=squeeze)


# --------------------------------------------------------------------------- #
# samples and stamps
# --------------------------------------------------------------------------- #

def load_samples(entry: dict, kind: str) -> dict:
    """Every sample of one kind of field, concatenated, with its field index
    and its band's σ in that field."""
    fields = [f for f in entry["fields"] if f["kind"] == kind]
    cols: dict[str, list] = {k: [] for k in ("band", "y", "x", "class", "value_e", "weight",
                                             "share", "field", "sigma")}
    for i, f in enumerate(fields):
        s = f["samples"]
        n = len(s["band"])
        band = np.asarray(s["band"], np.int64)
        cols["band"].append(band)
        cols["y"].append(np.asarray(s["y"], np.int64))
        cols["x"].append(np.asarray(s["x"], np.int64))
        cols["class"].append(np.asarray(s["class"]))
        cols["value_e"].append(np.asarray(s["value_e"], np.float64))
        cols["weight"].append(np.asarray(s["weight"], np.float64))
        cols["share"].append(np.asarray(s["knee_share_own_local"], np.float64).reshape(n, -1))
        cols["field"].append(np.full(n, i))
        cols["sigma"].append(np.asarray(f["noise_sigma_e"], np.float64)[band])
    out = {k: np.concatenate(v) for k, v in cols.items()} if fields else {}
    if fields:
        out["bright"] = out["value_e"] * SCALE ** 2 / out["sigma"]      # LR-equivalent σ
    return {"fields": fields, **out}


def field_planes(run_id: str, spec: str, fields: list[dict], kind: str) -> list[dict]:
    """Per field: the SR, the reference and scene (the truth on synthetic
    fields; on real tiles the LR above its sky level, smoothed 3×3, repeated
    to the HR grid) and the LR, float32. A real tile's sky level per band
    (:func:`kt.sky_level`) goes to its field row as ``sky_level_e``."""
    out = []
    for f in fields:
        with np.load(runs._field_path(run_id, spec, f["ref"])) as stored:
            sr = np.asarray(stored["sr"], np.float32)
            lr = np.asarray(stored["lr"], np.float32)
            sigma = ka.noise_sigma(lr).astype(np.float64)     # per-pixel σ (record ≤ 4 kept lag 1)
            f["sigma_e"] = sigma.tolist()
            if kind == "synthetic" and "truth" in stored:
                ref = np.asarray(stored["truth"], np.float32)
                scene = ref
            else:
                sky = np.array([kt.sky_level(lr[..., c], sigma[c]) for c in range(lr.shape[2])])
                f["sky_level_e"] = sky.tolist()
                above = lr - sky.astype(np.float32)
                smooth = uniform_filter(above, size=(3, 3, 1), mode="nearest") / SCALE ** 2
                ref = np.repeat(np.repeat(smooth, SCALE, axis=0), SCALE, axis=1)[:sr.shape[0], :sr.shape[1]]
                scene = ref
        out.append({"sr": sr, "ref": ref, "scene": scene, "lr": lr, "sigma": sigma})
    return out


def stamps(samples: dict, planes: list[dict]) -> dict[str, np.ndarray]:
    """SR, reference and scene stamps of every sample, in LR-equivalent σ,
    aligned on the scene (brighter side up)."""
    n = samples["band"].size
    out = {k: np.empty((n, STAMP, STAMP), np.float32) for k in ("sr", "ref", "scene")}
    for f in np.unique(samples["field"]):
        for b in np.unique(samples["band"]):
            rows = np.flatnonzero((samples["field"] == f) & (samples["band"] == b))
            if not rows.size:
                continue
            y, x = samples["y"][rows], samples["x"][rows]
            unit = (SCALE ** 2 / samples["sigma"][rows])[:, None, None]
            for key in out:
                out[key][rows] = kt.cut_stamps(planes[f][key][..., b], y, x, STAMP) * unit
    moves = kt.alignment(out["scene"])
    return {k: kt.apply_alignment(v, moves) for k, v in out.items()}


def lr_blocks(samples: dict, planes: list[dict]) -> np.ndarray:
    """Each sample's own-band LR block (the local split's 3×3, electrons;
    NaN beyond the field): ``(N, 9)``."""
    r = ka.LOCAL_RADIUS
    offs = np.arange(-r, r + 1)
    out = np.full((samples["band"].size, offs.size ** 2), np.nan)
    for f in np.unique(samples["field"]):
        rows = np.flatnonzero(samples["field"] == f)
        padded = np.pad(planes[f]["lr"], ((r, r), (r, r), (0, 0)), constant_values=np.nan)
        cy = samples["y"][rows] // SCALE + r
        cx = samples["x"][rows] // SCALE + r
        yy = (cy[:, None] + offs)[:, :, None]
        xx = (cx[:, None] + offs)[:, None, :]
        out[rows] = padded[yy, xx, samples["band"][rows][:, None, None]].reshape(rows.size, -1)
    return out


def _median(values: np.ndarray) -> float | None:
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else None


def describe(sub: dict, sel: np.ndarray, measures: dict[str, np.ndarray],
             class_names: tuple[str, ...], log_q: np.ndarray, count: np.ndarray) -> dict:
    w = sub["weight"][sel]
    centre = np.maximum(np.abs(measures["centre"][sel]), 1.0)
    return {"n": int(sel.sum()), "weight": float(w.sum()),
            "log_q_sigma_median": _median(log_q[sel]), "knee_count_median": _median(count[sel]),
            "log_q_mech_sigma_median": _median(sub["log_q_mech"][sel]),
            "learned_dex_median": _median(log_q[sel] - sub["log_q_mech"][sel]),
            "lr_block_sigma_median": _median(sub["lr_block_sigma"][sel]),
            "bright_sigma_median": _median(sub["bright"][sel]),
            "mean_share": [float(v) for v in np.average(sub["share"][sel], axis=0, weights=w)],
            "classes": kt.weighted_mix(sub["class"][sel], w, class_names),
            "nearest_source_px": _median(measures["nearest_source"][sel]),
            "stamp_peak_sigma": _median(measures["stamp_peak"][sel]),
            "peak_ratio": _median(measures["peak_ratio"][sel]),
            "local_max": float(np.average(measures["local_max"][sel], weights=w)),
            "rel_slope": _median(measures["slope"][sel] / centre)}


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def _bin_label(b: int, kind: str = "synthetic") -> str:
    lo, hi = EDGES[b], EDGES[b + 1]
    if b == 0:
        return f"sky < {REAL_SKY_SIGMA:g}σ" if kind == "real" else f"< {hi:g}σ"
    if b == len(EDGES) - 2:
        return f"≥ {lo:g}σ"
    return f"{lo:g}–{hi:g}σ"


def _norm(images) -> AsinhNorm:
    finite = np.concatenate([np.asarray(i)[np.isfinite(i)] for i in images] or [np.zeros(1)])
    finite = finite if finite.size else np.zeros(1)
    lo, hi = np.percentile(finite, [0.5, 99.9])
    return AsinhNorm(linear_width=0.3, vmin=min(float(lo), -0.3), vmax=max(float(hi), 1.0))


def _show(ax, image, norm, title=None, cmap="magma"):
    ax.imshow(image, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    c = STAMP // 2
    ax.plot([c], [c], marker="+", color="#36d1dc", ms=6, mew=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def _occ(ax, image, title=None):
    ax.imshow(image, origin="lower", cmap="cividis", vmin=0.0, vmax=1.0, interpolation="nearest")
    c = STAMP // 2
    ax.plot([c], [c], marker="+", color="#ff4f79", ms=6, mew=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def knee_map(path: Path, analysed: dict, knees, kind: str, title: str):
    fig, axes = _subplots(1, len(BANDS), figsize=(4.3 * len(BANDS), 4.1), constrained_layout=True)
    centres = 0.5 * (np.log10(EDGES[:-1]) + np.log10(EDGES[1:]))
    centres[0] = -2.5
    for ax, band in zip(axes, BANDS, strict=True):
        a = analysed.get(band)
        if a is None:
            ax.set_axis_off()
            continue
        logk = np.log10(knees / a["sigma"])
        x = np.log10(np.clip(a["bright"], 10 ** -2.95, 1e5))
        ok = np.isfinite(a["log_q"])
        ax.hist2d(x[ok], a["log_q"][ok], bins=[np.linspace(-3, 5, 65), np.linspace(logk[0], logk[-1], 49)],
                  weights=a["weight"][ok], cmap="Greys", norm=LogNorm(), rasterized=True)
        keep = [i for i, st in enumerate(a["bins"]) if st["n"] >= MIN_GROUP]
        ax.plot(centres[keep], [a["bins"][i]["median"] for i in keep], color="#c8502d", lw=1.8,
                marker="o", ms=3, label="median")
        ax.fill_between(centres[keep], [a["bins"][i]["q20"] for i in keep],
                        [a["bins"][i]["q80"] for i in keep], color="#c8502d", alpha=0.18, lw=0,
                        label="20–80 %")
        ax.plot(centres[keep], [a["mech_bins"][i] for i in keep], color="#e9a23b", lw=1.4, ls="-.",
                marker="s", ms=2.5, label="chain-rule baseline")
        ax.plot([-3, 5], [-3, 5], ls="--", color="#3b6fb6", lw=1, label="knee = brightness")
        for q, lq in zip(knees, logk, strict=True):
            ax.axhline(lq, color="#cccccc", lw=0.5, zorder=0)
            ax.text(4.95, lq, f"{q:g} e⁻", fontsize=6, color="#888888", ha="right", va="bottom")
        ax.axvline(0, ls=":", color="#2a9d8f", lw=1)
        ax.axvline(-2, color="#999999", lw=0.6)
        ax.text(-2.5, logk[-1], "empty\nsky", fontsize=6, ha="center", va="top", color="#666666")
        ax.set_xlim(-3, 5)
        ax.set_ylim(logk[0] - 0.1, logk[-1] + 0.1)
        ax.set_title(f"{band} (σ = {a['sigma']:.3g} e⁻; {a['blank_fraction']:.1%} blank left out)",
                     fontsize=9)
        ax.set_xlabel("log₁₀ pixel brightness / σ")
        ax.set_ylabel("log₁₀ effective knee q̄ / σ")
    _legend(axes[0], fontsize=7, loc="lower right")
    fig.suptitle(f"{title} — {kind}: the knee each pixel is read through vs its brightness "
                 "(grey = how common such pixels are)")
    fig.savefig(path, dpi=130)


def group_grid(path: Path, a: dict, band: str, kind: str, title: str, variant: str = "raw") -> bool:
    grouped = a["variants"][variant]
    rows = [b for b, cells in sorted(grouped["group_stacks"].items()) if len(cells) == len(kt.GROUPS)]
    if not rows:
        return False
    ref_name = "truth" if kind == "synthetic" else "LR 3×3"
    fig, axes = _subplots(len(rows), 3 * len(kt.GROUPS),
                             figsize=(1.3 * 3 * len(kt.GROUPS), 1.55 * len(rows) + 0.7),
                             squeeze=False, constrained_layout=True)
    for r, b in enumerate(rows):
        cells = grouped["group_stacks"][b]
        norm = _norm([c["sr"] for c in cells.values()] + [c["ref"] for c in cells.values()])
        for g, name in enumerate(kt.GROUPS):
            c = cells[name]
            head = f"{_bin_label(b, kind)}\n" if g == 0 else "\n"
            _show(axes[r, 3 * g], c["sr"], norm, f"{head}{name}: SR · n={c['n']}")
            _show(axes[r, 3 * g + 1], c["ref"], norm,
                  f"q̄={10 ** c['log_q']:.2g}σ ({c['learned']:+.2f} dex)\n{ref_name}")
            _occ(axes[r, 3 * g + 2], c["occ"], f"\nsources (>1σ) · d={c['dist']:.0f}px")
    how = ("knee" if variant == "raw" else
           "knee than the chain rule alone gives (the network's own preference)")
    fig.suptitle(f"{title} — {kind} {band}: pixels as bright as each other, read through a LOWER, the USUAL "
                 f"or a HIGHER {how}\nmedian stamps in σ (brighter side up); q̄ = median effective knee "
                 f"(± dex over the chain-rule baseline); occupancy 0–1; d = median distance to a >1σ pixel",
                 fontsize=8)
    fig.savefig(path, dpi=120)
    return True


FEATURES = [("lr_block_sigma_median", "LR 3×3 block mean |x| (σ)", True),
            ("nearest_source_px", "distance to nearest >1σ pixel (px)", False),
            ("stamp_peak_sigma", "brightest pixel within 20 px (σ)", True),
            ("peak_ratio", "brightest within 4 px / centre", True),
            ("local_max", "fraction at a local maximum", False),
            ("rel_slope", "|∇| / centre", False)]


def group_features(path: Path, analysed: dict, kind: str, title: str, variant: str = "raw"):
    extra = ("star fraction", STAR_CLASSES) if kind == "synthetic" else ("blank fraction", ("blank",))
    n_cols = len(FEATURES) + 1
    fig, axes = _subplots(len(BANDS), n_cols, figsize=(2.9 * n_cols, 2.2 * len(BANDS)),
                             squeeze=False, constrained_layout=True)
    centres = np.sqrt(EDGES[:-1] * EDGES[1:])
    centres[0] = 10 ** -2.5
    for r, band in enumerate(BANDS):
        a = analysed.get(band)
        if a is None:
            continue
        for name in kt.GROUPS:
            stats = {b: s["groups"][name] for b, s in enumerate(a["variants"][variant]["summary_bins"])
                     if name in s["groups"]}
            xs = [centres[b] for b in stats]
            for c, (key, _, _) in enumerate(FEATURES):
                axes[r, c].plot(xs, [st[key] for st in stats.values()], marker="o", ms=3,
                                color=GROUP_COLOURS[name], label=name)
            axes[r, -1].plot(xs, [sum(st["classes"].get(k, 0.0) for k in extra[1]) for st in stats.values()],
                             marker="o", ms=3, color=GROUP_COLOURS[name], label=name)
        for c, (_, label, logy) in enumerate([*FEATURES, (None, extra[0], False)]):
            ax = axes[r, c]
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("symlog", linthresh=1.0)
            ax.set_title(f"{band}: {label}", fontsize=8)
            ax.tick_params(labelsize=7)
    _legend(axes[0, 0], fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("pixel brightness / σ", fontsize=8)
    what = "knee" if variant == "raw" else "knee than the chain rule gives"
    scene = "truth" if kind == "synthetic" else "LR smoothed 3×3"
    fig.suptitle(f"{title} — {kind}: how pixels read through a lower/higher {what} differ from pixels "
                 f"as bright (measured on the {scene}; medians per brightness decade)",
                 fontsize=9)
    fig.savefig(path, dpi=120)


def type_figure(path: Path, a: dict, band: str, kind: str, knees, title: str):
    types = a["type_stacks"]
    if not types:
        return
    n = len(types)
    colours = _VIRIDIS(np.linspace(0.05, 0.95, n))
    fig, axes = _subplots(5, n, figsize=(1.9 * n, 9.0), squeeze=False, constrained_layout=True)
    labels = [f"{q:g}" for q in knees]
    for j, t in enumerate(types):
        axes[0, j].bar(range(len(knees)), t["mean_share"], color=colours[j])
        axes[0, j].set_xticks(range(len(knees)), labels, rotation=90, fontsize=6)
        axes[0, j].set_ylim(0, 1)
        axes[0, j].tick_params(labelsize=6)
        top = "\n".join(f"{k} {v:.0%}" for k, v in list(t["classes"].items())[:2])
        axes[0, j].set_title(f"type {j} · {t['prevalence']:.1%} of pixels (n={t['n']})\n"
                             f"q̄={10 ** t['log_q']:.2g}σ · b={t['bright_median']:.2g}σ\n{top}", fontsize=7)
        norm = _norm([t["sr"], t["ref"]])
        _show(axes[1, j], t["sr"], norm, "SR (σ)")
        _show(axes[2, j], t["ref"], norm, ("truth" if kind == "synthetic" else "LR 3×3") + " (σ)")
        _occ(axes[3, j], t["occ"], f"sources >1σ · d={t['dist']:.0f}px")
        x = np.log10(np.clip(t["bright"], 10 ** -2.95, 1e5))
        axes[4, j].hist(x, bins=np.linspace(-3, 5, 33), weights=t["weights"], color=colours[j])
        axes[4, j].axvline(0, ls=":", color="#2a9d8f", lw=1)
        axes[4, j].set_yticks([])
        axes[4, j].tick_params(labelsize=6)
        axes[4, j].set_xlabel("log₁₀ brightness / σ", fontsize=7)
    axes[0, 0].set_ylabel("mean knee share (e⁻ knees)", fontsize=7)
    fig.suptitle(f"{title} — {kind} {band}: knee types (k-means of the share profile, every brightness "
                 f"decade weighted alike), ordered by effective knee", fontsize=8)
    fig.savefig(path, dpi=120)


def types_by_brightness(path: Path, analysed: dict, kind: str, title: str):
    fig, axes = _subplots(1, len(BANDS), figsize=(4.0 * len(BANDS), 3.3), constrained_layout=True)
    n_bins = len(EDGES) - 1
    for ax, band in zip(axes, BANDS, strict=True):
        a = analysed.get(band)
        if a is None:
            ax.set_axis_off()
            continue
        k = int(a["types"].max()) + 1 if a["types"].size else 0
        colours = _VIRIDIS(np.linspace(0.05, 0.95, max(k, 1)))
        frac = np.zeros((k, n_bins))
        for b in range(n_bins):
            sel = (a["bin"] == b) & (a["types"] >= 0)
            if a["weight"][sel].sum() > 0:
                frac[:, b] = np.bincount(a["types"][sel], weights=a["weight"][sel], minlength=k) \
                    / a["weight"][sel].sum()
        bottom = np.zeros(n_bins)
        for j in range(k):
            ax.bar(range(n_bins), frac[j], bottom=bottom, color=colours[j], label=f"type {j}", width=0.9)
            bottom += frac[j]
        ax.set_xticks(range(n_bins), [_bin_label(b, kind) for b in range(n_bins)], rotation=50, fontsize=7,
                      ha="right")
        ax.set_title(band)
        ax.set_ylim(0, 1)
    _legend(axes[-1], fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle(f"{title} — {kind}: which knee types make up each brightness decade", fontsize=9)
    fig.savefig(path, dpi=120)


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #

def _stack(st: dict, sel: np.ndarray) -> dict:
    return {"sr": kt.stack(st["sr"][sel]), "ref": kt.stack(st["ref"][sel]),
            "occ": kt.occupancy(st["scene"][sel])}


def _groups(sub: dict, values: np.ndarray, st: dict, measures: dict, class_names, log_q, count,
            kind: str) -> dict:
    """Brightness-controlled groups of ``values`` (the effective knee, or
    its excess over the baseline): stacks and per-bin summaries."""
    bins, groups, stats = kt.residual_groups(values, sub["bright"], sub["weight"], EDGES)
    group_stacks: dict[int, dict] = {}
    summary_bins = []
    for i, s in enumerate(stats):
        row = dict(s, label=_bin_label(i, kind), groups={})
        for g, name in enumerate(kt.GROUPS):
            sel = (bins == i) & (groups == g)
            if sel.sum() < MIN_GROUP:
                continue
            row["groups"][name] = describe(sub, sel, measures, class_names, log_q, count)
            group_stacks.setdefault(i, {})[name] = _stack(st, sel) | {
                "n": int(sel.sum()), "log_q": float(np.nanmedian(log_q[sel])),
                "learned": float(np.nanmedian(log_q[sel] - sub["log_q_mech"][sel])),
                "dist": float(np.median(measures["nearest_source"][sel]))}
        summary_bins.append(row)
    return {"bins": bins, "stats": stats, "group_stacks": group_stacks, "summary_bins": summary_bins}


def prepare(run_id: str, spec: str, entry: dict, kind: str) -> tuple[dict, list[dict]] | None:
    """One kind of field of one model: the samples with their derived
    columns (``bright`` = the scene in σ, ``blank``, ``log_q`` and the
    chain-rule ``log_q_mech`` in σ, ``lr_block_sigma``) and the field
    planes; ``None`` without fields of that kind."""
    samples = load_samples(entry, kind)
    if not samples["fields"]:
        return None
    knees = np.asarray(entry["knees"], np.float64)
    planes = field_planes(run_id, spec, samples["fields"], kind)
    for f in np.unique(samples["field"]):                    # σ of the stored LR, lag-4 (true per pixel)
        rows = samples["field"] == f
        samples["sigma"][rows] = planes[f]["sigma"][samples["band"][rows]]
    samples["bright"] = samples["value_e"] * SCALE ** 2 / samples["sigma"]
    if kind == "real":                                       # the scene under the pixel, not the SR
        for f in np.unique(samples["field"]):
            rows = samples["field"] == f
            samples["bright"][rows] = (planes[f]["scene"][samples["y"][rows], samples["x"][rows],
                                                          samples["band"][rows]]
                                       * SCALE ** 2 / samples["sigma"][rows])
        samples["bright"] = np.where(samples["bright"] < REAL_SKY_SIGMA, 10 ** -2.5, samples["bright"])
    blocks = lr_blocks(samples, planes)
    log_sigma = np.log10(samples["sigma"])
    with np.errstate(all="ignore"):
        samples["blank"] = (blocks == 0).any(axis=1)
        samples["log_q"] = kt.effective_knee(samples["share"], knees) - log_sigma
        samples["log_q_mech"] = kt.effective_knee(kt.mechanical_share(blocks, knees), knees) - log_sigma
        samples["lr_block_sigma"] = np.nanmean(np.abs(blocks), axis=1) / samples["sigma"]
    return samples, planes


def analyse_band(sub: dict, planes: list[dict], knees: np.ndarray, class_names, k: int,
                 rng: np.random.Generator, kind: str) -> tuple[dict, dict]:
    sigma = float(np.median(sub["sigma"]))
    blank = sub["blank"]
    blank_fraction = float(sub["weight"][blank].sum() / sub["weight"].sum())
    sub = {key: (v[~blank] if isinstance(v, np.ndarray) else v) for key, v in sub.items()}
    log_q = sub["log_q"]
    count = kt.knee_count(sub["share"])
    st = stamps(sub, planes)
    measures = kt.stamp_features(st["scene"])
    variants = {"raw": _groups(sub, log_q, st, measures, class_names, log_q, count, kind),
                "learned": _groups(sub, log_q - sub["log_q_mech"], st, measures, class_names, log_q, count,
                                   kind)}
    bins = variants["raw"]["bins"]
    mech_bins = [_median(sub["log_q_mech"][bins == b]) for b in range(len(EDGES) - 1)]
    maxima = kt.neighbourhood_max(st["scene"])
    look = {"raw": kt.look_around(log_q, bins, maxima),
            "learned": kt.look_around(log_q - sub["log_q_mech"], bins, maxima)}

    types = kt.knee_types(sub["share"], kt.balanced_weights(bins, sub["weight"]), knees, k, rng)
    total_w = sub["weight"][types >= 0].sum()
    type_stacks, type_summary = [], []
    for j in range(int(types.max()) + 1 if types.size else 0):
        sel = types == j
        if not sel.any():
            continue
        d = describe(sub, sel, measures, class_names, log_q, count)
        d["prevalence"] = float(sub["weight"][sel].sum() / total_w)
        d["by_brightness"] = [float(sub["weight"][sel & (bins == b)].sum()) for b in range(len(EDGES) - 1)]
        type_summary.append(d)
        type_stacks.append(_stack(st, sel) | {
            "mean_share": d["mean_share"], "classes": d["classes"], "prevalence": d["prevalence"],
            "log_q": d["log_q_sigma_median"], "bright_median": d["bright_sigma_median"], "n": d["n"],
            "dist": d["nearest_source_px"], "bright": sub["bright"][sel], "weights": sub["weight"][sel]})

    analysed = {"bright": sub["bright"], "log_q": log_q, "log_q_mech": sub["log_q_mech"],
                "weight": sub["weight"], "bins": variants["raw"]["stats"], "bin": bins, "types": types,
                "sigma": sigma, "variants": variants, "mech_bins": mech_bins, "type_stacks": type_stacks,
                "blank_fraction": blank_fraction, "look": look}
    summary = {"noise_sigma_e": sigma, "n": int(sub["band"].size), "blank_fraction": blank_fraction,
               "sky_level_sigma_median": _median([f["sky_level_e"][int(sub["band"][0])] / f["sigma_e"]
                                                  [int(sub["band"][0])] for f in sub["fields"]
                                                  if "sky_level_e" in f]) if sub["band"].size else None,
               "bins": variants["raw"]["summary_bins"], "bins_learned": variants["learned"]["summary_bins"],
               "baseline_log_q_sigma_by_bin": mech_bins, "types": type_summary,
               "look_around": {"radii_px": list(kt.LOOK_RADII),
                               **{v: {_bin_label(b, kind): [x if np.isfinite(x) else None for x in r]
                                      for b, r in look[v].items()} for v in look}}}
    return analysed, summary


def look_around_figure(path: Path, analysed: dict, kind: str, title: str):
    fig, axes = _subplots(2, len(BANDS), figsize=(3.9 * len(BANDS), 6.4), squeeze=False,
                             constrained_layout=True)
    colours = _VIRIDIS(np.linspace(0.0, 0.92, len(EDGES) - 1))
    radii = np.asarray(kt.LOOK_RADII, np.float64)
    for c, band in enumerate(BANDS):
        a = analysed.get(band)
        if a is None:
            continue
        for r, (variant, label) in enumerate((("raw", "effective knee"),
                                              ("learned", "network's own preference"))):
            ax = axes[r, c]
            for b, rho in sorted(a["look"][variant].items()):
                ax.plot(radii, rho, marker="o", ms=3, color=colours[b], label=_bin_label(b, kind))
            ax.axhline(0, color="#999999", lw=0.6)
            ax.set_ylim(-0.3, 1.0)
            ax.set_title(f"{band}: {label}", fontsize=8)
            ax.set_xlabel("radius r (HR px)", fontsize=8)
            ax.set_ylabel("Spearman ρ with brightest within r", fontsize=7)
            ax.tick_params(labelsize=7)
    _legend(axes[0, -1], fontsize=6, title="pixel brightness", title_fontsize=6, loc="lower right")
    scene = "truth" if kind == "synthetic" else "LR 3×3"
    fig.suptitle(f"{title} — {kind}: how far around a pixel the knee choice looks (at fixed brightness, the "
                 f"rank correlation of its knee with the brightest {scene} pixel within r)", fontsize=8)
    fig.savefig(path, dpi=120)


def mechanics(path: Path, analysed: dict, knees, kind: str, title: str):
    fig, axes = _subplots(1, len(BANDS), figsize=(4.0 * len(BANDS), 3.9), constrained_layout=True)
    for ax, band in zip(axes, BANDS, strict=True):
        a = analysed.get(band)
        if a is None:
            ax.set_axis_off()
            continue
        logk = np.log10(knees / a["sigma"])
        ok = np.isfinite(a["log_q"]) & np.isfinite(a["log_q_mech"])
        edges = np.linspace(logk[0], logk[-1], 49)
        ax.hist2d(a["log_q_mech"][ok], a["log_q"][ok], bins=[edges, edges], weights=a["weight"][ok],
                  cmap="Greys", norm=LogNorm(), rasterized=True)
        ax.plot(edges[[0, -1]], edges[[0, -1]], ls="--", color="#3b6fb6", lw=1, label="network = chain rule")
        dex = a["log_q"][ok] - a["log_q_mech"][ok]
        lo, med, hi = kt.weighted_quantile(dex, a["weight"][ok], (0.2, 0.5, 0.8))
        ax.set_title(f"{band}: network − chain rule = {med:+.2f} dex (20–80 %: {lo:+.2f}…{hi:+.2f})",
                     fontsize=8)
        ax.set_xlabel("log₁₀ baseline knee / σ (same gradient on every knee)")
        ax.set_ylabel("log₁₀ effective knee / σ")
    _legend(axes[0], fontsize=7, loc="upper left")
    fig.suptitle(f"{title} — {kind}: how much of the knee choice the chain rule alone explains "
                 "(grey = prevalence)", fontsize=9)
    fig.savefig(path, dpi=120)


def analyse_kind(run_id: str, spec: str, entry: dict, kind: str, out: Path, k: int, seed: int,
                 progress: Progress = _quiet) -> dict:
    got = prepare(run_id, spec, entry, kind)
    if got is None:
        return {}
    samples, planes = got
    knees = np.asarray(entry["knees"], np.float64)
    class_names = ka.TRUTH_CLASSES if kind == "synthetic" else ka.REAL_CLASSES
    analysed: dict[str, dict] = {}
    summary: dict[str, dict] = {}
    for b, band in enumerate(BANDS):
        sel = samples["band"] == b
        if not sel.any():
            continue
        progress(f"{kind} {band}: {int(sel.sum())} samples")
        sub = {key: (v[sel] if isinstance(v, np.ndarray) else v) for key, v in samples.items()}
        analysed[band], summary[band] = analyse_band(sub, planes, knees, class_names, k,
                                                     np.random.default_rng([seed, b]), kind)
        group_grid(out / f"groups_{kind}_{band}.png", analysed[band], band, kind, spec)
        group_grid(out / f"groups_learned_{kind}_{band}.png", analysed[band], band, kind, spec, "learned")
        type_figure(out / f"types_{kind}_{band}.png", analysed[band], band, kind, knees, spec)
    knee_map(out / f"knee_map_{kind}.png", analysed, knees, kind, spec)
    group_features(out / f"group_features_{kind}.png", analysed, kind, spec)
    group_features(out / f"group_features_learned_{kind}.png", analysed, kind, spec, "learned")
    mechanics(out / f"mechanics_{kind}.png", analysed, knees, kind, spec)
    look_around_figure(out / f"look_around_{kind}.png", analysed, kind, spec)
    types_by_brightness(out / f"types_by_brightness_{kind}.png", analysed, kind, spec)
    write_analysis_views(out, analysed, knees, kind, spec)
    return summary


# --------------------------------------------------------------------------- #
# comparing two models
# --------------------------------------------------------------------------- #

def _keyed(samples: dict, kind: str) -> dict:
    """``(field ref, band, y, x) → row`` of the non-blank samples (first
    occurrence: the uniform draw comes first)."""
    out: dict = {}
    refs = [f["ref"] for f in samples["fields"]]
    for i in np.flatnonzero(~samples["blank"]):
        key = (refs[samples["field"][i]], int(samples["band"][i]), int(samples["y"][i]), int(samples["x"][i]))
        out.setdefault(key, i)
    return out


def _bin_medians(values: np.ndarray, bins: np.ndarray, weights: np.ndarray) -> list:
    n_bins = len(EDGES) - 1
    out = []
    for b in range(n_bins):
        sel = (bins == b) & np.isfinite(values)
        out.append(kt.weighted_quantile(values[sel], weights[sel], (0.2, 0.5, 0.8)).tolist()
                   if sel.sum() >= MIN_GROUP else None)
    return out


#: Model colours of the overlays (up to eight models).
MODEL_COLOURS = ["#3b6fb6", "#c8502d", "#2a9d8f", "#8e44ad", "#d4a017", "#5b5b5b", "#e377c2", "#17becf"]


def compare_curves(path: Path, prepared: dict, kind: str, title: str | None = None):
    """Effective knee (with each model's chain-rule baseline, dotted) and the
    network's own preference per brightness decade, every model overlaid
    (median and 20–80 % band)."""
    names = list(prepared)
    colours = MODEL_COLOURS
    centres = 0.5 * (np.log10(EDGES[:-1]) + np.log10(EDGES[1:]))
    centres[0] = -2.5
    fig, axes = _subplots(2, len(BANDS), figsize=(4.2 * len(BANDS), 7.0), squeeze=False,
                             constrained_layout=True)
    for c, band in enumerate(BANDS):
        for m, name in enumerate(names):
            s = prepared[name]["samples"]
            sel = (s["band"] == c) & ~s["blank"]
            bins = kt.brightness_bin(s["bright"][sel], EDGES)
            w = s["weight"][sel]
            for r, values in enumerate((s["log_q"][sel], s["log_q"][sel] - s["log_q_mech"][sel])):
                q = _bin_medians(values, bins, w)
                keep = [i for i, v in enumerate(q) if v is not None]
                ax = axes[r, c]
                ax.plot(centres[keep], [q[i][1] for i in keep], color=colours[m], marker="o", ms=3,
                        label=name)
                ax.fill_between(centres[keep], [q[i][0] for i in keep], [q[i][2] for i in keep],
                                color=colours[m], alpha=0.12, lw=0)
            mech = _bin_medians(s["log_q_mech"][sel], bins, w)
            keep = [i for i, v in enumerate(mech) if v is not None]
            axes[0, c].plot(centres[keep], [mech[i][1] for i in keep], color=colours[m], ls=":", lw=1)
        axes[0, c].plot([-3, 5], [-3, 5], ls="--", color="#999999", lw=0.8)
        axes[0, c].set_title(f"{band}: effective knee (dotted: chain-rule baseline)", fontsize=8)
        axes[1, c].set_title(f"{band}: network's own preference over the baseline", fontsize=8)
        axes[0, c].set_ylabel("log₁₀ q̄ / σ", fontsize=8)
        axes[1, c].set_ylabel("dex over baseline", fontsize=8)
        for r in (0, 1):
            axes[r, c].set_xlim(-3, 5)
            axes[r, c].set_xlabel("log₁₀ pixel brightness / σ", fontsize=8)
            axes[r, c].tick_params(labelsize=7)
    _legend(axes[0, 0], fontsize=7)
    fig.suptitle(f"{' vs '.join(names)} — {kind}: the knee each pixel is read through (median, 20–80 %)",
                 fontsize=9)
    fig.savefig(path, dpi=120)


def compare_share_diff(path: Path, prepared: dict, kind: str):
    """Mean own-band local knee share of model B minus model A, per
    brightness decade and knee, on the union of the two knee sets (a knee
    only one model reads carries share 0 in the other, and says so)."""
    (a_name, a), (b_name, b) = list(prepared.items())[:2]
    knees = np.unique(np.round(np.concatenate([a["knees"], b["knees"]]), 12))
    n_bins = len(EDGES) - 1
    fig, axes = _subplots(1, len(BANDS), figsize=(4.0 * len(BANDS), 3.9), constrained_layout=True)
    for c, band in enumerate(BANDS):
        means = []
        for m in (a, b):
            s = m["samples"]
            sel = (s["band"] == c) & ~s["blank"]
            bins = kt.brightness_bin(s["bright"][sel], EDGES)
            cols = np.searchsorted(knees, np.round(m["knees"], 12))         # the model's knees in the union
            block = np.full((len(knees), n_bins), np.nan)
            for i in range(n_bins):
                rows = bins == i
                if rows.sum() >= MIN_GROUP:
                    block[:, i] = 0.0
                    block[cols, i] = np.average(s["share"][sel][rows], axis=0, weights=s["weight"][sel][rows])
            means.append(block)
        diff = means[1] - means[0]
        ax = axes[c]
        lim = max(0.05, float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 0.05)
        im = ax.imshow(diff, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for (k, i), v in np.ndenumerate(diff):
            if np.isfinite(v) and abs(v) >= 0.02:
                ax.text(i, k, f"{v:+.2f}"[:5].replace("0.", "."), ha="center", va="center", fontsize=6)
        ax.set_xticks(range(n_bins), [_bin_label(i, kind) for i in range(n_bins)], rotation=50,
                      ha="right", fontsize=6)
        only = {name: set(np.round(m["knees"], 12)) for name, m in ((a_name, a), (b_name, b))}
        ax.set_yticks(range(len(knees)), [
            f"{q:g} e⁻" + ("" if all(q in ks for ks in only.values())
                           else f" ({next(n for n, ks in only.items() if q in ks)} only)")
            for q in knees], fontsize=6)
        ax.set_title(band, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{kind}: mean knee share, {b_name} − {a_name} (red: {b_name} leans on that knee more)",
                 fontsize=9)
    fig.savefig(path, dpi=120)


def compare_look_around(path: Path, prepared: dict, kind: str) -> dict:
    """:func:`kt.look_around` of the effective knee, every model overlaid
    (one colour per brightness decade, one line style per model)."""
    styles = ["-", "--", ":", "-."]
    colours = _VIRIDIS(np.linspace(0.0, 0.92, len(EDGES) - 1))
    radii = np.asarray(kt.LOOK_RADII, np.float64)
    fig, axes = _subplots(1, len(BANDS), figsize=(3.9 * len(BANDS), 3.6), constrained_layout=True)
    out: dict = {}
    for c, band in enumerate(BANDS):
        for m, (name, p) in enumerate(prepared.items()):
            s = p["samples"]
            sel = (s["band"] == c) & ~s["blank"]
            sub = {key: (v[sel] if isinstance(v, np.ndarray) else v) for key, v in s.items()}
            bins = kt.brightness_bin(sub["bright"], EDGES)
            look = kt.look_around(sub["log_q"], bins, kt.neighbourhood_max(stamps(sub, p["planes"])["scene"]))
            out.setdefault(band, {})[name] = {_bin_label(b, kind): [x if np.isfinite(x) else None for x in r]
                                              for b, r in look.items()}
            for b, rho in sorted(look.items()):
                axes[c].plot(radii, rho, ls=styles[m], color=colours[b],
                             label=f"{name} · {_bin_label(b, kind)}" if c == 0 else None)
        axes[c].axhline(0, color="#999999", lw=0.6)
        axes[c].set_ylim(-0.3, 1.0)
        axes[c].set_title(band, fontsize=9)
        axes[c].set_xlabel("radius r (HR px)", fontsize=8)
        axes[c].tick_params(labelsize=7)
    axes[0].set_ylabel("Spearman ρ, knee vs brightest within r", fontsize=7)
    if any(ax.get_legend_handles_labels()[0] for ax in axes):     # a tiny run may draw no curve
        fig.legend(fontsize=6, loc="outside right center")
    fig.suptitle(f"{' vs '.join(prepared)} — {kind}: how far around a pixel the knee choice looks "
                 f"({' / '.join(f'{st} {n}' for st, n in zip(styles, prepared, strict=False))})", fontsize=9)
    fig.savefig(path, dpi=120)
    return out


def compare_pairs(path: Path, prepared: dict, band: int, kind: str, view_path: Path | None = None) -> dict:
    """The same pixels in both models: within each brightness decade, the
    pixels B reads through a much LOWER / about the SAME / much HIGHER knee
    than A, as median stamps (SR of A, SR of B, scene)."""
    (a_name, a), (b_name, b) = list(prepared.items())[:2]
    ka_, kb_ = _keyed(a["samples"], kind), _keyed(b["samples"], kind)
    keys = [k for k in ka_ if k[1] == band and k in kb_]
    if not keys:
        return {}
    ia = np.array([ka_[k] for k in keys])
    ib = np.array([kb_[k] for k in keys])
    sa = a["samples"]
    sub = {key: (v[ia] if isinstance(v, np.ndarray) else v) for key, v in sa.items()}
    delta = b["samples"]["log_q"][ib] - sa["log_q"][ia]
    bins, groups, stats = kt.residual_groups(delta, sub["bright"], sub["weight"], EDGES)
    st_a = stamps(sub, a["planes"])
    sub_b = {key: (v[ib] if isinstance(v, np.ndarray) else v) for key, v in b["samples"].items()}
    st_b = stamps(sub_b, b["planes"])
    labels = ("B lower", "same", "B higher")
    rows = [i for i in range(len(EDGES) - 1)
            if all(((bins == i) & (groups == g)).sum() >= MIN_GROUP for g in range(3))]
    summary = {"n": len(keys), "bins": []}
    for i, s in enumerate(stats):
        summary["bins"].append({"label": _bin_label(i, kind), "n": s["n"], "delta_q20": s["q20"],
                                "delta_median": s["median"], "delta_q80": s["q80"]})
    if not rows:
        if view_path is not None:                           # the console says so instead of hiding the band
            atomic_files.write_json(view_path, {"side": STAMP, "models": [a_name, b_name], "n": len(keys),
                                                "rows": []}, allow_nan=False, separators=(",", ":"))
        return summary
    view_rows = []
    fig, axes = _subplots(len(rows), 9, figsize=(11.7, 1.55 * len(rows) + 0.8), squeeze=False,
                             constrained_layout=True)
    for r, i in enumerate(rows):
        cells = []
        for g in range(3):
            sel = (bins == i) & (groups == g)
            cells.append((sel, kt.stack(st_a["sr"][sel]), kt.stack(st_b["sr"][sel]),
                          kt.stack(st_a["ref"][sel])))
        norm = _norm([x for c in cells for x in c[1:]])
        params = _norm_params([x for c in cells for x in c[1:]])
        view_rows.append({"bin": int(i), "label": _bin_label(i, kind), "norm": params, "cells": {
            name: {"n": int(sel.sum()), "delta": _num(np.median(delta[sel])), "sr_a": _encode(sr_a, params),
                   "sr_b": _encode(sr_b, params), "ref": _encode(ref, params)}
            for name, (sel, sr_a, sr_b, ref) in zip(("lower", "same", "higher"), cells, strict=True)}})
        for g, (sel, sr_a, sr_b, ref) in enumerate(cells):
            head = f"{_bin_label(i, kind)}\n" if g == 0 else "\n"
            _show(axes[r, 3 * g], sr_a, norm, f"{head}{labels[g]} · n={int(sel.sum())}\n{a_name}")
            _show(axes[r, 3 * g + 1], sr_b, norm,
                  f"Δ={float(np.median(delta[sel])):+.2f} dex\n{b_name}")
            _show(axes[r, 3 * g + 2], ref, norm, "\n" + ("truth" if kind == "synthetic" else "LR 3×3"))
    fig.suptitle(f"{kind} {BANDS[band]}: the same pixels, read by {b_name} through a lower / the same / a "
                 f"higher knee than by {a_name}\n(Δ = log q̄_B − log q̄_A; median stamps in σ, "
                 "brighter side up)",
                 fontsize=8)
    fig.savefig(path, dpi=120)
    if view_path is not None:
        atomic_files.write_json(view_path, {"side": STAMP, "models": [a_name, b_name], "n": len(keys),
                                            "rows": view_rows}, allow_nan=False, separators=(",", ":"))
    return summary


# --------------------------------------------------------------------------- #
# browser-native data: what the console draws itself
# --------------------------------------------------------------------------- #

#: Brightness axis of the knee map's density (log₁₀ σ).
HEAT_X = np.linspace(-3.0, 5.0, 49)
#: Brightness axis of a knee type's histogram (log₁₀ σ).
TYPE_HIST_X = np.linspace(-3.0, 5.0, 33)
#: Radii (HR px) of a stamp's radial occupancy profile.
RADIAL_PX = np.arange(0, STAMP // 2 + 1)


def _num(v: Any, digits: int = 4) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if np.isfinite(f) else None


def _nums(values: Any, digits: int = 4) -> list:
    return [_num(v, digits) for v in np.asarray(values, np.float64).reshape(-1)]


def _bin_centres_log() -> list[float]:
    centres = 0.5 * (np.log10(EDGES[:-1]) + np.log10(EDGES[1:]))
    centres[0] = -2.5
    return [float(c) for c in centres]


def _norm_params(images) -> list[float]:
    """``[lo, hi, linear width]`` of one row of stamps: the asinh scale the
    PNGs use, shared by every stamp of the row."""
    finite = np.concatenate([np.asarray(i)[np.isfinite(i)] for i in images] or [np.zeros(1)])
    finite = finite if finite.size else np.zeros(1)
    lo, hi = np.percentile(finite, [0.5, 99.9])
    return [round(min(float(lo), -0.3), 4), round(max(float(hi), 1.0), 4), 0.3]


def _encode(image: np.ndarray, norm: Sequence[float]) -> str:
    """A stamp as base64 bytes (one per pixel, row 0 = top: the scene's
    brighter side, +row, is up as in the PNGs), asinh-scaled by ``norm``."""
    lo, hi, lw = norm
    a = np.asarray(image, np.float64)
    with np.errstate(all="ignore"):
        f0, f1 = np.arcsinh(lo / lw), np.arcsinh(hi / lw)
        t = (np.arcsinh(a / lw) - f0) / max(f1 - f0, 1e-9)
    t = np.where(np.isfinite(t), np.clip(t, 0.0, 1.0), 0.0)
    return base64.b64encode(np.round(t * 255).astype(np.uint8)[::-1].tobytes()).decode("ascii")


def _encode_occ(image: np.ndarray) -> str:
    """An occupancy map (0–1) as base64 bytes, row 0 = top."""
    t = np.where(np.isfinite(image), np.clip(image, 0.0, 1.0), 0.0)
    return base64.b64encode(np.round(t * 255).astype(np.uint8)[::-1].tobytes()).decode("ascii")


def _radial(image: np.ndarray) -> list:
    """The mean of a map at each whole radius from the centre: alignment-free
    (a dihedral move keeps every pixel's radius)."""
    a = np.asarray(image, np.float64)
    h, w = a.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.rint(np.hypot(yy - h // 2, xx - w // 2)).astype(int)
    out = []
    for radius in RADIAL_PX:
        ring = a[(r == radius) & np.isfinite(a)]
        out.append(_num(ring.mean(), 4) if ring.size else None)
    return out


def _cell(stack: Mapping[str, Any], norm: Sequence[float]) -> dict:
    return {"sr": _encode(stack["sr"], norm), "ref": _encode(stack["ref"], norm),
            "occ": _encode_occ(stack["occ"]), "radial": _radial(stack["occ"])}


def analysis_view(analysed: Mapping[str, dict], knees: np.ndarray, kind: str, spec: str) -> dict:
    """One field set of one analysis, as the console draws it (no stamps):
    per band the knee map (density, per-decade quantiles, chain-rule
    baseline), the group measures, the knee types, the type mix, the
    look-around and the chain-rule density."""
    extra = ("star fraction", STAR_CLASSES) if kind == "synthetic" else ("blank fraction", ("blank",))
    bands: dict[str, Any] = {}
    for band, a in analysed.items():
        logk = np.log10(np.asarray(knees, np.float64) / a["sigma"])
        ok = np.isfinite(a["log_q"])
        w = a["weight"]
        x = np.log10(np.clip(a["bright"], 10 ** -2.95, 1e5))
        y_edges = np.linspace(logk[0] - 0.1, logk[-1] + 0.1, 37)
        heat, _, _ = np.histogram2d(x[ok], a["log_q"][ok], bins=[HEAT_X, y_edges], weights=w[ok])
        curve = [{"q20": _num(st["q20"]), "median": _num(st["median"]), "q80": _num(st["q80"]),
                  "mech": _num(a["mech_bins"][i]), "n": int(st["n"])} if st["n"] >= MIN_GROUP else None
                 for i, st in enumerate(a["bins"])]
        features: dict[str, Any] = {}
        for variant in ("raw", "learned"):
            per_group: dict[str, Any] = {}
            for name in kt.GROUPS:
                cols: dict[str, list] = {key: [] for key, _, _ in FEATURES}
                cols["extra"] = []
                for row in a["variants"][variant]["summary_bins"]:
                    g = row["groups"].get(name)
                    for key, _, _ in FEATURES:
                        cols[key].append(_num(g.get(key)) if g else None)
                    cols["extra"].append(_num(sum(g["classes"].get(c, 0.0) for c in extra[1])) if g else None)
                per_group[name] = cols
            features[variant] = per_group
        types = []
        n_types = len(a["type_stacks"])
        bins_of = a["bin"]
        for j, t in enumerate(a["type_stacks"]):
            hist, _ = np.histogram(np.log10(np.clip(t["bright"], 10 ** -2.95, 1e5)), bins=TYPE_HIST_X,
                                   weights=t["weights"])
            total = hist.sum()
            types.append({"index": j, "prevalence": _num(t["prevalence"], 5), "n": int(t["n"]),
                          "mean_share": _nums(t["mean_share"]), "log_q": _num(t["log_q"]),
                          "bright_median": _num(t["bright_median"]), "dist": _num(t["dist"], 1),
                          "classes": {k: _num(v) for k, v in list(t["classes"].items())[:3]},
                          "hist": _nums(hist / total if total > 0 else hist)})
        mix = []
        for b in range(len(EDGES) - 1):
            sel = (bins_of == b) & (a["types"] >= 0)
            total = a["weight"][sel].sum()
            counts = np.bincount(a["types"][sel], weights=a["weight"][sel], minlength=n_types)[:n_types]
            mix.append(_nums(counts / total) if total > 0 else None)
        okm = ok & np.isfinite(a["log_q_mech"])
        m_edges = np.linspace(logk[0] - 0.1, logk[-1] + 0.1, 37)
        mech_heat, _, _ = np.histogram2d(a["log_q_mech"][okm], a["log_q"][okm], bins=[m_edges, m_edges],
                                         weights=w[okm])
        gap = kt.weighted_quantile(a["log_q"][okm] - a["log_q_mech"][okm], w[okm], (0.2, 0.5, 0.8))
        bands[band] = {
            "sigma": _num(a["sigma"]), "blank_fraction": _num(a["blank_fraction"], 5), "n": int(ok.sum()),
            "logk": _nums(logk), "heat": {"x_edges": _nums(HEAT_X), "y_edges": _nums(y_edges),
                                          "z": [[_num(v, 3) for v in row] for row in heat]},
            "curve": curve, "features": features, "types": types, "type_mix": mix,
            "look": {v: {str(b): _nums(r) for b, r in a["look"][v].items()} for v in a["look"]},
            "mech": {"edges": _nums(m_edges), "z": [[_num(v, 3) for v in row] for row in mech_heat],
                     "gap": _nums(gap)},
        }
    return {"kind": kind, "model": spec, "knees": [float(q) for q in knees],
            "edges_sigma": _nums(EDGES), "centres_log": _bin_centres_log(),
            "bin_labels": [_bin_label(i, kind) for i in range(len(EDGES) - 1)],
            "radii": [float(r) for r in kt.LOOK_RADII], "type_hist_x": _nums(TYPE_HIST_X),
            "features": [{"key": key, "label": label, "log": bool(log)} for key, label, log in FEATURES]
            + [{"key": "extra", "label": extra[0], "log": False}],
            "groups": list(kt.GROUPS), "bands": bands}


def stamps_view(a: Mapping[str, Any], kind: str) -> dict:
    """One band's median stamps: the groups (by the knee and by the
    preference) and the knee types, each row on its own asinh scale."""
    out: dict[str, Any] = {"side": STAMP, "radial_px": [int(r) for r in RADIAL_PX], "groups": {}, "types": []}
    for variant in ("raw", "learned"):
        rows = []
        for i, cells in sorted(a["variants"][variant]["group_stacks"].items()):
            if len(cells) != len(kt.GROUPS):
                continue
            norm = _norm_params([c["sr"] for c in cells.values()] + [c["ref"] for c in cells.values()])
            rows.append({"bin": int(i), "label": _bin_label(i, kind), "norm": norm, "cells": {
                name: _cell(c, norm) | {"n": int(c["n"]), "log_q": _num(c["log_q"]),
                                        "learned": _num(c["learned"]), "dist": _num(c["dist"], 1)}
                for name, c in cells.items()}})
        out["groups"][variant] = rows
    for t in a["type_stacks"]:
        norm = _norm_params([t["sr"], t["ref"]])
        out["types"].append(_cell(t, norm) | {"norm": norm})
    return out


def write_analysis_views(out: Path, analysed: Mapping[str, dict], knees: np.ndarray, kind: str,
                         spec: str) -> None:
    """``view_<kind>.json`` and ``stamps_<kind>_<band>.json`` of an analysis."""
    atomic_files.write_json(out / f"view_{kind}.json", analysis_view(analysed, knees, kind, spec),
                            allow_nan=False, separators=(",", ":"))
    for band, a in analysed.items():
        atomic_files.write_json(out / f"stamps_{kind}_{band}.json", stamps_view(a, kind),
                                allow_nan=False, separators=(",", ":"))


def curves_view(prepared: Mapping[str, dict]) -> dict:
    """Per band and model: the effective knee's and the preference's
    per-decade quantiles and the chain-rule baseline's median."""
    bands: dict[str, Any] = {}
    for c, band in enumerate(BANDS):
        per_model = {}
        for name, p in prepared.items():
            s = p["samples"]
            sel = (s["band"] == c) & ~s["blank"]
            if not sel.any():
                continue
            bins = kt.brightness_bin(s["bright"][sel], EDGES)
            w = s["weight"][sel]
            eff = _bin_medians(s["log_q"][sel], bins, w)
            pref = _bin_medians(s["log_q"][sel] - s["log_q_mech"][sel], bins, w)
            mech = _bin_medians(s["log_q_mech"][sel], bins, w)
            per_model[name] = {"knee": [None if q is None else _nums(q) for q in eff],
                               "preference": [None if q is None else _nums(q) for q in pref],
                               "mech": [None if q is None else _num(q[1]) for q in mech]}
        bands[band] = per_model
    return bands


def _share_diff(a: Mapping[str, Any], b: Mapping[str, Any], c: int) -> tuple[np.ndarray, np.ndarray]:
    """``(knees, B − A)``: the mean own-band local knee share per knee (the
    union of both knee sets) × brightness decade; NaN for a thin decade."""
    knees = np.unique(np.round(np.concatenate([a["knees"], b["knees"]]), 12))
    n_bins = len(EDGES) - 1
    means = []
    for m in (a, b):
        s = m["samples"]
        sel = (s["band"] == c) & ~s["blank"]
        bins = kt.brightness_bin(s["bright"][sel], EDGES)
        cols = np.searchsorted(knees, np.round(m["knees"], 12))
        block = np.full((len(knees), n_bins), np.nan)
        for i in range(n_bins):
            rows = bins == i
            if rows.sum() >= MIN_GROUP:
                block[:, i] = 0.0
                block[cols, i] = np.average(s["share"][sel][rows], axis=0, weights=s["weight"][sel][rows])
        means.append(block)
    return knees, means[1] - means[0]


def compare_view(prepared: Mapping[str, dict], kind: str, look: Mapping[str, Any]) -> dict:
    """A comparison's (or an overview's) field set: the curves, the share
    difference of the first two models and their look-around."""
    names = list(prepared)
    view: dict[str, Any] = {"kind": kind, "models": names, "centres_log": _bin_centres_log(),
                            "bin_labels": [_bin_label(i, kind) for i in range(len(EDGES) - 1)],
                            "radii": [float(r) for r in kt.LOOK_RADII], "curves": curves_view(prepared)}
    if len(names) >= 2 and "knees" in prepared[names[0]]:
        a, b = prepared[names[0]], prepared[names[1]]
        diff = {}
        for c, band in enumerate(BANDS):
            knees, d = _share_diff(a, b, c)
            diff[band] = {"knees": [float(q) for q in knees], "matrix": [_nums(row) for row in d]}
        view["share_diff"] = diff
    if look:
        view["look"] = look
    return view


def model_name(spec: str) -> str:
    """``universal:member_249`` → ``member_249``."""
    return spec.split(":")[-1]


def compare_folder(a: str, b: str, tag: str = "") -> str:
    """The comparison's folder name: ``compare_<A>_vs_<B>[_<tag>]``, every
    part path-safe (a milestone's ``member_257@200000`` → ``member_257__200000``)."""
    safe_tag = runs.slug(tag) if tag else ""
    return f"compare_{runs.slug(a)}_vs_{runs.slug(b)}" + (f"_{safe_tag}" if safe_tag else "")


def compare_models(a: tuple[str, str], b: tuple[str, str], *, tag: str = "",
                   progress: Progress = _quiet) -> Path:
    """Compare model ``b`` against model ``a`` (each ``(run id, spec)``) on
    the pixels both runs sampled (one seed draws the same uniform pixels of
    a field in every run since record version 4): the knee curves, the knee
    share difference, the look-around and, per band, the pixels ``b`` reads
    through a lower / the same / a higher knee. Returns the folder."""
    prepared: dict[str, dict] = {}
    for run_id, spec in (a, b):
        record = runs.get_run(run_id)
        if spec not in record.get("models", {}):
            raise KeyError(f"{spec} is not in run {run_id}")
        entry = record["models"][spec]
        name = model_name(spec)
        name = name if name not in prepared else f"{name}#{run_id[-6:]}"
        prepared[name] = {"run_id": run_id, "spec": spec, "entry": entry}
    names = list(prepared)
    out = report_root() / compare_folder(*names, tag)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"models": {n: p["spec"] for n, p in prepared.items()},
                     "runs": {n: p["run_id"] for n, p in prepared.items()},
                     "created": datetime.now(UTC).isoformat(), "tag": tag}
    for kind in ("synthetic", "real"):
        progress(f"compare {' vs '.join(names)}: {kind}")
        ready = {}
        for name, p in prepared.items():
            got = prepare(p["run_id"], p["spec"], p["entry"], kind)
            if got is not None:
                ready[name] = {"samples": got[0], "planes": got[1],
                               "knees": np.asarray(p["entry"]["knees"], np.float64)}
        if len(ready) < 2:
            continue
        compare_curves(out / f"compare_knee_{kind}.png", ready, kind)
        compare_share_diff(out / f"compare_share_diff_{kind}.png", ready, kind)
        summary[f"look_around_{kind}"] = compare_look_around(out / f"compare_look_around_{kind}.png",
                                                             ready, kind)
        summary[kind] = {BANDS[i]: compare_pairs(out / f"compare_pairs_{kind}_{BANDS[i]}.png", ready, i, kind,
                                                 out / f"pairs_{kind}_{BANDS[i]}.json")
                         for i in range(len(BANDS))}
        atomic_files.write_json(out / f"view_{kind}.json",
                                compare_view(ready, kind, summary[f"look_around_{kind}"]),
                                allow_nan=False, separators=(",", ":"))
    atomic_files.write_json(out / "summary.json", summary, allow_nan=False, default=float)
    return out


def overview(models: Sequence[tuple[str, str]], tag: str, *, progress: Progress = _quiet) -> Path:
    """Several models' knee curves on one figure (``models``: ``(run id,
    spec)``, up to eight): ``pixel_types/overview_<tag>/``, one PNG per
    field set. Not listed by the console (a comparison is two models)."""
    if not 2 <= len(models) <= len(MODEL_COLOURS):
        raise ValueError(f"an overview takes 2–{len(MODEL_COLOURS)} models")
    entries = {}
    for run_id, spec in models:
        record = runs.get_run(run_id)
        name = model_name(spec)
        key = name if name not in entries else f"{name}#{run_id[-6:]}"
        entries[key] = (run_id, spec, record["models"][spec])
    out = report_root() / f"overview_{runs.slug(tag)}"
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"models": {name: spec for name, (_, spec, _) in entries.items()},
                               "runs": {name: run_id for name, (run_id, _, _) in entries.items()},
                               "created": datetime.now(UTC).isoformat(), "tag": tag}
    for kind in ("synthetic", "real"):
        progress(f"overview {tag}: {kind}")
        ready = {}
        for name, (run_id, spec, entry) in entries.items():
            got = prepare(run_id, spec, entry, kind)
            if got is not None:
                ready[name] = {"samples": got[0], "planes": got[1]}
        if len(ready) >= 2:
            compare_curves(out / f"overview_knee_{kind}.png", ready, kind)
            atomic_files.write_json(out / f"view_{kind}.json", compare_view(ready, kind, {}),
                                    allow_nan=False, separators=(",", ":"))
            summary[kind] = True
    atomic_files.write_json(out / "summary.json", summary, allow_nan=False, default=float)
    return out


def analyse_run(run_id: str, specs: Sequence[str] | None = None, *, k: int = DEFAULT_TYPES, seed: int = 0,
                progress: Progress = _quiet) -> list[Path]:
    """Analyse every model of a run (or ``specs``) into
    ``pixel_types/<run id>/<model slug>/``; returns the folders."""
    record = runs.get_run(run_id)
    folders = []
    for spec, entry in record["models"].items():
        if specs and spec not in specs:
            continue
        out = report_root() / run_id / runs.slug(spec)
        out.mkdir(parents=True, exist_ok=True)
        summary = {"run_id": run_id, "model": spec, "knees": entry["knees"],
                   "params": record["params"], "record_version": record.get("version"),
                   "created": datetime.now(UTC).isoformat(), "stamp_side_hr": STAMP,
                   "units": "LR noise σ per pixel", "group_quantiles": list(kt.GROUP_QUANTILES),
                   "edges_sigma": [float(v) for v in EDGES]}
        name = model_name(spec)
        for kind in ("synthetic", "real"):
            progress(f"{name}: {kind}")
            summary[kind] = analyse_kind(run_id, spec, entry, kind, out, k, seed,
                                         lambda label, name=name: progress(f"{name}: {label}"))
        atomic_files.write_json(out / "summary.json", summary, allow_nan=False, default=float)
        folders.append(out)
    return folders


# --------------------------------------------------------------------------- #
# the console: listing, files, jobs
# --------------------------------------------------------------------------- #

_RUN_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")
_SAFE = re.compile(r"^[A-Za-z0-9._#-]+$")
_FILE = re.compile(r"^[A-Za-z0-9._-]+\.(png|json)$")


def report_root() -> Path:
    """``<knee_attribution root>/pixel_types``."""
    return runs.root() / "pixel_types"


def _files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.is_file() and _FILE.match(p.name))


def _read_summary(folder: Path) -> dict:
    try:
        return json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _created(folder: Path, summary: Mapping[str, Any]) -> str:
    if summary.get("created"):
        return str(summary["created"])
    try:
        return datetime.fromtimestamp((folder / "summary.json").stat().st_mtime, UTC).isoformat()
    except OSError:
        return ""


def list_reports() -> dict[str, list[dict[str, Any]]]:
    """Every finished analysis, comparison and overview (a folder with its
    ``summary.json``), newest first: ``{analyses: [{folder, run_id, model,
    created, kinds, files, params}], comparisons: [{folder, models, runs,
    created, kinds, files, tag}], overviews: [the same as a comparison]}``."""
    root = report_root()
    analyses, comparisons, overviews = [], [], []
    if root.is_dir():
        for top in sorted(root.iterdir()):
            if not top.is_dir():
                continue
            if top.name.startswith(("compare_", "overview_")) and (top / "summary.json").is_file():
                s = _read_summary(top)
                (comparisons if top.name.startswith("compare_") else overviews).append({
                    "folder": top.name, "models": s.get("models") or {}, "runs": s.get("runs") or {},
                    "created": _created(top, s), "tag": s.get("tag") or "",
                    "kinds": [k for k in ("synthetic", "real") if s.get(k)], "files": _files(top)})
            elif _RUN_ID.match(top.name):
                for sub in sorted(top.iterdir()):
                    if not (sub.is_dir() and (sub / "summary.json").is_file()):
                        continue
                    s = _read_summary(sub)
                    analyses.append({
                        "folder": f"{top.name}/{sub.name}", "run_id": top.name,
                        "model": s.get("model") or sub.name, "created": _created(sub, s),
                        "kinds": [k for k in ("synthetic", "real") if s.get(k)],
                        "files": _files(sub), "params": s.get("params") or {}})
    analyses.sort(key=lambda r: r["created"], reverse=True)
    comparisons.sort(key=lambda r: r["created"], reverse=True)
    overviews.sort(key=lambda r: r["created"], reverse=True)
    return {"analyses": analyses, "comparisons": comparisons, "overviews": overviews}


def report_file(folder: str, name: str) -> Path:
    """The path of one report file (``folder`` = ``<run id>/<model slug>``
    or ``compare_…``; ``name`` a ``.png``/``.json``); :class:`KeyError`
    for anything else or missing."""
    parts = str(folder).split("/")
    ok = ((len(parts) == 2 and _RUN_ID.match(parts[0]) and _SAFE.match(parts[1]))
          or (len(parts) == 1 and parts[0].startswith(("compare_", "overview_")) and _SAFE.match(parts[0])))
    if not ok or not _FILE.match(str(name)):
        raise KeyError("no such pixel-types file")
    root = report_root().resolve()
    path = (root / Path(*parts) / name).resolve()
    if root not in path.parents or not path.is_file():
        raise KeyError("no such pixel-types file")
    return path


def _finished_run(run_id: str) -> dict[str, Any]:
    try:
        record = runs.get_run(runs.check_run_id(run_id))
    except KeyError as exc:
        raise ValueError(f"unknown knee-attribution run {run_id!r}") from exc
    if record.get("status") == "running":
        raise ValueError(f"run {run_id} is still running")
    if not record.get("models"):
        raise ValueError(f"run {run_id} has no model results")
    return record


def parse_analyse_request(form: Mapping[str, Any]) -> dict[str, Any]:
    """``run_id`` (a finished run) and optional ``models`` (a comma list of
    the run's specs; default: every model) → job parameters."""
    run_id = str(form.get("run_id", "") or "").strip()
    record = _finished_run(run_id)
    wanted = [m.strip() for m in str(form.get("models", "") or "").split(",") if m.strip()]
    for spec in wanted:
        if spec not in record["models"]:
            raise ValueError(f"{spec} is not in run {run_id}")
    return {"run_id": run_id, "models": wanted or list(record["models"])}


def _model_ref(raw: Any, key: str) -> tuple[str, str]:
    run_id, _, spec = str(raw or "").strip().partition("|")
    if not run_id or not spec:
        raise ValueError(f"{key} must be '<run id>|<model spec>'")
    record = _finished_run(run_id)
    if spec not in record["models"]:
        raise ValueError(f"{spec} is not in run {run_id}")
    return run_id, spec


def parse_compare_request(form: Mapping[str, Any]) -> dict[str, Any]:
    """``a`` and ``b`` (``<run id>|<model spec>``: two models of finished
    runs, or one model in two runs) and an optional ``tag`` → job
    parameters. Runs pair their pixels only under one seed: two runs with
    different seeds are refused."""
    a = _model_ref(form.get("a", ""), "a")
    b = _model_ref(form.get("b", ""), "b")
    if a == b:
        raise ValueError("pick two different models (or one model in two runs)")
    seeds = {runs.get_run(run_id)["params"].get("seed") for run_id in {a[0], b[0]}}
    if len(seeds) > 1:
        raise ValueError("the two runs used different seeds, so they sampled different pixels")
    tag = str(form.get("tag", "") or "").strip()
    if tag and not _SAFE.match(tag):
        raise ValueError("tag: letters, digits, '.', '_', '#' and '-' only")
    return {"a": a, "b": b, "tag": tag}


def _ticker(cap, total: int) -> Progress:
    done = [0]

    def step(label: str) -> None:
        cap.check_cancelled()
        cap.tick(min(done[0], total), total, label)
        done[0] += 1
    return step


def job_analyse(cap, *, run_id: str, models: Sequence[str]) -> dict[str, Any]:
    """The analysis job (about two minutes a model): ``{run_id, folders}``."""
    total = max(1, len(models) * 2 * (1 + len(BANDS)))     # per model and kind, then per band
    folders = analyse_run(run_id, models, progress=_ticker(cap, total))
    cap.tick(total, total, "done")
    return {"run_id": run_id, "folders": [f"{run_id}/{p.name}" for p in folders]}


def job_compare(cap, *, a: tuple[str, str], b: tuple[str, str], tag: str = "") -> dict[str, Any]:
    """The comparison job: ``{folder}``."""
    out = compare_models(a, b, tag=tag, progress=_ticker(cap, 2))
    cap.tick(2, 2, "done")
    return {"folder": out.name}
