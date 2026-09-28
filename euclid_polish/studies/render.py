"""Publication figures of a study, in the plate style, with their CSVs.

Every chart is built from one table (:func:`table` → ``(columns, rows)``):
the figure draws exactly those rows and the CSV export is exactly those rows,
so an exported number is always the plotted number. Charts (spec
"Figures › Studies"):

* ``knee`` — PSNR vs knee per band, one curve per member (or per group:
  median with its p16–p84 band across members), plus the mean and the gate;
* ``integrated`` — knee-integrated PSNR by loss family × training knee, one
  dot per member (seeds visible), mean and gate as reference lines;
* ``paired`` — mean Δ integrated PSNR of each member / group against a
  reference (``mean``, ``gate``, ``best`` member, a member label or
  ``group:<name>``) with the 95 % paired bootstrap interval over fields;
* ``gate`` — the production gate's held-out weight per member and per family
  (``source`` = all pixels, source pixels or a brightness bin);
* ``training`` — validation PSNR (joint or per band) or loss vs step;
* ``real`` — real-tile metrics per model from a frozen Sky › Compare run.

A chart whose data the study lacks raises :class:`StudyError` (404) naming
what is missing — nothing is invented.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from euclid_polish.studies import stats
from euclid_polish.studies.store import StudyError, StudyStore
from euclid_polish.visualization.presentation_style import (
    LEGEND_SIZE,
    NOTE_SIZE,
    PANEL_TITLE_SIZE,
    TICK_LABEL_SIZE,
    apply_presentation_figure,
    presentation_rc,
)
from euclid_polish.web.helpers.publication_figures import GRID, INK, MUTED, PAPER

CHARTS = ("knee", "integrated", "paired", "gate", "training", "real")
FORMATS = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}
#: Recipe fields a study can group / colour members by.
GROUP_FIELDS = stats.GROUP_FIELDS
TRAINING_METRICS = ("psnr", "VIS", "Y_E", "J_E", "H_E", "loss")
REAL_METRICS = (("hole_pct", "hole %"), ("pct_R_lt_0p8", "% peaks R < 0.8"),
                ("median_R", "median enclosed-flux R"), ("flux_ratio", "flux ratio ΣSR/ΣLR"))
PALETTE = ("#1267d6", "#d95f02", "#168f65", "#7a3db8", "#c2185b", "#8c6d1f", "#0097a7",
           "#5f6b7c", "#e6a100", "#3949ab")
BAND_COLORS = {"VIS": "#1267d6", "Y_E": "#168f65", "J_E": "#d95f02", "H_E": "#c2185b"}
MEAN_COLOR = "#172033"
GATE_COLOR = "#e6a100"


@dataclass
class Selection:
    """Which members and how a chart groups them (all members by default)."""

    members: list[str] | None = None
    group: str | None = None
    reference: str = "mean"
    source: str = "all"
    metric: str = "psnr"
    experiment: str | None = None
    seed: int = 0
    resamples: int = stats.BOOTSTRAP_RESAMPLES


_LOADED: OrderedDict[tuple[str, str, str], StudyData] = OrderedDict()
_LOADED_LOCK = threading.Lock()
_LOADED_MAX = 8


@dataclass
class StudyData:
    """Everything a study's charts read (loaded once, all in memory)."""

    manifest: dict[str, Any]
    knee: dict[str, Any]
    curves: list[dict[str, Any]]
    gate: dict[str, Any]
    real: dict[str, Any]
    selections: list[dict[str, Any]] = field(default_factory=list)
    _integrated: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def load(cls, store: StudyStore, study_id: str) -> StudyData:
        """The study's numbers, memoized per (study, manifest sha256) — a
        complete study never changes; only the selections sidecar is re-read."""
        manifest = store.manifest(study_id)
        if not manifest.get("complete"):
            raise StudyError(409, f"study {study_id} is incomplete — resume or delete it")
        key = (store.root, study_id, store.manifest_sha256(study_id))
        with _LOADED_LOCK:
            cached = _LOADED.get(key)
            if cached is not None:
                _LOADED.move_to_end(key)
        if cached is None:
            cached = cls(manifest=manifest, knee=store.read_json(study_id, "knee_psnr.json"),
                         curves=store.read_json(study_id, "training_curves.json"),
                         gate=store.read_json(study_id, "gate.json"),
                         real=store.read_json(study_id, "real.json"))
            with _LOADED_LOCK:
                _LOADED[key] = cached
                while len(_LOADED) > _LOADED_MAX:
                    _LOADED.popitem(last=False)
        return dataclasses.replace(cached, selections=store.selections(study_id))

    @property
    def members(self) -> list[dict[str, Any]]:
        rows = []
        for member in (self.manifest.get("ensemble") or {}).get("members") or []:
            row = dict(member)
            knees = row.get("asinh_knees")
            row["training_knee"] = (stats.group_key(knees) if knees
                                    else stats.group_key(row.get("asinh_knee")))
            rows.append(row)
        return rows

    @property
    def model_ids(self) -> list[str]:
        return [m["id"] for m in self.knee.get("models") or []]

    def psnr(self) -> np.ndarray:
        """``(models, fields, knees, bands)``."""
        return np.asarray(self.knee["psnr"], np.float64)

    def per_field_integrated(self) -> np.ndarray:
        """``(models, fields, bands)`` knee-integrated PSNR of each field."""
        if self._integrated is None:
            self._integrated = stats.field_integrated(self.psnr(), self.knee["knees"])
        return self._integrated


# ---------------------------------------------------------------------------
# selection helpers
# ---------------------------------------------------------------------------

def _selected(data: StudyData, sel: Selection) -> list[dict[str, Any]]:
    rows = data.members
    if sel.group and sel.group not in GROUP_FIELDS:
        raise StudyError(400, f"cannot group by {sel.group!r} (use one of "
                              f"{', '.join(GROUP_FIELDS)})")
    if sel.members is None:
        return rows
    known = {row["label"]: row for row in rows}
    unknown = [label for label in sel.members if label not in known]
    if unknown:
        raise StudyError(400, "not members of this study: " + ", ".join(unknown))
    wanted = set(sel.members)
    picked = [row for row in rows if row["label"] in wanted]
    if not picked:
        raise StudyError(400, "the selection holds no member")
    return picked


def _groups(rows: Sequence[Mapping[str, Any]], sel: Selection) -> dict[str, list[str]]:
    return stats.groups(rows, sel.group)


def _index(data: StudyData, model_id: str) -> int:
    try:
        return data.model_ids.index(model_id)
    except ValueError as exc:
        raise StudyError(404, f"this study has no curves for {model_id!r}") from exc


def _color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def _knee_table(data: StudyData, sel: Selection) -> tuple[list[str], list[dict]]:
    rows = _selected(data, sel)
    psnr = stats.nan_mean(data.psnr(), axis=1)              # (models, knees, bands)
    knees, bands = data.knee["knees"], data.knee["bands"]
    out = []

    def emit(series: str, kind: str, n: int, curve: np.ndarray,
             lo: np.ndarray | None = None, hi: np.ndarray | None = None) -> None:
        for b, band in enumerate(bands):
            for k, knee in enumerate(knees):
                out.append({"series": series, "kind": kind, "n_members": n, "band": band,
                            "knee_e": float(knee), "psnr": float(curve[k, b]),
                            "lo": None if lo is None else float(lo[k, b]),
                            "hi": None if hi is None else float(hi[k, b])})

    if sel.group:
        for name, labels in _groups(rows, sel).items():
            band = stats.group_band([psnr[_index(data, lb)] for lb in labels])
            emit(name, "group", band["n"], band["median"], band["lo"], band["hi"])
    else:
        for row in rows:
            emit(row["label"], "member", 1, psnr[_index(data, row["label"])])
    for model in data.knee["models"]:
        if model["kind"] in ("mean", "gate"):
            emit(model["id"], model["kind"], len(data.members), psnr[_index(data, model["id"])])
    return ["series", "kind", "n_members", "band", "knee_e", "psnr", "lo", "hi"], out


def _integrated_table(data: StudyData, sel: Selection) -> tuple[list[str], list[dict]]:
    rows = _selected(data, sel)
    integ = stats.nan_mean(data.per_field_integrated(), axis=1)   # (models, bands)
    bands = data.knee["bands"]
    out = []
    for row in rows:
        m = _index(data, row["label"])
        for b, band in enumerate(bands):
            out.append({"series": row["label"], "kind": "member",
                        "loss": stats.group_key(row.get("loss")),
                        "training_knee": row["training_knee"], "seed": row.get("seed"),
                        "group": stats.group_key(row.get(sel.group)) if sel.group else None,
                        "band": band, "integrated_psnr": float(integ[m, b])})
    for model in data.knee["models"]:
        if model["kind"] in ("mean", "gate"):
            m = _index(data, model["id"])
            for b, band in enumerate(bands):
                out.append({"series": model["id"], "kind": model["kind"], "loss": None,
                            "training_knee": None, "seed": None, "group": None, "band": band,
                            "integrated_psnr": float(integ[m, b])})
    return (["series", "kind", "loss", "training_knee", "seed", "group", "band",
             "integrated_psnr"], out)


def _reference(data: StudyData, sel: Selection, rows: Sequence[Mapping[str, Any]]
               ) -> tuple[str, np.ndarray]:
    """``(name, (fields, bands))`` per-field integrated PSNR of the reference."""
    integ = data.per_field_integrated()
    ref = str(sel.reference or "mean")
    if ref == "best":
        means = {row["label"]: float(stats.nan_mean(integ[_index(data, row["label"])]))
                 for row in rows}
        ref = max(means, key=lambda label: means[label])
    if ref.startswith("group:"):
        if not sel.group:
            raise StudyError(400, "a group reference needs group=<recipe field>")
        name = ref.removeprefix("group:")
        labels = _groups(rows, sel).get(name)
        if not labels:
            raise StudyError(400, f"no group {name!r} in the selection")
        return ref, stats.nan_mean([integ[_index(data, lb)] for lb in labels], axis=0)
    return ref, integ[_index(data, ref)]


def _paired_table(data: StudyData, sel: Selection) -> tuple[list[str], list[dict]]:
    rows = _selected(data, sel)
    integ = data.per_field_integrated()
    ref_name, ref_values = _reference(data, sel, rows)
    bands = data.knee["bands"]
    targets = (_groups(rows, sel) if sel.group else {row["label"]: [row["label"]] for row in rows})
    out = []
    for name, labels in targets.items():
        values = stats.nan_mean([integ[_index(data, lb)] for lb in labels], axis=0)  # (F, C)
        for b, band in enumerate(bands):
            if np.array_equal(values[:, b], ref_values[:, b], equal_nan=True):
                paired = np.isfinite(values[:, b]) & np.isfinite(ref_values[:, b])
                result = {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n_fields": int(paired.sum()),
                          "n_resamples": int(sel.resamples), "seed": int(sel.seed)}
            else:
                try:
                    result = stats.paired_bootstrap(values[:, b], ref_values[:, b],
                                                    n=sel.resamples, seed=sel.seed)
                except ValueError as exc:
                    raise StudyError(409, (
                        f"{band}: fewer than 2 fields with both {name} and {ref_name} values "
                        "— no paired interval (drop that member or pick another reference)")
                    ) from exc
            out.append({"target": name, "n_members": len(labels), "reference": ref_name,
                        "band": band, "mean_delta": result["mean"], "lo": result["lo"],
                        "hi": result["hi"], "n_fields": result["n_fields"],
                        "n_resamples": result["n_resamples"], "seed": result["seed"]})
    return (["target", "n_members", "reference", "band", "mean_delta", "lo", "hi", "n_fields",
             "n_resamples", "seed"], out)


def _gate_table(data: StudyData, sel: Selection) -> tuple[list[str], list[dict]]:
    diag = data.gate.get("diagnostic") or {}
    if not diag.get("available"):
        raise StudyError(404, "this study has no gate weight diagnostic (the production gate "
                              "had no diagnostic payload at freeze)")
    labels = [str(v) for v in diag.get("labels") or []]
    bands = [str(v) for v in diag.get("bands") or data.knee["bands"]]
    source = str(sel.source or "all")
    names = [str(v) for v in diag.get("brightness_names") or []]
    if source == "all":
        table = diag.get("usage") or {}
    elif source == "sources":
        table = diag.get("usage_source") or {}
    elif source in names:
        b = names.index(source)
        table = {band: (rows[b] if b < len(rows) else [])
                 for band, rows in (diag.get("usage_by_brightness") or {}).items()}
    else:
        raise StudyError(400, f"gate source must be all, sources or one of {', '.join(names)}")
    rows = _selected(data, sel)
    family_field = sel.group or "loss"
    out, families = [], {}
    uniform = 1.0 / max(1, len(labels))
    for row in rows:
        if row["label"] not in labels:
            continue
        i = labels.index(row["label"])
        family = stats.group_key(row.get(family_field))
        for band in bands:
            values = table.get(band) or []
            if i >= len(values) or values[i] is None:
                continue
            weight = float(values[i])
            out.append({"level": "member", "name": row["label"], "family": family, "band": band,
                        "weight": weight, "uniform": uniform, "source": source})
            families.setdefault((family, band), []).append(weight)
    if not out:
        raise StudyError(404, "the gate diagnostic has no weights for the selected members")
    for (family, band), weights in families.items():
        out.append({"level": "family", "name": family, "family": family, "band": band,
                    "weight": float(np.sum(weights)), "uniform": uniform * len(weights),
                    "source": source})
    return ["level", "name", "family", "band", "weight", "uniform", "source"], out


def _training_table(data: StudyData, sel: Selection) -> tuple[list[str], list[dict]]:
    if sel.metric not in TRAINING_METRICS:
        raise StudyError(400, f"training metric must be one of {', '.join(TRAINING_METRICS)}")
    rows = _selected(data, sel)
    wanted = {row["label"]: row for row in rows}
    out = []
    for entry in data.curves:
        label = entry.get("label")
        if label not in wanted:
            continue
        if sel.metric == "psnr":
            series = entry.get("psnr") or []
        elif sel.metric == "loss":
            series = entry.get("loss_series") or []
        else:
            series = (entry.get("band_psnr") or {}).get(sel.metric) or []
        group = stats.group_key(wanted[label].get(sel.group)) if sel.group else label
        for step, value in series:
            out.append({"member": label, "group": group, "metric": sel.metric,
                        "step": int(step), "value": float(value)})
    if not out:
        raise StudyError(404, "this study has no training curves for the selected members")
    return ["member", "group", "metric", "step", "value"], out


def _real_table(data: StudyData, sel: Selection) -> tuple[list[str], list[dict]]:
    experiments = data.real.get("experiments") or []
    if not experiments:
        raise StudyError(404, "this study has no real-tile metrics: no Sky › Compare run used "
                              "its membership before the freeze")
    if sel.experiment:
        chosen = next((e for e in experiments if e.get("id") == sel.experiment), None)
        if chosen is None:
            raise StudyError(404, f"this study has no experiment {sel.experiment!r}")
    else:
        chosen = experiments[0]
    labels = {row["label"] for row in _selected(data, sel)}
    out = []
    for spec, entry in (chosen.get("specs") or {}).items():
        member = entry.get("member_label")
        if member is not None and member not in labels:
            continue
        summary = entry.get("summary") or {}
        for band, metrics in (summary.get("per_band") or {}).items():
            out.append({"experiment": chosen.get("id"), "spec": spec,
                        "model": member or spec, "band": band,
                        **{key: metrics.get(key) for key, _label in REAL_METRICS},
                        "n_tiles": summary.get("n_tiles")})
    if not out:
        raise StudyError(404, "the frozen experiment has no metrics for the selected models")
    return (["experiment", "spec", "model", "band", *(k for k, _l in REAL_METRICS), "n_tiles"],
            out)


_TABLES = {"knee": _knee_table, "integrated": _integrated_table, "paired": _paired_table,
           "gate": _gate_table, "training": _training_table, "real": _real_table}


def table(chart: str, data: StudyData, sel: Selection) -> tuple[list[str], list[dict]]:
    """``(columns, rows)`` — exactly what :func:`render` draws."""
    if chart not in _TABLES:
        raise StudyError(404, f"unknown chart {chart!r} (one of {', '.join(CHARTS)})")
    return _TABLES[chart](data, sel)


def csv_text(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if row.get(k) is None else
                             f"{row[k]:.6g}" if isinstance(row.get(k), float) else row.get(k))
                         for k in columns})
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def _finish(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=PANEL_TITLE_SIZE, fontweight=700, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=TICK_LABEL_SIZE, width=1.0, length=4.5)


def _by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[Any, list[Mapping[str, Any]]]:
    out: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        out.setdefault(row[key], []).append(row)
    return out


def _band_axes(fig: Figure, bands: Sequence[str]):
    return fig.subplots(1, len(bands), squeeze=False)[0]


def _legend(fig: Figure, handles, labels, ncol: int) -> None:
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=max(1, min(ncol, 8)),
                   frameon=False, fontsize=LEGEND_SIZE, bbox_to_anchor=(0.5, 0.035))


def _note(fig: Figure, text: str) -> None:
    fig.text(0.5, 0.0, text, ha="center", va="bottom", fontsize=NOTE_SIZE, color=MUTED)


def _knee_tick(value: Any) -> str:
    """A training-knee tick: a multi-knee member's long list is shortened."""
    text = str(value)
    if "+" in text:
        return f"{text.count('+') + 1} knees"
    return "default" if text == stats.MISSING else text


def _sparse_ticks(ax, labels: Sequence[str], *, limit: int = 20) -> None:
    step = max(1, int(np.ceil(len(labels) / limit)))
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels(list(labels)[::step], rotation=90)


def _draw_knee(fig: Figure, rows: list[dict], data: StudyData, sel: Selection) -> None:
    bands = data.knee["bands"]
    axes = _band_axes(fig, bands)
    series = list(dict.fromkeys(r["series"] for r in rows if r["kind"] in ("member", "group")))
    colors = {name: _color(i) for i, name in enumerate(series)}
    colors.update(mean=MEAN_COLOR, gate=GATE_COLOR)
    handles: dict[str, Any] = {}
    for ax, band in zip(axes, bands, strict=True):
        for name, items in _by([r for r in rows if r["band"] == band], "series").items():
            x = [r["knee_e"] for r in items]
            y = [r["psnr"] for r in items]
            kind = items[0]["kind"]
            style = {"mean": "--", "gate": "-."}.get(kind, "-")
            (line,) = ax.plot(x, y, style, color=colors[name],
                              linewidth=2.4 if kind in ("mean", "gate", "group") else 1.4,
                              label=name, alpha=0.95 if kind != "member" else 0.8)
            if kind == "group" and items[0]["lo"] is not None:
                ax.fill_between(x, [r["lo"] for r in items], [r["hi"] for r in items],
                                color=colors[name], alpha=0.16, linewidth=0)
            handles.setdefault(name, line)
        ax.set_xscale("log")
        _finish(ax, band, "asinh knee [e⁻]", "PSNR [dB]")
    _legend(fig, list(handles.values()), list(handles), len(handles))


def _knee_order(value: Any) -> tuple[int, float, str]:
    try:
        return (0, float(value), str(value))
    except (TypeError, ValueError):
        return (1, 0.0, str(value))


def _draw_integrated(fig: Figure, rows: list[dict], data: StudyData, sel: Selection) -> None:
    bands = data.knee["bands"]
    axes = _band_axes(fig, bands)
    members = [r for r in rows if r["kind"] == "member"]
    knees = sorted({r["training_knee"] for r in members}, key=_knee_order)
    colour_key = "group" if sel.group else "loss"
    families = list(dict.fromkeys(r[colour_key] for r in members))
    colors = {name: _color(i) for i, name in enumerate(families)}
    handles: dict[str, Any] = {}
    for ax, band in zip(axes, bands, strict=True):
        band_rows = [r for r in members if r["band"] == band]
        for x, knee in enumerate(knees):
            at = [r for r in band_rows if r["training_knee"] == knee]
            offsets = np.linspace(-0.25, 0.25, len(at)) if len(at) > 1 else [0.0]
            for offset, r in zip(offsets, at, strict=True):
                dot = ax.scatter([x + offset], [r["integrated_psnr"]], s=46,
                                 color=colors[r[colour_key]], edgecolor=PAPER, linewidth=0.8,
                                 zorder=3)
                handles.setdefault(str(r[colour_key]), dot)
        for r in rows:
            if r["kind"] in ("mean", "gate") and r["band"] == band:
                line = ax.axhline(r["integrated_psnr"], color=MEAN_COLOR if r["kind"] == "mean"
                                  else GATE_COLOR, linestyle="--" if r["kind"] == "mean"
                                  else "-.", linewidth=1.6)
                handles.setdefault(r["series"], line)
        ax.set_xticks(range(len(knees)))
        ax.set_xticklabels([_knee_tick(k) for k in knees], rotation=45, ha="right")
        _finish(ax, band, "training knee [e⁻]", "integrated PSNR [dB]")
    _legend(fig, list(handles.values()), list(handles), len(handles))


def _draw_paired(fig: Figure, rows: list[dict], data: StudyData, sel: Selection) -> None:
    bands = data.knee["bands"]
    axes = _band_axes(fig, bands)
    targets = list(dict.fromkeys(r["target"] for r in rows))
    for ax, band in zip(axes, bands, strict=True):
        for y, target in enumerate(targets):
            r = next(item for item in rows if item["target"] == target and item["band"] == band)
            ax.errorbar([r["mean_delta"]], [y],
                        xerr=[[r["mean_delta"] - r["lo"]], [r["hi"] - r["mean_delta"]]],
                        fmt="o", color=_color(y), ecolor=_color(y), elinewidth=2.0,
                        capsize=4, markersize=6)
        ax.axvline(0.0, color=MUTED, linewidth=1.0)
        ax.set_yticks(range(len(targets)))
        ax.set_yticklabels(targets if ax is axes[0] else [""] * len(targets))
        ax.invert_yaxis()
        _finish(ax, band, f"Δ integrated PSNR vs {rows[0]['reference']} [dB]", "")
    first = rows[0]
    _note(fig, (f"mean over {first['n_fields']} fields; 95 % paired bootstrap "
                f"({first['n_resamples']} resamples, seed {first['seed']})"))


def _draw_gate(fig: Figure, rows: list[dict], data: StudyData, sel: Selection) -> None:
    bands = list(dict.fromkeys(r["band"] for r in rows))
    grid = fig.subplots(2, len(bands), squeeze=False)
    families = list(dict.fromkeys(r["family"] for r in rows))
    colors = {name: _color(i) for i, name in enumerate(families)}
    handles: dict[str, Any] = {}
    for col, band in enumerate(bands):
        members = sorted((r for r in rows if r["level"] == "member" and r["band"] == band),
                         key=lambda r: (families.index(r["family"]), r["name"]))
        ax = grid[0][col]
        bars = ax.bar(range(len(members)), [r["weight"] for r in members],
                      color=[colors[r["family"]] for r in members])
        for bar, r in zip(bars, members, strict=True):
            handles.setdefault(r["family"], bar)
        if members:
            ax.axhline(members[0]["uniform"], color=MUTED, linestyle=":", linewidth=1.2)
        _sparse_ticks(ax, [r["name"].split("·")[0] for r in members])
        _finish(ax, f"{band} · per member", "member", "mean gate weight")
        fam = [r for r in rows if r["level"] == "family" and r["band"] == band]
        ax = grid[1][col]
        ax.bar(range(len(fam)), [r["weight"] for r in fam],
               color=[colors[r["family"]] for r in fam])
        ax.scatter(range(len(fam)), [r["uniform"] for r in fam], marker="_", s=300,
                   color=MUTED, zorder=3)
        ax.set_xticks(range(len(fam)))
        ax.set_xticklabels([r["name"] for r in fam], rotation=30)
        _finish(ax, f"{band} · per family", sel.group or "loss", "summed weight")
    _legend(fig, list(handles.values()), list(handles), len(handles))


def _draw_training(fig: Figure, rows: list[dict], data: StudyData, sel: Selection) -> None:
    ax = fig.subplots(1, 1)
    groups = list(dict.fromkeys(r["group"] for r in rows))
    colors = {name: _color(i) for i, name in enumerate(groups)}
    handles: dict[str, Any] = {}
    for items in _by(rows, "member").values():
        (line,) = ax.plot([r["step"] for r in items], [r["value"] for r in items],
                          color=colors[items[0]["group"]], linewidth=1.5, alpha=0.85)
        handles.setdefault(items[0]["group"], line)
    label = {"psnr": "validation PSNR (joint) [dB]", "loss": "combined loss"}.get(
        sel.metric, f"validation PSNR {sel.metric} [dB]")
    _finish(ax, "Training curves", "step", label)
    _legend(fig, list(handles.values()), list(handles), len(handles))


def _draw_real(fig: Figure, rows: list[dict], data: StudyData, sel: Selection) -> None:
    axes = fig.subplots(1, len(REAL_METRICS), squeeze=False)[0]
    models = list(dict.fromkeys(r["model"] for r in rows))
    bands = list(dict.fromkeys(r["band"] for r in rows))
    handles: dict[str, Any] = {}
    for ax, (key, label) in zip(axes, REAL_METRICS, strict=True):
        for b, band in enumerate(bands):
            xs, ys = [], []
            for x, model in enumerate(models):
                r = next((i for i in rows if i["model"] == model and i["band"] == band), None)
                if r is not None and r.get(key) is not None:
                    xs.append(x + (b - (len(bands) - 1) / 2) * 0.12)
                    ys.append(float(r[key]))
            dots = ax.scatter(xs, ys, s=36, color=BAND_COLORS.get(band, _color(b)), zorder=3)
            handles.setdefault(band, dots)
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.split("·")[0] for m in models], rotation=60)
        _finish(ax, label, "model", label)
    _note(fig, f"Sky › Compare run {rows[0]['experiment']} "
               f"({rows[0].get('n_tiles') or '?'} tiles)")
    _legend(fig, list(handles.values()), list(handles), len(handles))


_DRAW = {"knee": _draw_knee, "integrated": _draw_integrated, "paired": _draw_paired,
         "gate": _draw_gate, "training": _draw_training, "real": _draw_real}
_SIZES = {"knee": (18.0, 5.6), "integrated": (18.0, 5.8), "paired": (18.0, 5.2),
          "gate": (18.0, 9.0), "training": (10.0, 6.2), "real": (18.0, 5.8)}
_TITLES = {"knee": "PSNR vs asinh knee", "integrated": "Knee-integrated PSNR by loss × knee",
           "paired": "Paired difference over fields", "gate": "Production-gate weight",
           "training": "Training curves", "real": "Real-tile metrics"}


def render(chart: str, data: StudyData, sel: Selection, *, output_format: str = "png",
           dpi: int = 300) -> bytes:
    """One chart of ``data`` as PNG / PDF / SVG bytes."""
    fmt = str(output_format or "png").lower()
    if fmt not in FORMATS:
        raise StudyError(400, "format must be png, pdf or svg")
    if chart not in _DRAW:
        raise StudyError(404, f"unknown chart {chart!r} (one of {', '.join(CHARTS)})")
    dpi = max(120, min(int(dpi), 600))
    _columns, rows = table(chart, data, sel)
    style = {"figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
             "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED}
    with presentation_rc(style):
        fig = Figure(figsize=_SIZES[chart])
        FigureCanvasAgg(fig)
        _DRAW[chart](fig, rows, data, sel)
        manifest = data.manifest
        fig.suptitle(f"{_TITLES[chart]} — {manifest.get('name')}", x=0.01, ha="left")
        fig.text(0.99, 0.985, f"study {manifest.get('id')}", ha="right", va="top",
                 fontsize=NOTE_SIZE, color=MUTED)
        apply_presentation_figure(fig)
        fig.tight_layout(rect=(0, 0.11, 1, 0.94))
        buffer = io.BytesIO()
        fig.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight")
    return buffer.getvalue()


__all__ = [
    "CHARTS",
    "FORMATS",
    "GROUP_FIELDS",
    "Selection",
    "StudyData",
    "csv_text",
    "render",
    "table",
]
