"""Compare spatial-gate variants with the members, the mean and the RBF, and
fit a NAMED gate variant — the importable core of
``scripts/fit_spatial_gate.py`` (``compare`` / ``fit``), so the web console's
jobs run exactly the code the script runs.

Compare scores every method on the cached test member cubes (the natural
group) plus blackout-augmented copies of them (the blackout group), in the
same asinh space the gate is fitted in: per-band PSNR, VIS squared error per
brightness bin, in star halos and (blackout group) in the zeroed holes. It
also scores each combiner's PSNR-vs-knee curve on the natural group (the
knee-integrated PSNR of :mod:`euclid_polish.eval.knee_psnr`, directly
comparable with the Models › Leaderboard) and each gate's member usage.

A variant fitted for a SUBSET of the cube members (older 20/26-member gates,
pruned gates) is applied to exactly its members, picked by label from the
cube stack, instead of being refused.

This module never writes the production artifact: :func:`fit_gate_variant`
saves into the directory it is given, and the callers refuse the production
name.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, maximum_filter

from euclid_polish.config import Config
from euclid_polish.ensemble import default_ensemble_dir, member_fingerprints
from euclid_polish.eval.combiner import (
    COMBINER_MODELS,
    RAW_INCREMENTAL_MINMEANMAX_RBF_KIND,
    load_combiner,
)
from euclid_polish.eval.ensemble_cube_cache import (
    bucket_member_paths,
    manifest_member_labels,
    stale_bucket_members,
)
from euclid_polish.eval.knee_psnr import (
    KNEE_GRID_E,
    integrated_psnr,
    knee_psnr,
    stretched_truth,
)
from euclid_polish.eval.spatial_gate import (
    MIX_LINEAR,
    SpatialGateCombiner,
    band_scales,
    load_spatial_gate,
    restrict_to_available,
    save_spatial_gate,
)
from euclid_polish.eval.spatial_gate_fit import (
    ALL_KNEE_LOSS,
    GateField,
    MemberRunFn,
    build_blackout_fields,
    fit_spatial_gate,
    load_cube_fields,
    split_holdout,
)

BANDS = tuple(Config.HR_TARGET_BAND_NAMES)
BRIGHTNESS_EDGES = (0.02, 0.1, 0.5, 2.0)
BRIGHTNESS_NAMES = ("sky", "faint", "mid", "bright", "core")
PEAK_ASINH = 4.0
HALO_RADII = (3.0, 15.0)
#: Report layout version (1 = the script's pre-helper report).
REPORT_SCHEMA = 2
#: The production gate's artifact directory — never a fit destination here.
PRODUCTION_DIR = COMBINER_MODELS["spatial_gate"].artifact_dir
RBF_KIND = RAW_INCREMENTAL_MINMEANMAX_RBF_KIND

ProgressFn = Callable[[int, int, str], None]
LogFn = Callable[[str], None]


# --------------------------------------------------------------------------- #
# Scoring primitives
# --------------------------------------------------------------------------- #

class Scores:
    """Per-method accumulators over test fields."""

    def __init__(self) -> None:
        self.band_psnr: list[np.ndarray] = []
        self.bins = np.zeros(len(BRIGHTNESS_NAMES))
        self.halo = np.zeros(len(BANDS))
        self.hole = np.zeros(len(BANDS))

    def summary(self, n_bins, n_halo, n_hole) -> dict:
        # An empty group (a compare without blackout fields) has no PSNR:
        # null, never NaN (the report is served as JSON).
        psnr = (np.mean(self.band_psnr, axis=0).tolist() if self.band_psnr
                else [None] * len(BANDS))
        return {"band_psnr": psnr,
                "bin_mse": (self.bins / np.maximum(n_bins, 1)).tolist(),
                "halo_mse": (self.halo / np.maximum(n_halo, 1)).tolist(),
                "hole_mse": (self.hole / np.maximum(n_hole, 1)).tolist()}


def psnr_from_mse(mse: np.ndarray) -> np.ndarray:
    peak = float(Config.PSNR_PEAK_STRETCHED)
    return 10.0 * np.log10(peak * peak / np.maximum(mse, 1e-20))


def halo_mask(truth_vis: np.ndarray) -> np.ndarray:
    """Pixels 3–15 px from a bright (asinh > 4) local VIS peak."""
    peaks = (maximum_filter(truth_vis, size=7) == truth_vis) & (truth_vis > PEAK_ASINH)
    if not peaks.any():
        return np.zeros_like(peaks)
    dist = distance_transform_edt(~peaks)
    return (dist >= HALO_RADII[0]) & (dist <= HALO_RADII[1])


def hole_masks(field_: GateField, source_lr: np.ndarray) -> np.ndarray:
    """(H, W, C) HR mask of pixels whose LR was zeroed by the stamping."""
    new = (field_.lr_e == 0) & (source_lr != 0)
    grown = np.stack([binary_dilation(new[..., c], iterations=2)
                      for c in range(new.shape[-1])], -1)
    h, w = field_.shape
    return np.kron(grown, np.ones((2, 2, 1), bool))[:h, :w]


def member_positions(model_labels: Sequence[str], cube_labels: Sequence[str]
                     ) -> list[int] | None:
    """Positions of ``model_labels`` in the cube stack, or ``None`` when one
    of them was not evaluated into these cubes."""
    where = {str(label): i for i, label in enumerate(cube_labels)}
    out = [where.get(str(label)) for label in model_labels]
    return None if any(i is None for i in out) else [int(i) for i in out]


def resolve_active_members(members: Sequence[str] | None,
                           labels: Sequence[str]) -> list[int] | None:
    """Member numbers (``"170"``, ``"member_170"``, ``"170·psnr"``) → their
    positions in ``labels`` for a pruned gate; ``None``/empty → every member.
    Unknown numbers raise :class:`ValueError`."""
    wanted = [str(m).strip().removeprefix("member_").split("·")[0].lstrip("0") or "0"
              for m in (members or []) if str(m).strip()]
    if not wanted:
        return None
    number = {str(label).split("·")[0].lstrip("0") or "0": i
              for i, label in enumerate(labels)}
    missing = sorted({w for w in wanted if w not in number})
    if missing:
        raise ValueError(f"not members of these cubes: {', '.join(missing)}")
    return sorted({number[w] for w in wanted})


# --------------------------------------------------------------------------- #
# Methods (a combiner on its own member subset)
# --------------------------------------------------------------------------- #

@dataclass
class Method:
    """One combiner to score: ``model.apply_field(members[index], lr=…)``."""

    name: str
    model: Any
    index: list[int]
    kind: str                       # "gate" | "rbf"
    label: str = ""

    @property
    def use_lr(self) -> bool:
        return bool(getattr(self.model, "use_lr", False))

    @property
    def is_gate(self) -> bool:
        return self.kind == "gate"


def load_gate_methods(regime_dir: str, names: Sequence[str],
                      cube_labels: Sequence[str]) -> dict[str, Method]:
    """``gate:<dir>`` methods for the named ``spatial_gate_*`` directories.

    Raises :class:`ValueError` for a name that is not a gate directory, is not
    loadable, or reads a member the cubes lack. Fitted members it does not
    read may be missing (archived since the fit): they are dropped from the
    in-memory gate, which leaves its output unchanged."""
    out: dict[str, Method] = {}
    for name in names:
        name = str(name).strip()
        if not name or os.path.basename(name) != name or not name.startswith("spatial_gate"):
            raise ValueError(f"not a spatial gate directory name: {name!r}")
        gate = load_spatial_gate(os.path.join(regime_dir, name))
        if gate is None:
            raise ValueError(f"no loadable spatial gate at {name}")
        fitted = gate
        gate = restrict_to_available(fitted, [str(v) for v in cube_labels])
        index = None if gate is None else member_positions(gate.member_labels, cube_labels)
        if gate is None or index is None:
            missing = sorted(set(fitted.read_labels) - set(map(str, cube_labels)))
            raise ValueError(f"{name} reads members the test cubes lack: "
                             + ", ".join(missing[:8]))
        out[f"gate:{name}"] = Method(f"gate:{name}", gate, index, "gate", name)
    return out


def load_rbf_method(regime_dir: str, cube_labels: Sequence[str]) -> Method | None:
    """The RBF combiner on its own members, or ``None`` when it is absent or
    reads members the cubes lack."""
    rbf = load_combiner(regime_dir, member_labels=None,
                        artifact_dir=COMBINER_MODELS[RBF_KIND].artifact_dir)
    if rbf is None:
        return None
    index = member_positions(rbf.member_labels, cube_labels)
    return None if index is None else Method("rbf", rbf, index, "rbf", "RBF")


# --------------------------------------------------------------------------- #
# Compare
# --------------------------------------------------------------------------- #

@dataclass
class CompareResult:
    report: dict
    labels: list[str]
    methods: dict[str, Method]
    fields: list[GateField]
    blackouts: list[GateField]
    source_lr: dict[int, np.ndarray]
    figure_candidates: list[tuple[str, GateField, bool]] = field(default_factory=list)


def compare_methods(fields: Sequence[GateField], blackouts: Sequence[GateField],
                    labels: Sequence[str], methods: Mapping[str, Method], *,
                    source_lr: Mapping[int, np.ndarray],
                    cached_rbf: Callable[[int], np.ndarray | None] | None = None,
                    knee: bool = True,
                    progress: ProgressFn | None = None,
                    log: LogFn | None = None) -> tuple[dict, list]:
    """Score ``mean``, every method and every member on both field groups.

    ``cached_rbf(index)`` may return the RBF output the evaluation already
    baked for a natural test field (saves re-applying it). Returns
    ``(report, figure_candidates)``."""
    labels = [str(v) for v in labels]
    scales = band_scales(BANDS).astype(np.float32)
    method_names = ["mean", *methods]
    all_names = method_names + [f"member:{lb}" for lb in labels]
    scores = {group: {m: Scores() for m in all_names} for group in ("natural", "blackout")}
    counts = {group: {"bins": np.zeros(len(BRIGHTNESS_NAMES)), "halo": np.zeros(len(BANDS)),
                      "hole": np.zeros(len(BANDS)), "fields": 0} for group in scores}
    gates = {name: m for name, m in methods.items() if m.is_gate}
    usage = {name: np.zeros((len(m.index), len(BANDS))) for name, m in gates.items()}
    usage_src = {name: np.zeros((len(m.index), len(BANDS))) for name, m in gates.items()}
    n_usage = n_usage_src = 0
    timing: dict[str, list[float]] = {name: [] for name in methods}
    knee_sum: dict[str, np.ndarray | None] = dict.fromkeys(method_names) if knee else {}
    knee_n = 0
    candidates: list[tuple[str, GateField, bool]] = []
    total = len(fields) + len(blackouts)
    done = 0
    for group, group_fields in (("natural", list(fields)), ("blackout", list(blackouts))):
        for f in group_fields:
            members = f.members_e()
            truth = np.arcsinh(f.target_e / scales)
            outputs: dict[str, np.ndarray] = {"mean": members.mean(0)}
            weights: dict[str, np.ndarray] = {}
            for name, m in methods.items():
                lr = f.lr_e if m.use_lr else None
                cached = (cached_rbf(f.index) if (m.kind == "rbf" and group == "natural"
                                                  and cached_rbf is not None) else None)
                if cached is not None:
                    outputs[name] = cached
                    continue
                started = time.time()
                subset = members[m.index]
                outputs[name] = m.model.apply_field(subset, lr=lr) if m.is_gate \
                    else m.model.apply_field(subset)
                timing[name].append(time.time() - started)
                if m.is_gate:
                    weights[name] = m.model.weights_field(subset, lr=lr)
            for i, label in enumerate(labels):
                outputs[f"member:{label}"] = members[i]
            bins = np.digitize(truth[..., 0], BRIGHTNESS_EDGES)
            halo = halo_mask(truth[..., 0])
            holes = hole_masks(f, source_lr[f.index]) if group == "blackout" else None
            cnt = counts[group]
            cnt["fields"] += 1
            cnt["bins"] += np.bincount(bins.ravel(), minlength=len(BRIGHTNESS_NAMES))
            cnt["halo"] += halo.sum()
            if holes is not None:
                cnt["hole"] += holes.reshape(-1, len(BANDS)).sum(0)
            for method, image in outputs.items():
                err2 = (np.arcsinh(np.asarray(image, np.float32) / scales) - truth) ** 2
                s = scores[group][method]
                s.band_psnr.append(psnr_from_mse(err2.mean(axis=(0, 1))))
                s.bins += np.bincount(bins.ravel(), weights=err2[..., 0].ravel(),
                                      minlength=len(BRIGHTNESS_NAMES))
                s.halo += err2[halo].sum(0)
                if holes is not None:
                    s.hole += np.where(holes, err2, 0.0).reshape(-1, len(BANDS)).sum(0)
            if group == "natural":
                source = truth[..., 0] > 0.1
                for name, w in weights.items():
                    usage[name] += w.sum(axis=(0, 1))
                    usage_src[name] += w[source].sum(0)
                n_usage += truth.shape[0] * truth.shape[1]
                n_usage_src += int(source.sum())
                if knee:
                    truth_k = stretched_truth(f.target_e)
                    for name in method_names:
                        curve = knee_psnr(np.asarray(outputs[name], np.float32), f.target_e,
                                          truth_asinh=truth_k)
                        knee_sum[name] = curve if knee_sum[name] is None else knee_sum[name] + curve
                    knee_n += 1
            candidates.append((group, f, bool(halo.any())))
            done += 1
            if progress is not None:
                progress(done, total, f"{group} test field {f.index}")
            if log is not None:
                log(f"  [{group} {done}/{total}] field {f.index}")

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "members": labels,
        "bands": list(BANDS),
        "brightness_names": list(BRIGHTNESS_NAMES),
        "methods": method_names,
        "method_labels": {"mean": "mean", **{n: m.label or n for n, m in methods.items()}},
        "method_members": {n: [labels[i] for i in m.index] for n, m in methods.items()},
        "gates": {n: m.model.fit_meta.get("selected") for n, m in gates.items()},
        "n_fields": {g: int(c["fields"]) for g, c in counts.items()},
        "groups": {},
        "timing_s": {k: float(np.mean(v)) if v else None for k, v in timing.items()},
        "members_needed": {n: len(m.model.needed_member_indices()) for n, m in gates.items()},
        "usage": {},
    }
    for group in scores:
        c = counts[group]
        report["groups"][group] = {m: s.summary(c["bins"], c["halo"], c["hole"])
                                   for m, s in scores[group].items()}
    for name, m in gates.items():
        report["usage"][name] = {
            "labels": [labels[i] for i in m.index],
            "all_pixels": (usage[name] / max(n_usage, 1)).tolist(),
            "source_pixels": (usage_src[name] / max(n_usage_src, 1)).tolist()}
    if knee and knee_n:
        report["knee"] = {
            "knees": list(KNEE_GRID_E), "n_fields": knee_n,
            "methods": {name: {"psnr": np.round(curve / knee_n, 4).tolist(),
                               "integrated": np.round(integrated_psnr(curve / knee_n), 4).tolist()}
                        for name, curve in knee_sum.items() if curve is not None}}
    return report, candidates


def _cube_manifest(cubes_dir: str) -> dict:
    with open(os.path.join(cubes_dir, "viz_index.json")) as handle:
        return json.load(handle)


def require_current_cubes(cubes_dir: str, manifest: Mapping,
                          fingerprints: Mapping[str, str | None], refresh: str) -> None:
    """Refuse a bucket holding member cubes that the members' current
    checkpoints did not make (a member continued since, or a positional
    bucket of unknown provenance), or listing a field without every member's
    cube (a fill stopped half-way): :class:`RuntimeError` naming them and
    ``refresh`` (what re-infers just those members)."""
    name = os.path.basename(cubes_dir.rstrip("/"))
    stale = stale_bucket_members(manifest, fingerprints)
    if stale:
        names = ", ".join(label.split("·")[0] for label in stale[:8])
        more = f" … (+{len(stale) - 8})" if len(stale) > 8 else ""
        raise RuntimeError(f"{name} holds cubes of member(s) {names}{more} that their "
                           f"current checkpoints did not make — {refresh}")
    labels = manifest_member_labels(manifest)
    gaps = [rec for rec in (int(i) for i in manifest.get("indices", []) or [])
            if not all(path is not None and os.path.isfile(path)
                       for path in bucket_member_paths(manifest, cubes_dir, labels, rec))]
    if gaps:
        raise RuntimeError(f"{name} lacks member cubes of {len(gaps)} field(s) (a fill "
                           f"stopped half-way) — {refresh}")


def run_compare(*, regime_dir: str, records_dir: str, gates: Sequence[str],
                runner: MemberRunFn | None,
                blackout_fields: int = 40, seed: int = 0, target_name: str = "hr",
                include_rbf: bool = True, knee: bool = True,
                progress: ProgressFn | None = None, log: LogFn | None = None) -> CompareResult:
    """Load the regime's test cubes (+ blackouts) and score ``gates`` against
    the mean, the RBF and every member. ``runner`` runs the members on a
    stamped LR (only the blackout cubes the cache lacks or holds for another
    checkpoint; the blackout copies are keyed on the test records too). The
    test cubes are refused unless the members' current checkpoints (the
    runner's fingerprints, else those under the ensemble dir) made them."""
    cubes = os.path.join(regime_dir, "cubes")
    cube_manifest = _cube_manifest(cubes)
    fingerprints = getattr(runner, "fingerprints", None)
    if fingerprints is None:
        fingerprints = member_fingerprints(default_ensemble_dir(),
                                           manifest_member_labels(cube_manifest))
    # The blackout copies are re-inferred for a changed member; the natural
    # cubes must be the same checkpoints for the two groups to agree.
    require_current_cubes(cubes, cube_manifest, fingerprints,
                          "evaluate the ensemble first (it re-infers only those members)")
    fwhm = float(cube_manifest["target_psf_fwhm_arcsec"])
    fields, labels = load_cube_fields(cubes, records_dir, "test", target_name=target_name,
                                      target_fwhm_arcsec=fwhm, progress=progress)
    if not fields:
        raise RuntimeError("no cached test fields — evaluate the ensemble first")
    source_lr = {f.index: f.lr_e for f in fields}
    methods: dict[str, Method] = {}
    if include_rbf:
        rbf = load_rbf_method(regime_dir, labels)
        if rbf is not None:
            methods["rbf"] = rbf
    methods.update(load_gate_methods(regime_dir, gates, labels))
    blackouts: list[GateField] = []
    if blackout_fields > 0 and runner is not None:
        blackouts = build_blackout_fields(
            fields, labels, runner, os.path.join(regime_dir, "cubes_blackout"),
            max_fields=blackout_fields, seed=seed + 1,
            source_fingerprint=cube_manifest.get("records_fp"), progress=progress)
    rbf_prefix = COMBINER_MODELS[RBF_KIND].cube_prefix
    rbf_method = methods.get("rbf")

    def cached_rbf(index: int) -> np.ndarray | None:
        if rbf_method is None or rbf_method.index != list(range(len(labels))):
            return None
        path = os.path.join(cubes, f"{rbf_prefix}_{index:05d}.npy")
        return np.load(path) if os.path.isfile(path) else None

    report, candidates = compare_methods(
        fields, blackouts, labels, methods, source_lr=source_lr, cached_rbf=cached_rbf,
        knee=knee, progress=progress, log=log)
    seconds = getattr(runner, "seconds", None)
    report["member_inference_s_per_field"] = float(np.mean(seconds)) if seconds else None
    report["created"] = datetime.now(UTC).isoformat(timespec="seconds")
    report["regime_dir"] = os.path.basename(regime_dir.rstrip("/"))
    report["blackout_seed"] = int(seed) + 1
    return CompareResult(report, labels, methods, fields, blackouts, source_lr, candidates)


def best_member(report: dict, group: str = "natural", band: int = 0) -> str | None:
    """The ``member:<label>`` with the best PSNR in one band of a group
    (None when the group scored no fields)."""
    block = report.get("groups", {}).get(group, {})
    members = {k: v for k, v in block.items() if k.startswith("member:")
               and v["band_psnr"][band] is not None and np.isfinite(v["band_psnr"][band])}
    if not members:
        return None
    return max(members, key=lambda k: members[k]["band_psnr"][band])


def format_report(report: dict) -> str:
    """The script's plain-text comparison table (one block per group)."""
    lines: list[str] = []
    gates = [n for n in report.get("methods", []) if n.startswith("gate:")]
    for group, block in report["groups"].items():
        best = best_member(report, group)
        if best is None:
            continue
        lines.append(f"\n== {group} test fields ==  (best single member by VIS: {best})")
        lines.append(f"{'method':32s} {'VIS':>7s} {'Y':>7s} {'J':>7s} {'H':>7s}   "
                     + " ".join(f"{n:>7s}" for n in BRIGHTNESS_NAMES)
                     + f"  {'halo':>7s}  {'holes':>7s}")
        ref = block[best]
        for row in ["mean", best, "rbf", *gates]:
            if row not in block:
                continue
            s = block[row]
            rel = [s["bin_mse"][i] / max(ref["bin_mse"][i], 1e-30)
                   for i in range(len(BRIGHTNESS_NAMES))]
            halo = s["halo_mse"][0] / max(ref["halo_mse"][0], 1e-30)
            hole = (np.mean(s["hole_mse"]) / max(np.mean(ref["hole_mse"]), 1e-30)
                    if group == "blackout" else float("nan"))
            lines.append(f"{row:32s} " + " ".join(f"{v:7.3f}" for v in s["band_psnr"])
                         + "   " + " ".join(f"{v:7.3f}" for v in rel)
                         + f"  {halo:7.3f}  {hole:7.3f}")
    lines.append("\n(bin/halo/hole columns: VIS squared error relative to the best single member)")
    for name in gates:
        use = report["usage"].get(name) or {}
        src = np.asarray(use.get("source_pixels") or [], np.float64)
        names = use.get("labels") or []
        if src.size:
            order = np.argsort(-src[:, 0])[:6]
            lines.append(f"{name} mean VIS weight on source pixels: "
                         + ", ".join(f"{names[i]} {src[i, 0]:.2f}" for i in order))
    knee_block = report.get("knee") or {}
    for name, entry in (knee_block.get("methods") or {}).items():
        lines.append(f"{name} knee-integrated PSNR (VIS/Y/J/H): "
                     + " ".join(f"{v:.3f}" for v in entry["integrated"]))
    lines.append("timing (s/field): " + str({k: (round(v, 3) if v else v)
                                             for k, v in report["timing_s"].items()})
                 + f" member inference: {report.get('member_inference_s_per_field')}")
    lines.append(f"members each gate needs: {report['members_needed']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Fit-in-progress marker and the promotion guard
# --------------------------------------------------------------------------- #

#: Written into a variant directory while :func:`fit_gate_variant` fits it.
FIT_MARKER = ".fitting.json"


def _pid_alive(pid: int) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False                   # not written by fit_gate_variant
    if pid <= 0:                       # os.kill(-1 or 0) would probe a group
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OverflowError):
        return True                    # exists but not ours: assume live
    return True


def fit_in_progress(directory: str) -> dict | None:
    """The marker of a fit still writing ``directory`` (``{pid, host,
    started}``), or ``None``. A marker whose process is gone is ignored (a
    crashed fit must not block its variant forever). Gate fits run on this
    machine, so liveness is the pid alone: the host name is informational
    (a laptop's DHCP host name changes with the network)."""
    try:
        with open(os.path.join(directory, FIT_MARKER)) as handle:
            marker = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(marker, dict):
        return None
    if not _pid_alive(marker.get("pid", -1)):
        return None
    return marker


def promotion_refusal(directory: str, manifest: Mapping[str, Any] | None) -> str | None:
    """Why the gate variant in ``directory`` must not become production now
    (``None`` when it may): a fit is still writing it, or its fit did not run
    to completion (``fit_meta.complete`` is not true — a stopped fit keeps
    its best checkpoint so far, and a running one rewrites it)."""
    name = os.path.basename(os.path.normpath(directory))
    marker = fit_in_progress(directory)
    if marker is not None:
        return (f"{name} is still being fitted (pid {marker.get('pid')} on "
                f"{marker.get('host')}, since {marker.get('started')}) — wait for the fit "
                f"to finish (if that process is not a gate fit, delete "
                f"{os.path.join(directory, FIT_MARKER)})")
    meta = (manifest or {}).get("fit_meta") or {}
    complete = meta.get("complete") if isinstance(meta, Mapping) else None
    if complete is not True:
        state = ("its fit_meta has no completion flag (fitted before progressive "
                 "checkpoints)" if complete is None else
                 f"its fit stopped at step {meta.get('steps_run')} of {meta.get('steps')}")
        return f"{name} is not a complete fit: {state} — refit it to completion"
    return None


# --------------------------------------------------------------------------- #
# Fit a named variant
# --------------------------------------------------------------------------- #

def parse_loss_knees(raw: str | Sequence[float] | None) -> tuple[float, ...] | None:
    """``"all"`` (default: 11 knees 0.1–1e4 e⁻), ``"band"`` (the band knee
    only → ``None``) or a comma list of positive electrons."""
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in ("", "all")):
        return ALL_KNEE_LOSS
    if isinstance(raw, str):
        if raw.strip().lower() == "band":
            return None
        try:
            values = [float(t) for t in raw.split(",") if t.strip()]
        except ValueError as exc:
            raise ValueError(f"loss knees must be numbers: {raw!r}") from exc
    else:
        values = [float(v) for v in raw]
    if not values or any(not np.isfinite(v) or v <= 0 for v in values):
        raise ValueError("loss knees must be positive electrons")
    return tuple(sorted(set(values)))


def fit_gate_variant(fields: Sequence[GateField], labels: Sequence[str], *,
                     out_dir: str,
                     holdout: int = 15, seed: int = 0,
                     blackout_fields: int = 40,
                     runner: MemberRunFn | None = None,
                     blackout_dir: str | None = None,
                     source_fingerprint: str | None = None,
                     width: int = 32, use_lr: bool = False, steps: int = 2000,
                     batch_size: int = 8, crop: int = 192, learning_rate: float = 2e-3,
                     eval_every: int = 250,
                     members: Sequence[str] | None = None,
                     loss_knees: Sequence[float] | None = ALL_KNEE_LOSS,
                     mix_space: str = MIX_LINEAR,
                     starfull: bool = True, records_fp: str | None = None,
                     extra_meta: Mapping[str, Any] | None = None,
                     progress: ProgressFn | None = None,
                     log: LogFn | None = None) -> SpatialGateCombiner:
    """Fit a gate on ``fields`` (train/held-out split + blackout copies of the
    training fields) and save it — progressively, the best checkpoint so far —
    into ``out_dir``. ``out_dir`` must not be the production artifact."""
    if os.path.basename(os.path.normpath(out_dir)) == PRODUCTION_DIR:
        raise ValueError("refusing to fit into the production gate directory; "
                         "fit a named variant and promote it")
    if len(fields) < 2:
        raise RuntimeError("the spatial gate needs at least two fields")
    labels = [str(v) for v in labels]
    active = resolve_active_members(members, labels)
    train, held = split_holdout(fields, holdout, seed)
    extra: list[GateField] = []
    if blackout_fields > 0 and runner is not None and blackout_dir:
        extra = build_blackout_fields(
            train, labels, runner, blackout_dir, max_fields=blackout_fields, seed=seed,
            source_fingerprint=source_fingerprint, progress=progress)
    if log is not None:
        log(f"{len(train)} train fields (+{len(extra)} blackout), {len(held)} held out, "
            f"{len(labels) if active is None else len(active)} members")

    def save(comb: SpatialGateCombiner) -> None:
        comb.starfull = bool(starfull)
        comb.records_fp = records_fp
        comb.fit_meta["blackout_fields"] = len(extra)
        comb.fit_meta["holdout_count"] = len(held)
        comb.fit_meta["seed"] = int(seed)
        comb.fit_meta["eval_every"] = int(eval_every)
        comb.fit_meta.update(dict(extra_meta or {}))
        save_spatial_gate(comb, out_dir)

    os.makedirs(out_dir, exist_ok=True)
    marker = os.path.join(out_dir, FIT_MARKER)
    with open(marker, "w") as handle:
        json.dump({"pid": os.getpid(), "host": socket.gethostname(),
                   "started": datetime.now(UTC).isoformat(timespec="seconds")}, handle)
    try:
        comb = fit_spatial_gate(
            list(train) + extra, held, labels, width=int(width), use_lr=bool(use_lr),
            steps=int(steps), batch_size=int(batch_size), crop=int(crop),
            learning_rate=float(learning_rate), eval_every=int(eval_every), seed=int(seed),
            active_members=active, loss_knees=loss_knees, mix_space=mix_space,
            progress=progress, log=log, checkpoint=save)
        save(comb)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(marker)
    return comb


def load_fit_fields(cubes_dir: str, records_dir: str, *, subset: str = "validate",
                    target_name: str = "hr", progress: ProgressFn | None = None,
                    fingerprints: Mapping[str, str | None] | None = None
                    ) -> tuple[list[GateField], list[str]]:
    """The cached ``subset`` member cubes paired with their target and LR —
    refused unless every member's cubes were made by its current checkpoint
    (``fingerprints``; default: the members under the ensemble dir now)."""
    manifest = _cube_manifest(cubes_dir)
    if fingerprints is None:
        fingerprints = member_fingerprints(default_ensemble_dir(),
                                           manifest_member_labels(manifest))
    require_current_cubes(cubes_dir, manifest, fingerprints,
                          "refresh them with a combiner fit in the console (it re-infers "
                          "only those members)")
    fwhm = float(manifest["target_psf_fwhm_arcsec"])
    return load_cube_fields(cubes_dir, records_dir, subset, target_name=target_name,
                            target_fwhm_arcsec=fwhm, progress=progress)
