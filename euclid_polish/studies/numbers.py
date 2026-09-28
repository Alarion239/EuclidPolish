"""The numbers of a study — everything its charts draw, frozen at once.

:func:`build` reads the live ensemble (never writes) and returns a
:class:`NumbersBundle`: the files (bytes, ready for the store) and the
snapshot the manifest records.

* ``members.csv`` — one row per active member: recipe (``origin.json``),
  steps and status, checkpoint fingerprint, the test scores, the
  knee-integrated PSNR per band (mean over the frozen per-field curves) and
  the production gate's usage.
* ``knee_psnr.json`` — PSNR vs knee on ``KNEE_GRID_E`` for every member, the
  plain mean and each baked combiner (the production gate is ``gate``),
  **per field**: ``psnr[model][field][knee][band]`` — the same loop as the
  Leaderboard's curves (:func:`ensemble_viz.knee_psnr_fields`), so its mean
  over fields is the Leaderboard curve and paired statistics are possible.
* ``integrated.csv`` — ``integrated_psnr`` of each per-field curve:
  ``model, kind, field, band, integrated_psnr``.
* ``training_curves.json`` — validation PSNR / loss vs step per member.
* ``gate.json`` — the production gate's held-out weight diagnostic, every
  gate variant's summary and the latest combiner comparison report (or
  ``null`` with ``compare_note`` saying it is absent).
* ``real.json`` — the Sky › Compare experiments whose spec fingerprints are
  this membership's (STARFULL only).
"""

from __future__ import annotations

import csv
import io
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from euclid_polish.config import Config
from euclid_polish.ensemble import member_fingerprint
from euclid_polish.eval.combiner import BAND_NAMES, COMBINER_MODELS, combiner_artifact_fingerprint
from euclid_polish.eval.knee_psnr import KNEE_GRID_E
from euclid_polish.eval.spatial_gate import SPATIAL_GATE_KIND
from euclid_polish.studies import candidates, stats
from euclid_polish.studies.store import StudyError
from euclid_polish.web.helpers import ensemble_viz as ev
from euclid_polish.web.helpers import model_catalog, sky_records

NUMBERS_SCHEMA = 1
FILES = ("members.csv", "knee_psnr.json", "integrated.csv", "training_curves.json",
         "gate.json", "real.json")
#: Recipe columns of members.csv (``origin.json`` / the members table).
RECIPE_COLUMNS = ("label", "name", "loss", "asinh_knee", "asinh_knees", "output_knee",
                  "knee_loss", "blocks", "bootstrap", "noise_aug", "icnr", "seed", "step",
                  "target_steps", "status", "created_at", "commit", "forked_from", "op",
                  "noise_model", "fingerprint")
_DECIMALS = 4

Progress = Callable[[int, int, str], None]


@dataclass
class NumbersBundle:
    files: dict[str, bytes]
    snapshot: dict[str, Any]


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def _csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _cell(row.get(k)) for k in columns})
    return buffer.getvalue().encode("utf-8")


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, list | tuple):
        return ";".join(f"{v:g}" if isinstance(v, int | float) else str(v) for v in value)
    if isinstance(value, float):
        return f"{value:.6g}" if np.isfinite(value) else ""
    return value


def _finite(values: np.ndarray) -> list:
    """Round and map non-finite values to ``None`` (strict JSON)."""
    array = np.round(np.asarray(values, np.float64), _DECIMALS)
    if array.ndim == 0:
        return None if not np.isfinite(array) else float(array)  # type: ignore[return-value]
    return [_finite(v) for v in array]


# ---------------------------------------------------------------------------
# knee curves
# ---------------------------------------------------------------------------

def knee_numbers(starless: bool, progress: Progress | None = None) -> dict[str, Any]:
    """The per-field knee payload (see the module docstring)."""
    fields = ev.knee_psnr_fields(starless, progress=progress)
    if fields is None:
        raise StudyError(409, "no current cached test cubes for the active members — "
                              "re-evaluate the ensemble on the test set before freezing")
    labels = list(fields["labels"])
    models = []
    for model_id in fields["model_ids"]:
        if model_id.startswith("member_"):
            label = labels[int(model_id.removeprefix("member_"))]
            models.append({"id": label, "kind": "member", "label": label})
        elif model_id == "ensemble_mean":
            models.append({"id": "mean", "kind": "mean", "label": "mean of members"})
        elif model_id == SPATIAL_GATE_KIND:
            models.append({"id": "gate", "kind": "gate", "label": "production gate"})
        else:
            models.append({"id": model_id, "kind": "combiner",
                           "label": COMBINER_MODELS[model_id].label})
    curves = np.asarray(fields["curves"], np.float64)          # (F, M, K, C)
    per_model = np.transpose(curves, (1, 0, 2, 3))             # (M, F, K, C)
    scored = [int(r) for r in fields["fields"]]
    dropped = sorted({int(i) for i in fields["identity"]["indices"]} - set(scored))
    return {
        "schema": NUMBERS_SCHEMA, "regime": candidates.regime_slug(starless),
        "knees": list(KNEE_GRID_E), "bands": list(BAND_NAMES),
        "fields": scored, "models": models,
        # Evaluated fields left out because a model's cube (or the target
        # record) was missing — never silently averaged over.
        "dropped_fields": dropped,
        "psnr": _finite(per_model),
        "integration": {"from_e": KNEE_GRID_E[0], "to_e": KNEE_GRID_E[-1],
                        "weighting": "uniform in log10(knee), trapezoid rule"},
        "identity": fields["identity"],
    }


def integrated_rows(knee: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``integrated_psnr`` of every per-field curve (one row per model ×
    field × band)."""
    integrated = stats.field_integrated(knee["psnr"], knee["knees"])   # (M, F, C)
    rows = []
    for m, model in enumerate(knee["models"]):
        for f, field in enumerate(knee["fields"]):
            values = integrated[m, f]
            for b, band in enumerate(knee["bands"]):
                rows.append({"model": model["id"], "kind": model["kind"], "field": field,
                             "band": band, "integrated_psnr": round(float(values[b]), _DECIMALS)})
    return rows


def mean_integrated(knee: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """``{model id: {band: integrated PSNR}}`` — the NaN-aware mean over
    fields of the per-field integrated PSNR (the rule of every study chart)."""
    means = stats.nan_mean(stats.field_integrated(knee["psnr"], knee["knees"]), axis=1)
    out = {}
    for m, model in enumerate(knee["models"]):
        out[model["id"]] = {band: float(means[m, b]) for b, band in enumerate(knee["bands"])}
    return out


# ---------------------------------------------------------------------------
# members
# ---------------------------------------------------------------------------

def member_rows(starless: bool, active: list[str], knee: Mapping[str, Any]
                ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(csv rows, manifest member snapshots)`` in active order."""
    payload = ev.members_payload(starless)
    rows_by_label = {row["label"]: row for row in payload["members"]}
    missing = [label for label in active if label not in rows_by_label]
    if missing:
        raise StudyError(409, "active members without a checkpoint: " + ", ".join(missing))
    integrated = mean_integrated(knee)
    base = ev.ensemble_dir()
    csv_rows, snapshots = [], []
    for label in active:
        row = rows_by_label[label]
        fingerprint = member_fingerprint(os.path.join(base, row["name"]))
        knee_int = integrated.get(label, {})
        values = [v for v in knee_int.values() if np.isfinite(v)]
        entry = {**{k: row.get(k) for k in RECIPE_COLUMNS if k in row},
                 "fingerprint": fingerprint, "psnr_joint": row.get("psnr"),
                 "vis_psnr": row.get("vis_psnr"), "used_by_gate": row.get("used_by_gate"),
                 "timeout": row.get("timeout")}
        for band in BAND_NAMES:
            entry[f"knee_int_{band}"] = knee_int.get(band)
            entry[f"gate_usage_{band}"] = (row.get("gate_usage") or {}).get(band)
            entry[f"gate_usage_source_{band}"] = (row.get("gate_usage_source") or {}).get(band)
        entry["knee_int_mean"] = float(np.mean(values)) if values else None
        csv_rows.append(entry)
        snapshots.append({**{k: v for k, v in entry.items() if not k.startswith("gate_usage")},
                          "origin": row.get("origin") or {},
                          "job": row.get("job"), "size_mb": row.get("size_mb")})
    return csv_rows, snapshots


def member_columns() -> list[str]:
    return [*RECIPE_COLUMNS, "timeout", "psnr_joint", "vis_psnr",
            *(f"knee_int_{b}" for b in BAND_NAMES), "knee_int_mean",
            *(f"gate_usage_{b}" for b in BAND_NAMES),
            *(f"gate_usage_source_{b}" for b in BAND_NAMES), "used_by_gate"]


# ---------------------------------------------------------------------------
# gate / real / records
# ---------------------------------------------------------------------------

def gate_numbers(starless: bool, gate: Mapping[str, Any]) -> dict[str, Any]:
    report = ev._latest_compare(starless)
    variants = ev.combiner_variants(starless)["variants"]
    return {
        "schema": NUMBERS_SCHEMA,
        "production": dict(gate),
        "diagnostic": ev._gate_usage(starless),
        "variants": variants,
        "compare": report,
        "compare_note": (None if report is not None else
                         "No combiner comparison report existed at freeze (natural and "
                         "blackout test fields were not compared); this study keeps the "
                         "gate's weight diagnostic only."),
    }


def real_numbers(starless: bool) -> dict[str, Any]:
    out = []
    for record in candidates.matching_experiments(starless):
        specs = record["matching_specs"]
        results = record.get("results") or {}
        out.append({
            "id": record.get("id"), "label": record.get("label"),
            "created": record.get("created"), "status": record.get("status"),
            "tiles": record.get("tiles") or [], "definitions": record.get("definitions"),
            "specs": {spec: {
                "label": (record.get("model_labels") or {}).get(spec),
                "member_label": (model_catalog.member_label(spec.split(":", 1)[1])
                                 if spec.startswith(model_catalog.MEMBER_PREFIX) else None),
                "fingerprint": (record.get("fingerprints") or {}).get(spec),
                "summary": (record.get("summary") or {}).get(spec),
                "per_tile": {tile: (results.get(tile) or {}).get(spec, {}).get("metrics")
                             for tile in record.get("tiles") or []},
            } for spec in specs},
        })
    return {"schema": NUMBERS_SCHEMA, "regime": candidates.regime_slug(starless),
            "experiments": out,
            "note": ("Real tiles are STARFULL only." if starless else
                     None if out else "No Sky › Compare run used this membership.")}


def gate_identity(starless: bool, gate: Mapping[str, Any]) -> dict[str, Any]:
    regime = ev._regime_dir_ro(starless)
    manifest = {}
    path = os.path.join(regime, candidates.PRODUCTION_DIR, "combiner.json")
    try:
        with open(path) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        manifest = {}
    return {
        **{k: gate.get(k) for k in ("available", "state", "detail", "name", "dir", "kind",
                                    "member_labels", "reads", "mix_space", "use_lr",
                                    "fitted_at")},
        "width": manifest.get("width"),
        "fingerprint": (combiner_artifact_fingerprint(regime, candidates.PRODUCTION_DIR)
                        if gate.get("available") else None),
        "fit": ev._fit_summary(manifest.get("fit_meta") or {}) if manifest else None,
        "promoted_from": (manifest.get("fit_meta") or {}).get("promoted_from"),
    }


def records_identity(starless: bool, manifest: Mapping[str, Any],
                     snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    rdir = ev._sky_records_local_dir()
    subset = str(manifest.get("subset") or "test")
    return {
        "records_fp": manifest.get("records_fp"), "subset": subset,
        "indices": sorted(int(i) for i in manifest.get("indices") or []),
        "target": "clean" if starless else "hr",
        "target_psf_fwhm_arcsec": manifest.get("target_psf_fwhm_arcsec"),
        "records_dir": rdir,
        "generation": sky_records.records_generation(rdir, subset) if rdir else None,
        "noise_models": sorted({str(s.get("noise_model")) for s in snapshots
                                if s.get("noise_model") is not None}),
        "psnr_peak_e": float(Config.PSNR_PEAK_E),
    }


def live_identity(starless: bool) -> dict[str, Any]:
    """What a study's numbers depend on: the members (labels + checkpoint
    fingerprints), the test records and the production gate artifact."""
    active = candidates.active_labels(starless)
    base = ev.ensemble_dir()
    regime = ev._regime_dir_ro(starless)
    cubes = candidates.test_cubes_state(starless, active)
    return {
        "labels": active,
        "fingerprints": {label: member_fingerprint(os.path.join(
            base, model_catalog.member_name(label))) for label in active},
        "records_fp": (cubes.get("manifest") or {}).get("records_fp"),
        "gate_fingerprint": combiner_artifact_fingerprint(regime, candidates.PRODUCTION_DIR),
    }


# ---------------------------------------------------------------------------
# the bundle
# ---------------------------------------------------------------------------

def build(starless: bool, *, progress: Progress | None = None,
          check: Callable[[], None] | None = None) -> NumbersBundle:
    """Every numbers file + the manifest snapshot; :class:`StudyError` 409
    when the test cubes are not the active membership's. ``check`` (a job's
    cancel check) runs at every progress step, the per-field knee loop
    included, so a cancelled freeze stops within one field."""
    report = progress or (lambda *_a: None)
    steps = 6

    def tick(current: float, total: int, label: str) -> None:
        # One scale for the whole build, in thousandths: the knee loop (which
        # dominates) runs 0 → 5 of 6, the short steps after it share the last.
        if check is not None:
            check()
        report(int(1000 * current / total), 1000, label)

    active = candidates.active_labels(starless)
    if not active:
        raise StudyError(409, "no active members to freeze")
    cubes = candidates.test_cubes_state(starless, active)
    if cubes["state"] != "current":
        raise StudyError(409, cubes["detail"])
    tick(0, steps, "knee curves per field")
    knee = knee_numbers(starless, progress=lambda i, n, _l: tick(
        5.0 * i / max(1, n), steps, "knee curves per field"))
    tick(5.0, steps, "members")
    csv_rows, snapshots = member_rows(starless, active, knee)
    gate = candidates.production_gate(starless, active)
    tick(5.2, steps, "training curves")
    curves = [entry for entry in ev.training_curves_payload()
              if bool(entry.get("starless")) == bool(starless) and entry.get("label") in active]
    curves.sort(key=lambda entry: active.index(entry["label"]))
    tick(5.4, steps, "gate")
    gate_payload = gate_numbers(starless, gate_identity(starless, gate))
    tick(5.6, steps, "real-tile metrics")
    real = real_numbers(starless)
    tick(5.8, steps, "writing")
    overview = ev.ensemble_overview(starless)
    files = {
        "members.csv": _csv_bytes(member_columns(), csv_rows),
        "knee_psnr.json": _json_bytes(knee),
        "integrated.csv": _csv_bytes(("model", "kind", "field", "band", "integrated_psnr"),
                                     integrated_rows(knee)),
        "training_curves.json": _json_bytes(curves),
        "gate.json": _json_bytes(gate_payload),
        "real.json": _json_bytes(real),
    }
    summary = candidates.ensemble_summary(starless)
    summary.pop("_cubes", None)
    snapshot = {
        "ensemble": {"members": snapshots, "labels": active, "n_members": len(active)},
        "gate": gate_payload["production"],
        "records": records_identity(starless, cubes["manifest"], snapshots),
        "evaluation": {"evaluated_at": overview.get("evaluated_at"),
                       "checks": overview.get("checks"),
                       "headline": overview.get("headline")},
        "blocks": summary["blocks"],
        "warnings": [f"{b['title']}: {b['detail']}" for b in summary["blocks"]
                     if b["state"] != "current"] + ([
            f"Test-set curves: {len(knee['dropped_fields'])} of "
            f"{len(knee['dropped_fields']) + len(knee['fields'])} test fields were dropped "
            f"(a missing cube or target record): {knee['dropped_fields']}"]
            if knee["dropped_fields"] else []),
        "identity": live_identity(starless),
        "models": knee["models"], "knees": knee["knees"], "bands": knee["bands"],
        "knee_fields": knee["fields"],
    }
    tick(steps, steps, "numbers ready")
    return NumbersBundle(files=files, snapshot=snapshot)


__all__ = [
    "FILES",
    "NumbersBundle",
    "build",
    "integrated_rows",
    "knee_numbers",
    "live_identity",
    "mean_integrated",
    "member_columns",
]
