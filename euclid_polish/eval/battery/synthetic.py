"""Test 1 — the synthetic score: knee-integrated PSNR per field, knee and band
(the Leaderboard's metric) and the error direction of the spectra record, on
the score's test fields.

Read from the console's universal-score store (``Models › Universal`` and its
``milestones/``) when it holds an entry of the same checkpoint and identity;
otherwise — or with ``rescore`` — measured here from the SR cache with the
score job's own per-field function (:func:`universal_models._field_measures`),
fields in the same order, so a measured score equals a stored one bit for bit.
Nothing is written to the console's store.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from euclid_polish.eval.battery.common import (
    Battery,
    ModelInfo,
    field_key,
    score_identity,
    target_fwhm,
    test_fields,
)
from euclid_polish.eval.combiner import BAND_NAMES
from euclid_polish.eval.knee_psnr import KNEE_GRID_E, integrated_psnr
from euclid_polish.eval.model_spectra import SpectraAccumulator
from euclid_polish.training.inference import infer_checkpoint_learned_output_knee
from euclid_polish.web.helpers import ensemble_viz as ev
from euclid_polish.web.helpers import milestones
from euclid_polish.web.helpers import universal_models as um

STAGE = "synthetic"
#: Period bands of the error-direction axis / diagonal ratio in the report.
FINE_PERIODS = ("2-3",)
COARSE_PERIODS = ("3-6", "6-12", "12-21", "21-42")


def _store_paths(info: ModelInfo) -> tuple[dict, dict | None, str, str]:
    """``(index, its entry for this model, curves path, spectra path)`` in the
    console's score store."""
    if info.kind == "milestone":
        index = milestones.read_scores(info.member)
        entry = (index.get("steps") or {}).get(str(info.step))
        base, stem = str(milestones.scores_dir()), f"{info.member}@{info.step}"
    else:
        index = ev.read_universal_scores()
        entry = (index.get("models") or {}).get(info.member)
        base, stem = ev.universal_scores_dir(), info.member
    return index, entry, os.path.join(base, f"{stem}.npz"), os.path.join(base, f"{stem}.spectra.json")


def stored_score(info: ModelInfo) -> tuple[dict, np.ndarray, np.ndarray, dict] | None:
    """``(entry, fields, curves (F, K, C), spectra record)`` of a current
    console score of this checkpoint, else ``None``."""
    index, entry, curves_path, spectra_path = _store_paths(info)
    if not entry or entry.get("fingerprint") != info.fingerprint:
        return None
    if not ev.universal_identity_matches(index.get("identity"), score_identity()):
        return None
    if not (os.path.isfile(curves_path) and os.path.isfile(spectra_path)):
        return None
    with open(spectra_path, encoding="utf-8") as fh:
        record = json.load(fh)
    if record.get("fingerprint") != info.fingerprint:
        return None
    with np.load(curves_path) as data:
        return entry, np.asarray(data["fields"]), np.asarray(data["curves"], np.float64), record


def measure_score(bt: Battery, info: ModelInfo) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(fields, curves (F, K, C), spectra record)`` measured on the SR cache
    exactly as the score job measures them."""
    fwhm = target_fwhm()
    fields: list[int] = []
    curves: list[np.ndarray] = []
    spectra = SpectraAccumulator()
    pending = []

    def drain(limit: int) -> None:
        while len(pending) > limit:
            index, future = pending.pop(0)
            curve, sums = future.result()
            curves.append(curve)
            spectra.merge(sums)
            fields.append(index)

    with ThreadPoolExecutor(max_workers=ev._KNEE_PSNR_WORKERS) as pool:
        for index, lr, target in test_fields(bt.limits.fields):
            sr = bt.cache.get(info, field_key(index), lr)
            pending.append((index, pool.submit(um._field_measures, sr, target, fwhm)))
            drain(2 * ev._KNEE_PSNR_WORKERS)
        drain(0)
    return np.asarray(fields, np.int64), np.stack(curves), spectra.result(list(BAND_NAMES))


def axis_diag_ratios(record: dict) -> dict[str, list[float]]:
    """``{period band: [axis / diagonal per band]}`` of a spectra record's
    error direction (1 = no preferred direction; > 1 = a pixel-axis grid)."""
    pairs = ((record.get("direction") or {}).get("axis_diag")) or {}
    return {band: [round(a / d, 4) if d else None for a, d in rows] for band, rows in pairs.items()}


def per_field_integrated(curves: np.ndarray) -> np.ndarray:
    """``(F, C)`` ∫PSNR of each field's own curve."""
    return np.stack([integrated_psnr(c) for c in np.asarray(curves, np.float64)])


def run(bt: Battery, info: ModelInfo) -> dict:
    def compute():
        stored = stored_score(info)
        check = None
        if stored is not None and not bt.rescore and bt.limits.fields is None:
            entry, fields, curves, record = stored
            source = "console score store"
        else:
            fields, curves, record = measure_score(bt, info)
            source = "measured from the SR cache"
            if stored is not None and bt.limits.fields is None:
                same = np.array_equal(stored[1], fields)
                check = {"fields_same": bool(same),
                         "max_abs_curve_diff_db": (float(np.max(np.abs(stored[2] - curves))) if same
                                                   and stored[2].shape == curves.shape else None),
                         "stored_mean": stored[0].get("mean")}
        integrated = integrated_psnr(curves.mean(axis=0))
        learned = infer_checkpoint_learned_output_knee(info.member_dir)
        payload = {"source": source, "check": check, "n_fields": int(len(fields)),
                   "integrated": [round(float(v), 4) for v in integrated],
                   "mean": round(float(np.mean(integrated)), 4),
                   "learned_knees": None if learned is None else [round(float(q), 4) for q in learned],
                   "axis_diag": (record.get("direction") or {}).get("axis_diag"),
                   "axis_diag_ratio": axis_diag_ratios(record),
                   "spectra_available": bool(record.get("available"))}
        arrays = {"fields": np.asarray(fields, np.int64), "knees": np.asarray(KNEE_GRID_E, np.float64),
                  "curves": np.asarray(curves, np.float64)}
        return payload, arrays

    return bt.stage(info, STAGE, compute)


def summary(bt: Battery, infos: list[ModelInfo]) -> dict:
    """Per model: ∫PSNR (mean, per band), the error-direction ratio ranges,
    the learned knees; per model vs the first: fields won and per-knee Δ."""
    out: dict = {"models": {}, "vs_baseline": {}}
    arrays = {}
    for info in infos:
        res = bt.store.read(info, STAGE)
        if res is None:
            continue
        arrays[info.spec] = bt.store.arrays(info, STAGE)
        ratio = res["axis_diag_ratio"]
        fine = [v for p in FINE_PERIODS for v in ratio.get(p, []) if v is not None]
        coarse = [v for p in COARSE_PERIODS for v in ratio.get(p, []) if v is not None]
        out["models"][info.spec] = {
            "mean": res["mean"], "integrated": res["integrated"], "n_fields": res["n_fields"],
            "source": res["source"], "check": res["check"], "learned_knees": res["learned_knees"],
            "axis_diag_2_3": [min(fine), max(fine)] if fine else None,
            "axis_diag_3_42": [min(coarse), max(coarse)] if coarse else None}
    if not infos or infos[0].spec not in arrays:
        return out
    base = arrays[infos[0].spec]
    base_fields = per_field_integrated(base["curves"]).mean(axis=1)
    for info in infos[1:]:
        if info.spec not in arrays:
            continue
        cur = arrays[info.spec]
        if not np.array_equal(cur["fields"], base["fields"]):
            continue
        per_field = per_field_integrated(cur["curves"]).mean(axis=1)
        delta = cur["curves"].mean(axis=0) - base["curves"].mean(axis=0)       # (K, C)
        out["vs_baseline"][info.spec] = {
            "fields_better": int((per_field > base_fields).sum()), "n_fields": int(per_field.size),
            "per_knee_delta_db": np.round(delta, 3).tolist(),
            "knees": [float(k) for k in cur["knees"]],
            "nisp_delta_range_db": [round(float(delta[:, 1:].min()), 3), round(float(delta[:, 1:].max()), 3)],
            "vis_delta_at_lowest_knee_db": round(float(delta[0, 0]), 3)}
    return out
