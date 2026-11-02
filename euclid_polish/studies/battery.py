"""Battery studies: a finished model-battery run (:mod:`euclid_polish.eval.battery`,
``scripts/model_battery.py --study``) frozen as a study of Figures › Studies.

An ensemble study freezes the active membership and recomputes its numbers;
a battery study freezes any chosen single models (``universal:``,
``milestone:``, production) through what the battery already measured: the
comparison table and the paired comparisons (``battery.json``), the report
(``battery.md``) and every figure (``figure-<name>.png``). The files are
copied as they are, hashed into the manifest and sealed read-only like any
study; the manifest names the models with their checkpoint fingerprints, the
test records (directory and fingerprint) and the code. A smoke run
(``--smoke``) is refused: its numbers are not comparable.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from euclid_polish.provenance.gitinfo import capture_git
from euclid_polish.studies.freeze import mirror_numbers
from euclid_polish.studies.store import StudyError, StudyStore

KIND = "battery"
REPORT = "battery.md"
RESULTS = "battery.json"
FIGURE_PREFIX = "figure-"
#: What a battery study keeps of each model (the battery's ModelInfo).
MODEL_KEYS = ("spec", "name", "kind", "member", "step", "fingerprint")


def figure_number(path: str) -> str:
    """The numbers-file name of a battery figure (``figures/x.png`` → ``figure-x.png``)."""
    return FIGURE_PREFIX + os.path.basename(path)


def describe(meta: Mapping[str, Any], source: str) -> dict[str, Any]:
    """The manifest block of a battery run's ``meta``."""
    return {"models": [{k: m.get(k) for k in MODEL_KEYS} for m in meta.get("models") or []],
            "tests": list(meta.get("tests") or []), "created": meta.get("created"),
            "records_fp": meta.get("records_fp"), "records_dir": meta.get("records_dir"),
            "seconds": meta.get("seconds"), "source": os.path.abspath(source),
            "figures": [figure_number(p) for p in meta.get("figures") or []]}


def freeze(battery_dir: str, *, name: str, note: str = "", store: StudyStore | None = None,
           ssh: Any = None, remote_tracking_dir: str | None = None) -> str:
    """Freeze the battery run in ``battery_dir`` as a complete study; returns
    its id. :class:`StudyError` when the run is missing, a smoke run or
    without figures; nothing is left behind then."""
    store = store or StudyStore()
    results = os.path.join(battery_dir, RESULTS)
    try:
        with open(results, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        raise StudyError(404, f"no battery results in {battery_dir}: {exc}") from exc
    meta = payload.get("meta") or {}
    if meta.get("limits"):
        raise StudyError(409, f"{battery_dir} is a smoke run ({meta['limits']}): its numbers are "
                              "not comparable")
    if not meta.get("models"):
        raise StudyError(409, f"{battery_dir}: the run lists no models")
    figures = list(meta.get("figures") or [])
    if not figures:
        raise StudyError(409, f"{battery_dir}: the run has no figures (run it with 'figures')")
    warnings = [f"figure {fig} failed: {why}" for fig, why in (meta.get("figures_failed") or {}).items()]
    files = [(RESULTS, results), (REPORT, os.path.join(battery_dir, REPORT))]
    files += [(figure_number(p), os.path.join(battery_dir, p)) for p in figures]
    missing = [path for _, path in files if not os.path.isfile(path)]
    if missing:
        raise StudyError(409, "battery files missing: " + ", ".join(missing))
    study_id = store.create(name, note, regime="starfull", extra={
        "kind": KIND, "commit": capture_git(), "battery": describe(meta, battery_dir),
        "warnings": warnings})
    try:
        for number, path in files:
            with open(path, "rb") as fh:
                store.write_number(study_id, number, fh.read())
        store.mark_complete(study_id)
    except Exception:
        store.delete(study_id, None, remote_tracking_dir="", local_only=True)
        raise
    if ssh is not None and remote_tracking_dir:
        mirror_numbers(store, study_id, ssh, remote_tracking_dir)
    return study_id


__all__ = ["FIGURE_PREFIX", "KIND", "REPORT", "RESULTS", "describe", "figure_number", "freeze"]
