"""Resource advisor routes (Runs › Resources): what past FASRC runs asked for
vs used, and what to ask for next.

* ``GET  /api/fasrc/resources``                      — per-step usage summaries.
* ``GET  /api/fasrc/resources/<step_id>``            — summary, recent runs and a
  recommendation for "the next run like the last one".
* ``POST /api/fasrc/resources/<step_id>/recommend``  — a recommendation for the
  planned task params (completed as a submit completes them) + the form's
  current resources.

All local and offline: the job ledger (``fasrc_jobs.JOBLOG``) is a local CSV,
so nothing here is ``@requires_fasrc`` and nothing starts a job. The POST is a
POST only because the params are large; it mutates nothing. The recommender
itself is pure (:mod:`euclid_polish.observability.resource_advisor`).
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Sequence
from typing import Any

from flask import abort, jsonify, request

from euclid_polish.observability import resource_advisor as advisor
from euclid_polish.observability.resource_advisor import RunUsage
from euclid_polish.web import fasrc_jobs, job_config
from euclid_polish.web.fasrc_pipeline import REGISTRY as STEP_REGISTRY
from euclid_polish.web.fasrc_pipeline import FASRCPipelineStep, TaskParamError

#: ``GET /api/fasrc/resources/<step_id>`` lists at most this many runs.
_MAX_RUNS = 200
#: Form keys that are neither task params nor recommendable resources.
_META_KEYS = frozenset({"partition", "confirm", "label", "preset", "step_id"})

# The ledger is re-read per request, but its ~3 MB of rows are parsed only
# when the CSV changes: cached on (path, mtime_ns, size).
_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"key": None, "runs": ()}


def _ledger_runs() -> tuple[RunUsage, ...]:
    """Every ledger row, normalised (cached until the CSV changes)."""
    log = fasrc_jobs.JOBLOG
    path = log.csv_path
    try:
        stat = os.stat(path)
    except OSError:
        return ()
    key = (path, stat.st_mtime_ns, stat.st_size)
    with _CACHE_LOCK:
        if _CACHE["key"] == key:
            return _CACHE["runs"]
    runs = advisor.normalize_rows(log.list_all())
    with _CACHE_LOCK:
        _CACHE.update(key=key, runs=runs)
    return runs


def _step(step_id: str) -> FASRCPipelineStep | None:
    return STEP_REGISTRY.by_id.get(step_id)


def _summary(step_id: str, runs: Sequence[RunUsage]) -> dict[str, Any]:
    """The advisor's StepSummary plus ``registered``: whether the console
    still submits the step (a historical one only has its ledger rows)."""
    step = _step(step_id)
    summary = advisor.summarize_step(
        step_id, runs,
        label=step.label if step is not None else None,
        needs_gpu=step.needs_gpu if step is not None else None)
    return {**summary, "registered": step is not None}


def _planned_params(step_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """The posted task params completed the way the submit route completes
    them (``routes/fasrc.py``) before a run reaches the ledger: absent or
    blank task params take the step's schema defaults, then /config's params
    are merged in (``synthetic_generate``'s scene counts and image size, which
    no step card posts). Without this a plan's work and similarity key never
    match its own history. A value the schema refuses (a field being typed)
    keeps the posted params as they are; a historical step has no schema."""
    step = _step(step_id)
    if step is None:
        return params
    with contextlib.suppress(TaskParamError):
        params = step.fill_task_params(params)
    return job_config.with_fasrc_params(step_id, params)


def _recommendation(step_id: str, runs: Sequence[RunUsage], params: dict[str, Any],
                    current: dict[str, Any], *,
                    defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    step = _step(step_id)
    if defaults is None and step is not None:
        defaults = step.defaults.to_dict()
    return {"ok": True, **advisor.recommend(
        step_id, runs, params=params, current=current,
        needs_gpu=step.needs_gpu if step is not None else None,
        fixed_cpus=step.fixed_cpus if step is not None else None,
        fixed_gpus=step.fixed_gpus if step is not None else None,
        defaults=defaults)}


def _step_runs_or_404(step_id: str) -> list[RunUsage]:
    """The step's runs; 404 for a step neither registered nor in the ledger
    (a historical step with rows still answers)."""
    runs = [r for r in _ledger_runs() if r.step_id == step_id]
    if not runs and _step(step_id) is None:
        abort(404, description=f"unknown step: {step_id}")
    return runs


def _recommend_body() -> tuple[dict[str, Any], dict[str, Any]]:
    """``(params, resources)`` from a JSON ``{params, resources}`` body, or a
    flat form / JSON dict (the resource keys split out, the rest params)."""
    if request.is_json:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            abort(400, description="the body must be a JSON object")
        if "params" in body or "resources" in body:
            params, resources = body.get("params") or {}, body.get("resources") or {}
            if not isinstance(params, dict) or not isinstance(resources, dict):
                abort(400, description="params and resources must be JSON objects")
            flat = {**params, **resources}
        else:
            flat = dict(body)
    else:
        flat = request.form.to_dict(flat=True)
    resources = {k: flat[k] for k in advisor.RESOURCE_FIELDS if k in flat}
    params = {k: v for k, v in flat.items()
              if k not in advisor.RESOURCE_FIELDS and k not in _META_KEYS}
    return params, resources


def register(app):
    # =========================================================================
    # Resource advisor — past-run usage (local job ledger) + recommendations.
    # =========================================================================

    @app.route("/api/fasrc/resources")
    def api_fasrc_resources():
        """Every step's usage summary: ``ensemble_train`` and
        ``synthetic_generate`` first, then the most recently submitted."""
        runs = _ledger_runs()
        step_ids = list(dict.fromkeys(r.step_id for r in runs if r.step_id))
        steps = advisor.order_summaries(_summary(s, runs) for s in step_ids)
        return jsonify({"ok": True, "steps": steps})

    @app.route("/api/fasrc/resources/<step_id>")
    def api_fasrc_resources_step(step_id: str):
        """One step: summary, its runs (newest first, ≤ 200) and the
        recommendation for the next run like the latest counted one (its
        params and resources; the step's defaults fill any blank)."""
        runs = _step_runs_or_404(step_id)
        latest = advisor.latest_counted(runs)
        params = dict(latest.params) if latest is not None else {}
        current: dict[str, Any] = {}
        if latest is not None:
            # A bare ledger number is SLURM megabytes; the form reads it as GB.
            memory = latest.req_memory.strip()
            current = {"n_cpus": latest.cpus, "n_gpus": latest.gpus,
                       "memory": f"{memory}M" if memory.isdigit() else memory,
                       "time_limit": latest.req_time_limit}
        return jsonify({
            "ok": True,
            "step_id": step_id,
            "summary": _summary(step_id, runs),
            "runs": [r.to_dict() for r in advisor.newest_first(runs)[:_MAX_RUNS]],
            "recommendation": _recommendation(step_id, runs, params, current),
        })

    @app.route("/api/fasrc/resources/<step_id>/recommend", methods=["POST"])
    def api_fasrc_resources_recommend(step_id: str):
        """Recommendation for the planned run: JSON ``{params, resources}``
        or form fields; the params are completed as the submit would (schema
        defaults, /config). Read-only (a POST only for the payload size)."""
        runs = _step_runs_or_404(step_id)
        params, resources = _recommend_body()
        return jsonify(_recommendation(step_id, runs, _planned_params(step_id, params), resources))
