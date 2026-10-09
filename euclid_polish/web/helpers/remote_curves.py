"""Training curves of members that are still on FASRC.

Models › Curves draws every pulled member from its ``training_log.csv``. A
member that is training — or finished but not pulled yet — has no local log,
so this module reads its curve from the job's Reporter event stream instead:
the trainer writes one ``metric`` event per validation with the same columns
as a training-log row. Every member of a recent ``ensemble_train`` submission
(:data:`MAX_AGE_S`) that is not an active local member and was never archived
gets an entry; one SSH call reads the ``metric`` lines of all their streams,
and the result is reused for :data:`TTL_S` seconds because the page polls
while a member trains.
"""

from __future__ import annotations

import json
import math
import shlex
import threading
import time
from collections.abc import Iterable, Sequence
from typing import Any

from euclid_polish.config import Config
from euclid_polish.training import log_plot
from euclid_polish.web import fasrc_jobs
from euclid_polish.web.remote import STATE

#: Members of ``ensemble_train`` jobs submitted this recently (seconds) are
#: shown until they are pulled.
MAX_AGE_S = 3 * 24 * 3600.0
#: Seconds one fetch of the remote curves is reused.
TTL_S = 20.0
#: Band → training-log column of its validation PSNR.
BAND_LOG_COLUMNS = {"VIS": "psnr_vis", "Y_E": "psnr_y_e",
                    "J_E": "psnr_j_e", "H_E": "psnr_h_e"}

_RECORD_SEP = "\x1e"
_cache: dict[str, Any] = {"at": 0.0, "key": None, "members": []}
_lock = threading.Lock()


def _finite(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def series_from_records(records: Iterable[dict]) -> dict | None:
    """One member's rollback-deduped validation history from training-log
    rows (or ``metric`` events, which carry the same columns): joint and
    per-band PSNR, the combined training loss, the raw loss, the gradient norm
    (mean and max) and the wall time per 1000 steps, each ``[[step, value],
    …]``. ``None`` when there is neither a PSNR nor a loss to draw."""
    recs = [r for r in records if str(r.get("is_baseline", "")).strip()
            not in ("1", "1.0", "true", "True")]
    recs = log_plot.dedupe_latest_per_step(recs)

    def col(key: str) -> list[list[float]]:
        # 5 significant digits: plenty for a chart, ~40 % less JSON.
        out = []
        for r in recs:
            v = _finite(r.get(key))
            if v is not None and r.get("step") is not None:
                out.append([int(r["step"]), float(f"{v:.5g}")])
        return out

    step_time = []
    previous = None
    for r in recs:
        step, dur = r.get("step"), _finite(r.get("duration_s"))
        if step is not None and dur is not None and dur > 0:
            span = int(step) - (previous if previous is not None else 0)
            if span > 0:
                step_time.append([int(step), float(f"{dur * 1000.0 / span:.4g}")])
        if step is not None:
            previous = int(step)
    series = {
        "psnr": col("psnr_stretched"),
        "band_psnr": {band: col(key) for band, key in BAND_LOG_COLUMNS.items()},
        "loss_series": col("combined_loss"),
        "train_loss": col("loss"),
        "gnorm": col("gnorm_avg"),
        "gnorm_max": col("gnorm_max"),
        "step_time": step_time,
    }
    if not (series["psnr"] or series["loss_series"]):
        return None
    return series


def _int(value: Any) -> int | None:
    v = _finite(value)
    return int(v) if v is not None else None


def _member_specs(params: dict) -> list[dict]:
    try:
        spec = json.loads(params.get("member_spec") or "[]")
    except (TypeError, ValueError):
        return []
    return [s if isinstance(s, dict) else {} for s in spec] if isinstance(spec, list) else []


def remote_tasks(rows: Sequence[dict], skip: set[str], *,
                 now: float, max_age_s: float = MAX_AGE_S) -> list[dict]:
    """One entry per member of the ``rows`` (``ensemble_train`` job-DB rows,
    newest first) submitted within ``max_age_s`` whose name is not in
    ``skip``: ``{name, jobid, events_path, facets}``. A member trained by
    several of these jobs is taken from the newest."""
    out: list[dict] = []
    seen = set(skip)
    for row in rows:
        submitted = _finite(row.get("submitted_at"))
        if submitted is None or now - submitted > max_age_s:
            continue
        try:
            params = json.loads(row.get("params_json") or "{}")
        except (TypeError, ValueError):
            continue
        if not isinstance(params, dict):
            continue
        mode = str(params.get("mode") or "add")
        raw = params.get("members") if mode == "continue" else params.get("member_names")
        names = [n.strip() for n in str(raw or "").split(",") if n.strip()]
        count = max(1, _int(params.get("array_count")) or 1)
        specs = _member_specs(params)
        jobid = str(row.get("jobid") or "")
        for i, name in enumerate(names):
            if name in seen:
                continue
            path = (fasrc_jobs.expand_array_path(row.get("events_path"), jobid, i)
                    if count > 1 else row.get("events_path"))
            if not path or "%" in str(path):
                continue
            seen.add(name)
            spec = specs[i] if i < len(specs) else {}
            learned = bool(spec.get("learn_output_knee", params.get("learn_output_knee")))
            knees = spec.get("asinh_knees", params.get("asinh_knees"))
            if isinstance(knees, str):
                knees = [float(q) for q in knees.split(",") if q.strip()] or None
            out.append({"name": name, "jobid": jobid, "events_path": str(path), "facets": {
                "loss_norm": str(spec.get("loss") or params.get("loss") or "l1"),
                "blocks": _int(spec.get("num_res_blocks", params.get("num_res_blocks")))
                or Config.DEFAULT_NUM_RES_BLOCKS,
                "asinh_knee": _finite(spec.get("asinh_knee", params.get("asinh_knee"))),
                "asinh_knees": knees or None,
                "output_knee": (None if learned
                                else _finite(spec.get("output_knee", params.get("output_knee")))),
                "learned_output_knee": learned,
                "knee_loss": spec.get("knee_loss", params.get("knee_loss")) if knees else None,
                "target_steps": _int(params.get("steps")),
                "starless": str(spec.get("starless", params.get("starless", "0"))).strip()
                in ("1", "true", "True"),
            }})
    return out


def parse_metric_streams(text: str, n: int) -> list[list[dict]]:
    """Split the output of :func:`_fetch_command` into ``n`` lists of
    ``metric`` event values (malformed lines are skipped)."""
    out: list[list[dict]] = [[] for _ in range(n)]
    for chunk in text.split(_RECORD_SEP)[1:]:
        head, _, body = chunk.partition("\n")
        try:
            i = int(head.strip())
        except ValueError:
            continue
        if not 0 <= i < n:
            continue
        for line in body.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            value = event.get("value") if isinstance(event, dict) else None
            if event.get("kind") == "metric" and isinstance(value, dict):
                out[i].append(value)
    return out


def _fetch_command(paths: Sequence[str]) -> str:
    parts = [f"printf '{_RECORD_SEP}{i}\\n'; grep -E '\"kind\" *: *\"metric\"' "
             f"{shlex.quote(p)} 2>/dev/null;" for i, p in enumerate(paths)]
    return " ".join(parts) + " exit 0"


def remote_entries(tasks: Sequence[dict], streams: Sequence[list[dict]]) -> list[dict]:
    """Curve entries (the shape of the local ones, plus ``remote``, ``jobid``,
    ``last_step`` and ``finished``) for the tasks whose stream has a curve."""
    out = []
    for task, values in zip(tasks, streams, strict=True):
        series = series_from_records(values)
        if series is None:
            continue
        facets = task["facets"]
        last = max((int(v["step"]) for v in values if _int(v.get("step")) is not None),
                   default=None)
        target = facets.get("target_steps") or _int(next(
            (v.get("total") for v in reversed(values) if v.get("total")), None))
        out.append({
            "name": task["name"],
            "label": f"{task['name'].removeprefix('member_')}·psnr",
            **series,
            "loss": series["loss_series"],
            **facets,
            "target_steps": target,
            "test_psnr": None,
            "remote": True,
            "jobid": task["jobid"],
            "last_step": last,
            "finished": bool(target and last is not None and last >= target),
        })
    return out


def remote_training_curves(local_names: Iterable[str], archived_names: Iterable[str]
                           ) -> list[dict]:
    """Curve entries of the members of recent ``ensemble_train`` jobs that are
    neither active locally nor archived, read from their Reporter streams on
    FASRC; ``[]`` without a connection. Cached for :data:`TTL_S` seconds."""
    ssh = STATE.ssh
    if ssh is None or not ssh.is_connected():
        return []
    skip = set(local_names) | set(archived_names)
    now = time.time()
    tasks = remote_tasks(fasrc_jobs.DB.list_by_step("ensemble_train"), skip, now=now)
    key = tuple((t["name"], t["events_path"]) for t in tasks)
    with _lock:
        if _cache["key"] == key and now - _cache["at"] < TTL_S:
            return list(_cache["members"])
    if not tasks:
        members: list[dict] = []
    else:
        try:
            rc, text, _err = ssh.run(_fetch_command([t["events_path"] for t in tasks]),
                                     timeout=20)
        except Exception:  # noqa: BLE001 — a failed read just shows no remote curves
            return []
        if rc != 0:
            return []
        members = remote_entries(tasks, parse_metric_streams(text, len(tasks)))
    with _lock:
        _cache.update(at=now, key=key, members=members)
    return list(members)
