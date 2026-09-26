"""Universal job-config API (the Settings › Config tab).

The shared per-job knobs (:mod:`euclid_polish.web.job_config`) persist to
``~/.euclid_polish/job_config.json`` and get injected into the relevant job
submissions. Reachable offline — it's local persistence.

Lost-update protection: ``GET /api/config`` returns a ``version`` (content
hash). ``POST /api/config/save`` applies ONLY the posted fields; with a
``base_version`` it refuses (409 ``config_conflict``) when any posted field
was changed server-side since that version (another tab, a calibration
activation writing ``galaxy_density_arcmin2`` …), and merges untouched fields.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from flask import jsonify, request

from euclid_polish.web import job_config

#: Recently served config snapshots by version, so a save can tell which
#: posted fields changed server-side since the client loaded its copy. An
#: unknown ``base_version`` (older than the window, or a server restart) is
#: treated as "every posted field that differs now is a conflict".
_SNAPSHOT_LIMIT = 64
_SNAPSHOTS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_LOCK = threading.Lock()


def _remember(values: dict[str, Any]) -> str:
    version = job_config.version_of(values)
    with _LOCK:
        _SNAPSHOTS[version] = dict(values)
        _SNAPSHOTS.move_to_end(version)
        while len(_SNAPSHOTS) > _SNAPSHOT_LIMIT:
            _SNAPSHOTS.popitem(last=False)
    return version


def _coerced(field: str, raw: Any) -> Any:
    """``raw`` (a form string) as the field's declared type, for an equality
    check against the loaded config (flags as 0/1, numbers as numbers)."""
    value = job_config.coerce_field(field, raw)
    return raw if value is None else value


def _conflicts(posted: dict[str, Any], base_version: str,
               current: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Posted fields whose server-side value moved since ``base_version``."""
    with _LOCK:
        base = _SNAPSHOTS.get(base_version)
    out: dict[str, dict[str, Any]] = {}
    for field, raw in posted.items():
        if field not in current:
            continue
        if base is not None:
            if base.get(field) != current[field]:
                out[field] = {"base": base.get(field), "current": current[field]}
        elif _coerced(field, raw) != current[field]:
            out[field] = {"base": None, "current": current[field]}
    return out


def register(app):

    @app.route("/api/config")
    def api_config_get():
        values = job_config.load().to_dict()
        return jsonify({"ok": True, "config": values, "version": _remember(values)})

    @app.route("/api/config/save", methods=["POST"])
    def api_config_save():
        posted = request.form.to_dict()
        base_version = (posted.pop("base_version", "") or "").strip()
        # Only real JobConfig fields with a value; ``update`` ignores the rest.
        fields = {key: value for key, value in posted.items()
                  if value != "" and hasattr(job_config.JobConfig, key)}
        current = job_config.load().to_dict()
        if base_version and base_version != job_config.version_of(current):
            conflicts = _conflicts(fields, base_version, current)
            if conflicts:
                return jsonify({
                    "ok": False, "code": "config_conflict",
                    "error": ("the config changed since you loaded it: "
                              + ", ".join(sorted(conflicts))
                              + " — reload and re-apply your edits"),
                    "conflicts": conflicts,
                    "config": current, "version": _remember(current),
                }), 409
        cfg = job_config.update(fields)
        note = None
        try:
            requested = int(fields.get("vis_pixels", cfg.vis_pixels))
            if requested != cfg.vis_pixels:
                note = (f"VIS cutout must be odd — adjusted "
                        f"{requested} → {cfg.vis_pixels}.")
        except (TypeError, ValueError):
            pass
        values = cfg.to_dict()
        return jsonify({"ok": True, "config": values, "note": note,
                        "version": _remember(values)})
