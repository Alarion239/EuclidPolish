"""Dedicated local controls and artifact status for lens isolation."""

from __future__ import annotations

import glob
import json
import os

from flask import jsonify, render_template, request

from euclid_polish.experiments.lens_isolation.config import ExperimentPaths
from euclid_polish.web import fasrc_config
from euclid_polish.web.remote import STATE


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _status() -> dict:
    paths = ExperimentPaths()
    dataset = _read_json(os.path.join(paths.records, "dataset.json"))
    metrics = _read_json(os.path.join(paths.evaluation, "metrics.json"))
    members = []
    for directory in sorted(glob.glob(os.path.join(paths.ensemble, "member_*"))):
        if not os.path.isdir(directory):
            continue
        origin = _read_json(os.path.join(directory, "origin.json")) or {}
        members.append(
            {
                "name": os.path.basename(directory),
                "checkpoint": bool(
                    os.path.isfile(os.path.join(directory, "checkpoint"))
                    or glob.glob(os.path.join(directory, "*.index"))
                ),
                "source": origin.get("source"),
                "seed": origin.get("seed"),
            }
        )
    return {
        "ok": True,
        "root": os.path.abspath(paths.root),
        "records": {"present": dataset is not None, "dataset": dataset},
        "ensemble": {"present": bool(members), "members": members},
        "evaluation": {"present": metrics is not None, "metrics": metrics},
    }


def register(app):
    @app.route("/lens-isolation")
    def lens_isolation_page():
        return render_template("lens_isolation.html", status=_status())

    @app.route("/api/lens-isolation/status")
    def api_lens_isolation_status():
        return jsonify(_status())

    @app.route("/api/lens-isolation/sync", methods=["POST"])
    def api_lens_isolation_sync():
        if STATE.ssh is None or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        paths = ExperimentPaths()
        cfg = fasrc_config.load()
        include_records = str(request.form.get("records", "")).lower() in {"1", "true", "yes", "on"}
        include_ensemble = str(request.form.get("ensemble", "")).lower() in {"1", "true", "yes", "on"}
        selected = [("evaluation", paths.evaluation)]
        if include_records:
            selected.append(("records", paths.records))
        if include_ensemble:
            selected.append(("ensemble", paths.ensemble))
        synced = []
        for name, local in selected:
            remote = cfg.data_dir.rstrip("/") + f"/experiments/lens_isolation/{name}/"
            os.makedirs(local, exist_ok=True)
            rc, _out, err = STATE.ssh.rsync_pull(remote, local, timeout=1800)
            if rc != 0:
                return jsonify({"ok": False, "error": err.strip() or f"rsync {name} exited {rc}"}), 500
            synced.append(name)
        return jsonify({"ok": True, "synced": synced, "status": _status()})
