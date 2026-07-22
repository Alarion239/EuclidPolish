"""Dedicated local controls and artifact status for lens isolation."""

from __future__ import annotations

import json
import os

from flask import abort, jsonify, render_template, request, send_file

from euclid_polish.experiments.lens_isolation.config import ExperimentPaths
from euclid_polish.web import fasrc_config
from euclid_polish.web.helpers.lens_isolation_viz import (
    combiner_payload_path,
    compute_combiner_payload,
    compute_evaluation_payload,
    job_combiner_fit,
    job_evaluate,
    payload_path,
    pixel_trace,
    training_curves_payload,
)
from euclid_polish.web.helpers.lens_isolation_viz import (
    status as ensemble_status,
)
from euclid_polish.web.jobs import REGISTRY
from euclid_polish.web.remote import STATE


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _status() -> dict:
    paths = ExperimentPaths()
    metrics = _read_json(os.path.join(paths.evaluation, "metrics.json"))
    result = ensemble_status()
    result.update({
        "ok": True,
        "root": os.path.abspath(paths.root),
        "ensemble": {"present": bool(result["members"]), "members": result["members"]},
        "evaluation": {"present": metrics is not None, "metrics": metrics},
    })
    return result


def _truthy(value: object) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(request.form.get(name, default) or default)))
    except (TypeError, ValueError):
        return default


def _positive_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(request.form.get(name, default) or default))
    except (TypeError, ValueError):
        return default


def _selected_subsets() -> list[str]:
    raw = str(request.form.get("subsets", "") or "").strip()
    if not raw and _truthy(request.form.get("records")):
        return ["train", "validate", "test"]
    subsets = [value.strip() for value in raw.split(",") if value.strip()]
    invalid = sorted(set(subsets) - {"train", "validate", "test"})
    if invalid:
        raise ValueError(f"unknown record subset(s): {', '.join(invalid)}")
    return list(dict.fromkeys(subsets))


def register(app):
    @app.route("/lens-isolation")
    def lens_isolation_page():
        return render_template("lens_isolation.html", status=_status())

    @app.route("/api/lens-isolation/status")
    def api_lens_isolation_status():
        return jsonify(_status())

    @app.route("/api/lens-isolation/ensemble/evaluate", methods=["POST"])
    def api_lens_isolation_ensemble_evaluate():
        num_images = _bounded_int("num_images", 100, 1, 2000)
        force = _truthy(request.form.get("force"))
        job_id = REGISTRY.spawn(
            f"lens isolation: evaluate {num_images} test fields",
            target=lambda cap: job_evaluate(cap, num_images=num_images, force=force),
        )
        return jsonify({"job_id": job_id})

    @app.route("/api/lens-isolation/ensemble/evals.json")
    def api_lens_isolation_ensemble_evals():
        path = payload_path()
        fresh = _truthy(request.args.get("fresh"))
        if ((fresh or not os.path.isfile(path))
                and compute_evaluation_payload() is None
                and not os.path.isfile(path)):
            abort(404)
        return send_file(path, mimetype="application/json", max_age=0)

    @app.route("/api/lens-isolation/ensemble/combiner/fit", methods=["POST"])
    def api_lens_isolation_combiner_fit():
        num_images = _bounded_int("num_images", 100, 1, 2000)
        n_kernels = _positive_int("n_kernels", 128, 2)
        try:
            min_usage = max(0.0, min(0.5, float(request.form.get("min_usage", 0.0) or 0.0)))
        except (TypeError, ValueError):
            min_usage = 0.0
        job_id = REGISTRY.spawn(
            f"lens isolation: fit combiner on {num_images} validate fields (K={n_kernels})",
            target=lambda cap: job_combiner_fit(
                cap,
                num_images=num_images,
                n_kernels=n_kernels,
                min_usage=min_usage,
            ),
        )
        return jsonify({"job_id": job_id})

    @app.route("/api/lens-isolation/ensemble/combiner.json")
    def api_lens_isolation_combiner():
        path = combiner_payload_path()
        if compute_combiner_payload() is None and not os.path.isfile(path):
            abort(404)
        return send_file(path, mimetype="application/json", max_age=0)

    @app.route("/api/lens-isolation/ensemble/training-curves.json")
    def api_lens_isolation_training_curves():
        return jsonify({"members": training_curves_payload()})

    @app.route("/api/lens-isolation/ensemble/pixel-trace.json")
    def api_lens_isolation_pixel_trace():
        diag = str(request.args.get("diag", "") or "").strip()
        if diag not in {"std_err", "bright_std"}:
            abort(404)
        try:
            i = int(request.args.get("i", ""))
            j = int(request.args.get("j", ""))
        except (TypeError, ValueError):
            abort(400)
        return jsonify(pixel_trace(diag, i, j))

    @app.route("/api/lens-isolation/sync", methods=["POST"])
    def api_lens_isolation_sync():
        if STATE.ssh is None or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        paths = ExperimentPaths()
        cfg = fasrc_config.load()
        try:
            subsets = _selected_subsets()
        except ValueError as error:
            return jsonify({"ok": False, "error": str(error)}), 400
        include_ensemble = _truthy(request.form.get("ensemble"))
        include_evaluation = request.form.get("evaluation") is None or _truthy(
            request.form.get("evaluation")
        )
        selected = []
        if include_evaluation:
            selected.append(("evaluation", "evaluation/", paths.evaluation))
        if include_ensemble:
            selected.append(("ensemble", "ensemble/", paths.ensemble))
        if subsets:
            selected.append(("dataset metadata", "records/dataset.json", paths.records))
            for subset in subsets:
                selected.extend(
                    (f"{subset} {label}", f"records/{filename}", paths.records)
                    for label, filename in (
                        ("dirty", f"dirty_{subset}.tfrecord"),
                        ("lens", f"lens_{subset}.tfrecord"),
                        ("sources", f"sources_{subset}.csv"),
                        ("metadata", f"split_{subset}.json"),
                    )
                )
        synced = []
        for name, relative, local in selected:
            remote = cfg.data_dir.rstrip("/") + f"/experiments/lens_isolation/{relative}"
            os.makedirs(local, exist_ok=True)
            rc, _out, err = STATE.ssh.rsync_pull(remote, local, timeout=1800)
            if rc != 0:
                return jsonify({"ok": False, "error": err.strip() or f"rsync {name} exited {rc}"}), 500
            synced.append(name)
        return jsonify({"ok": True, "synced": synced, "status": _status()})
