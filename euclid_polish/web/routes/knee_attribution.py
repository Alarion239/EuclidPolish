"""Knee-attribution routes (``/api/knee-attribution``, spec
``docs/superpowers/specs/2026-10-15-knee-attribution-design.md``): list and
read runs, the multi-knee models a run can take, field images, and starting a
run (a local job; nothing starts on a GET)."""

from __future__ import annotations

from flask import Response, jsonify, request

from euclid_polish.web import errors
from euclid_polish.web.helpers import knee_attribution_runs as runs
from euclid_polish.web.jobs import REGISTRY


def _fail(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


def register(app):
    errors.json_errors_for(app, "/api/knee-attribution")

    @app.get("/api/knee-attribution")
    def api_knee_attribution_list():
        return jsonify({"runs": runs.list_runs()})

    @app.get("/api/knee-attribution/models")
    def api_knee_attribution_models():
        return jsonify({"models": runs.candidate_models()})

    @app.get("/api/knee-attribution/<run_id>")
    def api_knee_attribution_get(run_id: str):
        try:
            return jsonify(runs.get_run(run_id))
        except KeyError as exc:
            return _fail(str(exc.args[0] if exc.args else exc), 404)

    @app.get("/api/knee-attribution/<run_id>/field/<model_slug>/<ref_slug>.png")
    def api_knee_attribution_field(run_id: str, model_slug: str, ref_slug: str):
        try:
            png = runs.field_png(run_id, model_slug, ref_slug,
                                 band=request.args.get("band", "VIS"),
                                 layer=request.args.get("layer", "sr"))
        except KeyError as exc:
            return _fail(str(exc.args[0] if exc.args else exc), 404)
        except ValueError as exc:
            return _fail(str(exc))
        return Response(png, mimetype="image/png",
                        headers={"Cache-Control": "private, max-age=3600"})

    @app.post("/api/knee-attribution")
    def api_knee_attribution_start():
        try:
            params = runs.parse_request(request.form)
        except ValueError as exc:
            return _fail(str(exc))
        run_id = runs.new_run_id()
        n_fields = len(runs.synthetic_indices(params["synthetic"])) + len(params["tiles"])

        def target(cap):
            return runs.job_knee_attribution(cap, run_id=run_id, **params)

        job_id = REGISTRY.spawn(
            f"knee attribution · {len(params['models'])} model(s) × {n_fields} field(s)",
            target, kind="knee-attribution")
        return jsonify({"ok": True, "job_id": job_id, "run_id": run_id})
