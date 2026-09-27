"""model routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

import json

from flask import jsonify

from euclid_polish.web.helpers.real_field import (
    FIELD_SIZE,
    REAL_FIELD_DIAGNOSTICS_VERSION,
    cache_real_field,
    field_dir,
    latest_field,
    refresh_real_field_combiners,
)
from euclid_polish.web.jobs import REGISTRY


def register(app):

    @app.route("/api/inference/field.json")
    def api_inference_field():
        return jsonify({"field": latest_field(), "field_size": FIELD_SIZE})

    @app.route("/api/inference/diagnostics.json")
    def api_inference_diagnostics():
        field = latest_field()
        if field is None:
            return jsonify({"diagnostics": None})
        try:
            with (field_dir(str(field["field_id"])) / "diagnostics.json").open() as f:
                diagnostics = json.load(f)
            if diagnostics.get("version") != REAL_FIELD_DIAGNOSTICS_VERSION:
                diagnostics = None
        except (OSError, ValueError, KeyError):
            diagnostics = None
        return jsonify({"diagnostics": diagnostics})

    @app.route("/inference/refresh-combiners", methods=["POST"])
    def inference_refresh_combiners():
        """Apply the newest STARFULL combiner, rebuilding stale members."""
        field = latest_field()
        if field is None:
            return jsonify({"error": "no cached real Euclid field"}), 400
        identifier = str(field["field_id"])

        def refresh(cap):
            def progress(done, total, label):
                cap.tick(done, total, label)

            try:
                return refresh_real_field_combiners(
                    identifier, progress=progress)
            except RuntimeError as error:
                if "member cache is stale" not in str(error):
                    raise
                return cache_real_field(
                    float(field["ra"]), float(field["dec"]),
                    progress=progress)

        job_id = REGISTRY.spawn(
            label=f"refresh real-field inference ({identifier})",
            target=refresh,
        )
        return jsonify({"job_id": job_id})
