"""model routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

from flask import jsonify, redirect, render_template, request

from euclid_polish.config import Config
from euclid_polish.web.helpers.real_field import FIELD_SIZE, cache_real_field, field_dir, latest_field
from euclid_polish.web.helpers.status import _checkpoints_status
from euclid_polish.web.jobs import REGISTRY


def register(app):

    # ---------------- Training (folded into /ensemble) ----------------
    @app.route("/training")
    def training_page():
        """Training is ensemble-only now — the /ensemble page owns TFRecord
        status, the ensemble_train step card, curves and member management."""
        return redirect("/ensemble", code=302)

    # ---------------- Inference page ----------------
    @app.route("/inference")
    def inference_page():
        return render_template(
            "inference.html",
            checkpoints=_checkpoints_status(),
            field=latest_field(),
            default_num_res_blocks=Config.DEFAULT_NUM_RES_BLOCKS,
        )

    @app.route("/api/inference/field.json")
    def api_inference_field():
        return jsonify({"field": latest_field(), "field_size": FIELD_SIZE})

    @app.route("/api/inference/diagnostics.json")
    def api_inference_diagnostics():
        field = latest_field()
        if field is None:
            return jsonify({"diagnostics": None})
        try:
            import json
            with (field_dir(str(field["field_id"])) / "diagnostics.json").open() as f:
                diagnostics = json.load(f)
        except (OSError, ValueError, KeyError):
            diagnostics = None
        return jsonify({"diagnostics": diagnostics})

    @app.route("/inference/cache-real-field", methods=["POST"])
    def inference_cache_real_field():
        try:
            ra  = float(request.form["ra"])
            dec = float(request.form["dec"])
        except (KeyError, ValueError):
            return jsonify({"error": "ra and dec must be valid floats (degrees)"}), 400
        if not (0.0 <= ra < 360.0):
            return jsonify({"error": f"ra={ra} out of range [0, 360)"}), 400
        if not (-90.0 <= dec <= 90.0):
            return jsonify({"error": f"dec={dec} out of range [-90, 90]"}), 400
        job_id = REGISTRY.spawn(
            label=f"cache real Euclid field @ ({ra:.4f}, {dec:+.4f})",
            target=lambda cap: cache_real_field(
                ra, dec, progress=lambda done, total, label: cap.tick(done, total, label)),
        )
        return jsonify({"job_id": job_id})
