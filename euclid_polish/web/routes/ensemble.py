"""Ensemble routes: view member status, render a field's disagreement
(hallucination cross-check), and evaluate the ensemble on the held-out test set.
"""

from __future__ import annotations

from flask import jsonify, render_template, request

from euclid_polish.web.helpers.ensemble_viz import (
    ensemble_status,
    job_ensemble_evaluate,
    job_ensemble_render,
)
from euclid_polish.web.jobs import REGISTRY


def register(app):

    @app.route("/ensemble")
    def ensemble_page():
        return render_template("ensemble.html", **ensemble_status())

    @app.route("/ensemble/render", methods=["POST"])
    def ensemble_render():
        try:
            index = max(0, int(request.form.get("index", 0) or 0))
        except (TypeError, ValueError):
            index = 0
        job_id = REGISTRY.spawn(
            f"ensemble: disagreement @ test field {index}",
            target=lambda cap: job_ensemble_render(cap, index=index),
        )
        return jsonify({"job_id": job_id})

    @app.route("/ensemble/evaluate", methods=["POST"])
    def ensemble_evaluate():
        try:
            num_images = max(1, int(request.form.get("num_images", 100) or 100))
        except (TypeError, ValueError):
            num_images = 100
        job_id = REGISTRY.spawn(
            f"ensemble: evaluate on {num_images} test fields",
            target=lambda cap: job_ensemble_evaluate(cap, num_images=num_images),
        )
        return jsonify({"job_id": job_id})
