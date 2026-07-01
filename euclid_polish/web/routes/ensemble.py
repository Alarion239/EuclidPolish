"""Ensemble routes: view member status, render a field's disagreement
(hallucination cross-check), and evaluate the ensemble on the held-out test set.
"""

from __future__ import annotations

import os

from flask import abort, jsonify, render_template, request, send_file

from euclid_polish.training.log_plot import plot_ensemble_training_curves
from euclid_polish.web.helpers.ensemble_viz import (
    _ensemble_out_dir,
    ensemble_dir,
    ensemble_status,
    job_ensemble_evaluate,
    job_ensemble_pull,
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

    @app.route("/ensemble/training-curves.png")
    def ensemble_training_curves():
        """Per-member PSNR + loss curves (rollback-deduped). 404 → card hides."""
        out_png = os.path.join(_ensemble_out_dir(), "training_curves.png")
        fresh = request.args.get("fresh", "").lower() in ("1", "true", "yes")
        if fresh or not os.path.isfile(out_png):
            if plot_ensemble_training_curves(ensemble_dir(), out_png) is None:
                abort(404)
        return send_file(out_png, mimetype="image/png", max_age=0)

    @app.route("/ensemble/pull", methods=["POST"])
    def ensemble_pull():
        job_id = REGISTRY.spawn(
            "ensemble: download from FASRC",
            target=job_ensemble_pull,
        )
        return jsonify({"job_id": job_id})
