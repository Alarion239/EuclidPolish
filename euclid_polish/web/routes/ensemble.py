"""Ensemble routes: view member status, render a field's disagreement
(hallucination cross-check), and evaluate the ensemble on the held-out test set.
"""

from __future__ import annotations

import os

from flask import abort, jsonify, render_template, request, send_file

from euclid_polish.web.helpers.ensemble_viz import (
    EVAL_DIAGNOSTIC_PNGS,
    _ensemble_out_dir,
    _evals_payload_path,
    compute_evaluation_payload,
    ensemble_dir,
    ensemble_status,
    job_archive_member,
    job_ensemble_evaluate,
    job_ensemble_pull,
    job_ensemble_render,
    job_member_psnr,
    regenerate_eval_diagnostics,
    regenerate_power_spectrum,
    training_curves_payload,
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
        # Star regime: starfull (reconstruct stars, hr target) vs starless
        # (erase them, clean target). Default starless (the current regime).
        starless = (request.form.get("mode", "starless").lower() != "starfull")
        regime = "starless" if starless else "starfull"
        job_id = REGISTRY.spawn(
            f"ensemble: evaluate {regime} on {num_images} test fields",
            target=lambda cap: job_ensemble_evaluate(
                cap, num_images=num_images, starless=starless),
        )
        return jsonify({"job_id": job_id})

    @app.route("/ensemble/member-psnr", methods=["POST"])
    def ensemble_member_psnr():
        """Refresh the members table's test PSNRs (asinh space). Fingerprint-
        cached per checkpoint — only changed/unscored members are evaluated."""
        job_id = REGISTRY.spawn(
            "ensemble: member test PSNR (changed members only)",
            target=job_member_psnr,
        )
        return jsonify({"job_id": job_id})

    @app.route("/ensemble/training-curves.json")
    def ensemble_training_curves_json():
        """Per-member PSNR + loss series (rollback-deduped) for the in-browser
        chart — registry-active members only, with depth + cached test PSNR
        for the coloring modes. Empty ``members`` → the client hides the card."""
        return jsonify({"members": training_curves_payload()})

    @app.route("/ensemble/power-spectrum.png")
    def ensemble_power_spectrum():
        """Serve the ensemble power-spectrum PNG. ``?fresh=1`` re-renders it from
        the cached per-field cubes (no full re-run / inference)."""
        out_png = os.path.join(_ensemble_out_dir(), "ensemble_power_spectrum.png")
        fresh = request.args.get("fresh", "").lower() in ("1", "true", "yes")
        color = request.args.get("color", "").lower()
        color_by = color if color in ("loss", "depth") else None
        if fresh or not os.path.isfile(out_png):
            if (regenerate_power_spectrum(color_by=color_by) is None
                    and not os.path.isfile(out_png)):
                abort(404)
        return send_file(out_png, mimetype="image/png", max_age=0)

    @app.route("/ensemble/evals.json")
    def ensemble_evals_json():
        """The Evaluations card's dataset: power-spectrum curves, diagnostic
        histograms, calibration stats and per-member loss/depth meta. The
        FRONTEND renders all figures from this JSON, so styling (member-line
        coloring, tab switches) never recomputes anything. ``?fresh=1``
        recomputes the payload from the cached cubes (one sweep, seconds)."""
        path = _evals_payload_path()
        fresh = request.args.get("fresh", "").lower() in ("1", "true", "yes")
        if fresh or not os.path.isfile(path):
            if compute_evaluation_payload() is None and not os.path.isfile(path):
                abort(404)
        return send_file(path, mimetype="application/json", max_age=0)

    @app.route("/ensemble/eval-plot/<plot>.png")
    def ensemble_eval_plot(plot: str):
        """Serve a pixel-level evaluation diagnostic (std-error /
        std-brightness / calibration). Renders lazily from the cached
        per-field cubes on first request; ``?fresh=1`` forces a re-render
        (all three figures share one pass, so they regenerate together)."""
        png_name = EVAL_DIAGNOSTIC_PNGS.get(plot)
        if png_name is None:
            abort(404)
        out_png = os.path.join(_ensemble_out_dir(), png_name)
        fresh = request.args.get("fresh", "").lower() in ("1", "true", "yes")
        if fresh or not os.path.isfile(out_png):
            if regenerate_eval_diagnostics() is None and not os.path.isfile(out_png):
                abort(404)
        return send_file(out_png, mimetype="image/png", max_age=0)

    @app.route("/ensemble/archive-member", methods=["POST"])
    def ensemble_archive_member():
        """Retire one member: zip → tracking campaign, registry tombstone,
        member dir deleted, cube cache purged. Reduces the ensemble."""
        name = (request.form.get("member") or "").strip()
        job_id = REGISTRY.spawn(
            f"ensemble: archive {name} → tracking",
            target=lambda cap: job_archive_member(cap, name=name),
        )
        return jsonify({"job_id": job_id})

    @app.route("/ensemble/pull", methods=["POST"])
    def ensemble_pull():
        job_id = REGISTRY.spawn(
            "ensemble: download from FASRC",
            target=job_ensemble_pull,
        )
        return jsonify({"job_id": job_id})
