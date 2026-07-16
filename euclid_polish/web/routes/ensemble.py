"""Ensemble routes: view member status, render a field's disagreement
(hallucination cross-check), and evaluate the ensemble on the held-out test set.
"""

from __future__ import annotations

import os

from flask import abort, jsonify, render_template, request, send_file

from euclid_polish.eval.combiner import COMBINER_MODELS
from euclid_polish.web.helpers.ensemble_viz import (
    EVAL_DIAGNOSTIC_PNGS,
    _combined_payload_path,
    _combiner_payload_path,
    _ensemble_regime_dir,
    _evals_payload_path,
    compute_combined_combiner_payload,
    compute_combiner_payload,
    compute_evaluation_payload,
    ensemble_status,
    job_archive_member,
    job_combined_combiner_fit,
    job_combiner_fit,
    job_ensemble_evaluate,
    job_ensemble_pull,
    job_ensemble_render,
    job_member_psnr,
    pixel_trace,
    refresh_evaluation_diagnostics,
    regenerate_eval_diagnostics,
    regenerate_power_spectrum,
    training_curves_payload,
)
from euclid_polish.web.jobs import REGISTRY


def _mode_starless(default: str = "starless") -> bool:
    """Star regime for a request (``?mode=`` / form ``mode=``). starfull and
    starless artifacts are fully detached; the client sends the active regime
    on every read so the page shows that regime's data."""
    src = request.args if request.args.get("mode") is not None else request.form
    return (src.get("mode", default) or default).lower() != "starfull"


def register(app):

    @app.route("/ensemble")
    def ensemble_page():
        return render_template("ensemble.html", **ensemble_status())

    @app.route("/ensemble/status.json")
    def ensemble_status_json():
        """Everything the members table + summary render from — the JSON twin of
        the classic page's render context (members, archived, eval summary,
        data presence). Consumed by the React console. ``?mode=`` selects which
        regime's eval summary + staleness to report (mode-specific badge)."""
        mode = request.args.get("mode")
        starless = None if mode is None else (mode.lower() != "starfull")
        return jsonify(ensemble_status(starless))

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

    @app.route("/ensemble/combiner/fit", methods=["POST"])
    def ensemble_combiner_fit():
        """Fit the combiner for the requested star regime locally on the
        validate split. Available in both regimes — starfull fuses star
        reconstructions, starless fuses the star-erasing members."""
        starless = _mode_starless(default="starfull")
        try:
            num_images = max(1, int(request.form.get("num_images", 100) or 100))
        except (TypeError, ValueError):
            num_images = 100
        try:
            n_kernels = max(2, min(64, int(request.form.get("n_kernels", 12) or 12)))
        except (TypeError, ValueError):
            n_kernels = 12
        raw_min_usage = request.form.get("min_usage")
        try:
            min_usage = (None if raw_min_usage in (None, "")
                         else max(0.0, float(raw_min_usage)))
        except (TypeError, ValueError):
            min_usage = None
        from euclid_polish.eval.combiner import (
            DEFAULT_N_KERNELS,
            combiner_model_spec,
            normalize_model_kind,
        )
        try:
            model_kind = normalize_model_kind(request.form.get("model_kind"))
        except ValueError:
            model_kind = "rbf_gate"
        spec = combiner_model_spec(model_kind)
        if spec.feature_names is not None and n_kernels == DEFAULT_N_KERNELS:
            n_kernels = spec.default_kernels
        regime = "starless" if starless else "starfull"
        model_label = f"{spec.label} K={n_kernels}"
        job_id = REGISTRY.spawn(
            f"combiner: fit {regime} on validate ({num_images} fields, {model_label})",
            target=lambda cap: job_combiner_fit(
                cap, num_images=num_images, n_kernels=n_kernels,
                min_usage=min_usage,
                starless=starless,
                model_kind=model_kind),
        )
        return jsonify({"job_id": job_id})

    @app.route("/ensemble/combiner.json")
    def ensemble_combiner_json():
        """The Combiner card's dataset for a regime (``?mode=``): per-band
        effective-weight curves, survivors, val loss and per-member meta
        (loss/depth/PSNR — the facets the gate plot colors by). Always recomputed
        from the saved combiner (cheap: reads the npz + member origins, no
        inference) so the member meta stays current; 404 before any fit."""
        starless = _mode_starless(default="starfull")
        from euclid_polish.eval.combiner import normalize_model_kind
        try:
            model_kind = normalize_model_kind(request.args.get("model_kind"))
        except ValueError:
            model_kind = "rbf_gate"
        path = _combiner_payload_path(starless, model_kind)
        if compute_combiner_payload(starless, model_kind=model_kind) is None and not os.path.isfile(path):
            abort(404)
        return send_file(path, mimetype="application/json", max_age=0)

    @app.route("/ensemble/combined-combiner/fit", methods=["POST"])
    def ensemble_combined_combiner_fit():
        """Fit the selected target's additive all-member experimental gate."""
        starless = _mode_starless(default="starfull")
        try:
            num_images = max(1, int(request.form.get("num_images", 100) or 100))
        except (TypeError, ValueError):
            num_images = 100
        try:
            n_kernels = max(2, min(32, int(request.form.get("n_kernels", 12) or 12)))
        except (TypeError, ValueError):
            n_kernels = 12
        raw_min_usage = request.form.get("min_usage")
        try:
            min_usage = (None if raw_min_usage in (None, "")
                         else max(0.0, float(raw_min_usage)))
        except (TypeError, ValueError):
            min_usage = None
        from euclid_polish.eval.combiner import normalize_model_kind
        try:
            model_kind = normalize_model_kind(request.form.get("model_kind"))
        except ValueError:
            model_kind = "rbf_gate"
        regime = "starless" if starless else "starfull"
        model_label = f"RBF K={n_kernels}"
        job_id = REGISTRY.spawn(
            f"combined combiner: fit {regime} on validate ({num_images} fields, {model_label})",
            target=lambda cap: job_combined_combiner_fit(
                cap, num_images=num_images, n_kernels=n_kernels,
                min_usage=min_usage, starless=starless, model_kind=model_kind),
        )
        return jsonify({"job_id": job_id})

    @app.route("/ensemble/combined-combiner.json")
    def ensemble_combined_combiner_json():
        starless = _mode_starless(default="starfull")
        payload = compute_combined_combiner_payload(starless)
        path = _combined_payload_path(starless)
        if not os.path.isfile(path):
            return jsonify(payload)
        return send_file(path, mimetype="application/json", max_age=0)

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
        starless = _mode_starless()
        out_png = os.path.join(_ensemble_regime_dir(starless),
                               "ensemble_power_spectrum.png")
        fresh = request.args.get("fresh", "").lower() in ("1", "true", "yes")
        color = request.args.get("color", "").lower()
        color_by = color if color in ("loss", "depth", "knee") else None
        if ((fresh or not os.path.isfile(out_png))
                and regenerate_power_spectrum(starless, color_by=color_by) is None
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
        starless = _mode_starless()
        path = _evals_payload_path(starless)
        fresh = request.args.get("fresh", "").lower() in ("1", "true", "yes")
        needs_diagnostics = False
        if os.path.isfile(path) and not fresh:
            # Older payloads predate one or more cache-derived diagnostics.
            # Rebuild once from existing cubes, with no model inference or
            # recaching; this also refreshes the per-model trace sidecar.
            try:
                import json
                with open(path) as f:
                    cached = json.load(f)
                    fresh = "coherence" not in cached
                    feature_error = cached.get("combiner_feature_error") or {}
                    needs_diagnostics = "axes" not in feature_error
            except (OSError, ValueError):
                fresh = True
        if (needs_diagnostics
                and refresh_evaluation_diagnostics(starless) is None):
            fresh = True
        if ((fresh or not os.path.isfile(path))
                and compute_evaluation_payload(starless) is None
                and not os.path.isfile(path)):
            abort(404)
        return send_file(path, mimetype="application/json", max_age=0)

    @app.route("/ensemble/pixel-trace.json")
    def ensemble_pixel_trace():
        """Back-trace a diagnostic heatmap cell to real image stamps.

        ``?mode=&diag=std_err|bright_std|combiner_feature_error&model=&axis=&i=&j=``
        → up to a handful of
        VIS zoom stamps (HR / ensemble-mean SR / cross-member std, electrons) of
        the actual pixels that fell into the clicked cell, each with the exact
        per-pixel σ / |error| / brightness so the user can see WHY it landed
        there. Empty ``stamps`` when nothing was sampled for that cell."""
        starless = _mode_starless()
        diag = (request.args.get("diag") or "").strip()
        if diag not in ("std_err", "bright_std", "combiner_feature_error"):
            abort(404)
        model_kind = (request.args.get("model") or "").strip()
        axis_mode = (request.args.get("axis") or "").strip()
        if model_kind and model_kind not in ("ensemble_mean", *COMBINER_MODELS):
            abort(400)
        if (diag == "combiner_feature_error"
                and model_kind not in ("ensemble_mean", *COMBINER_MODELS)):
            abort(400)
        if diag == "combiner_feature_error" and axis_mode not in (
                "mean_std", "min_max"):
            abort(400)
        try:
            i = int(request.args.get("i", ""))
            j = int(request.args.get("j", ""))
        except (TypeError, ValueError):
            abort(400)
        return jsonify(pixel_trace(starless, diag, i, j,
                                   model_kind=model_kind or None,
                                   axis_mode=axis_mode or None))

    @app.route("/ensemble/eval-plot/<plot>.png")
    def ensemble_eval_plot(plot: str):
        """Serve a pixel-level evaluation diagnostic (std-error /
        std-brightness / calibration). Renders lazily from the cached
        per-field cubes on first request; ``?fresh=1`` forces a re-render
        (all three figures share one pass, so they regenerate together)."""
        png_name = EVAL_DIAGNOSTIC_PNGS.get(plot)
        if png_name is None:
            abort(404)
        starless = _mode_starless()
        out_png = os.path.join(_ensemble_regime_dir(starless), png_name)
        fresh = request.args.get("fresh", "").lower() in ("1", "true", "yes")
        if ((fresh or not os.path.isfile(out_png))
                and regenerate_eval_diagnostics(starless) is None
                and not os.path.isfile(out_png)):
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
