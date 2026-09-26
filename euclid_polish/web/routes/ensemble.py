"""Ensemble routes: view member status, render a field's disagreement
(hallucination cross-check), and evaluate the ensemble on the held-out test set.
"""

from __future__ import annotations

import json
import os

from flask import abort, jsonify, request, send_file

from euclid_polish.config import Config
from euclid_polish.eval.combiner import (
    ACTIVE_COMBINER_KINDS,
    combiner_model_spec,
    normalize_model_kind,
)
from euclid_polish.training.target_blur import validate_target_fwhm_arcsec
from euclid_polish.web.fasrc_gate import requires_fasrc
from euclid_polish.web.helpers.ensemble_viz import (
    _combiner_payload_path,
    _evals_payload_path,
    compute_combiner_payload,
    compute_evaluation_payload,
    ensemble_status,
    job_archive_member,
    job_combiner_fit,
    job_ensemble_evaluate,
    job_ensemble_pull,
    job_knee_psnr,
    job_member_psnr,
    knee_psnr_status,
    pixel_trace,
    refresh_evaluation_diagnostics,
    training_curves_payload,
)
from euclid_polish.web.jobs import REGISTRY


def _mode_starless(default: str = "starfull") -> bool:
    """Star regime for a request (``?mode=`` / form ``mode=``). starfull and
    starless artifacts are fully detached; the client sends the active regime
    on every read so the page shows that regime's data. STARFULL is the
    default (the production regime since 3aa5c86); starless is opt-in."""
    src = request.args if request.args.get("mode") is not None else request.form
    return (src.get("mode", default) or default).lower() == "starless"


def register(app):

    @app.route("/ensemble/status.json")
    def ensemble_status_json():
        """Everything the members table + summary render from — the JSON twin of
        the classic page's render context (members, archived, eval summary,
        data presence). Consumed by the React console. ``?mode=`` selects which
        regime's eval summary + staleness to report (default starfull)."""
        return jsonify(ensemble_status(_mode_starless()))

    @app.route("/ensemble/evaluate", methods=["POST"])
    def ensemble_evaluate():
        try:
            num_images = max(1, int(request.form.get("num_images", 100) or 100))
        except (TypeError, ValueError):
            num_images = 100
        # Star regime: starfull (reconstruct stars, hr target — the default)
        # vs starless (erase them, clean target; opt-in).
        starless = _mode_starless()
        try:
            target_fwhm = validate_target_fwhm_arcsec(
                float(request.form.get("target_psf_fwhm_arcsec",
                                       Config.TARGET_PSF_FWHM_ARCSEC)))
        except (TypeError, ValueError):
            abort(400, description="invalid target_psf_fwhm_arcsec")
        regime = "starless" if starless else "starfull"
        job_id = REGISTRY.spawn(
            f"ensemble: evaluate {regime} on {num_images} test fields",
            target=lambda cap: job_ensemble_evaluate(
                cap, num_images=num_images, starless=starless,
                target_fwhm_arcsec=target_fwhm),
        )
        return jsonify({"job_id": job_id})

    @app.route("/ensemble/combiner/fit", methods=["POST"])
    def ensemble_combiner_fit():
        """Fit the combiner for the requested star regime locally on the
        validate split. Available in both regimes — starfull fuses star
        reconstructions, starless fuses the star-erasing members."""
        starless = _mode_starless()
        try:
            target_fwhm = validate_target_fwhm_arcsec(
                float(request.form.get("target_psf_fwhm_arcsec",
                                       Config.TARGET_PSF_FWHM_ARCSEC)))
        except (TypeError, ValueError):
            abort(400, description="invalid target_psf_fwhm_arcsec")
        try:
            num_images = max(1, int(request.form.get("num_images", 100) or 100))
        except (TypeError, ValueError):
            num_images = 100
        gate_members = [token.strip() for token in
                        str(request.form.get("members", "") or "").split(",")
                        if token.strip()]
        if any(not token.isdigit() for token in gate_members):
            abort(400, description="members must be comma-separated member numbers")
        raw_min_usage = request.form.get("min_usage")
        try:
            min_usage = (None if raw_min_usage in (None, "")
                         else max(0.0, float(raw_min_usage)))
        except (TypeError, ValueError):
            min_usage = None
        try:
            model_kind = normalize_model_kind(
                request.form.get("model_kind"))
        except ValueError:
            abort(400)
        if model_kind not in ACTIVE_COMBINER_KINDS:
            abort(400)
        spec = combiner_model_spec(model_kind)
        raw_kernels = request.form.get("n_kernels")
        try:
            n_kernels = max(
                2, int(raw_kernels if raw_kernels not in (None, "")
                       else spec.default_kernels))
        except (TypeError, ValueError):
            n_kernels = spec.default_kernels
        regime = "starless" if starless else "starfull"
        model_label = (spec.label if spec.default_kernels <= 0
                       else f"{spec.label} K={n_kernels}")
        job_id = REGISTRY.spawn(
            f"combiner: fit {regime} on validate ({num_images} fields, {model_label})",
            target=lambda cap: job_combiner_fit(
                cap, num_images=num_images, n_kernels=n_kernels,
                min_usage=min_usage,
                starless=starless,
                model_kind=model_kind,
                gate_members=gate_members or None,
                target_fwhm_arcsec=target_fwhm),
        )
        return jsonify({"job_id": job_id})

    @app.route("/ensemble/combiner.json")
    def ensemble_combiner_json():
        """The Combiner card's dataset for a regime (``?mode=``): per-band
        effective-weight curves, survivors, val loss and per-member meta
        (loss/depth/PSNR — the facets the gate plot colors by). Always recomputed
        from the saved combiner (cheap: reads the npz + member origins, no
        inference) so the member meta stays current; 404 before any fit."""
        starless = _mode_starless()
        try:
            model_kind = normalize_model_kind(
                request.args.get("model_kind"))
        except ValueError:
            abort(400)
        if model_kind not in ACTIVE_COMBINER_KINDS:
            abort(400)
        path = _combiner_payload_path(starless, model_kind)
        if compute_combiner_payload(starless, model_kind=model_kind) is None:
            return jsonify({
                "available": False,
                "stale": False,
                "kind": model_kind,
                "reason": "no current combiner fitted for this model",
                "member_labels": [], "members": [], "band_names": [],
                "eff_weights": {}, "feature_grid": {}, "surviving": {},
            })
        return send_file(path, mimetype="application/json", max_age=0)

    @app.route("/ensemble/knee-psnr.json")
    def ensemble_knee_psnr_json():
        """PSNR-vs-knee curves + integrated PSNR for every model of a regime
        (``?mode=``), flagged ``stale`` when the cubes or combiners changed."""
        return jsonify(knee_psnr_status(_mode_starless()))

    @app.route("/ensemble/knee-psnr", methods=["POST"])
    def ensemble_knee_psnr_compute():
        starless = _mode_starless()
        regime = "starless" if starless else "starfull"
        job_id = REGISTRY.spawn(
            f"ensemble: PSNR vs knee ({regime})",
            target=lambda cap: job_knee_psnr(cap, starless=starless),
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
                with open(path) as f:
                    cached = json.load(f)
                    fresh = "coherence" not in cached
                    feature_error = cached.get("combiner_feature_error") or {}
                    std_err = cached.get("std_err") or {}
                    needs_diagnostics = (
                        "axes" not in feature_error
                        or not std_err.get("adaptive_range", False)
                    )
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
        if model_kind and model_kind not in ("ensemble_mean", *ACTIVE_COMBINER_KINDS):
            abort(400)
        if (diag == "combiner_feature_error"
                and model_kind not in ("ensemble_mean", *ACTIVE_COMBINER_KINDS)):
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
    @requires_fasrc
    def ensemble_pull():
        job_id = REGISTRY.spawn(
            "ensemble: download from FASRC",
            target=job_ensemble_pull,
        )
        return jsonify({"job_id": job_id})
