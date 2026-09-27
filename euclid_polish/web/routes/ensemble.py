"""Ensemble routes: member status, the joined members table and one member's
detail, training curves, the combiner variant registry (fit a named variant,
compare, promote), evaluation, PSNR vs knee, archive / restore, the FASRC
pull and the train-command preview. Every ``/ensemble/*`` route defaults
``mode`` to STARFULL; errors under ``/ensemble/`` are JSON ``{error}``.
"""

from __future__ import annotations

import json
import os

from flask import abort, jsonify, request, send_file

from euclid_polish import ensemble_registry
from euclid_polish.config import Config
from euclid_polish.ensemble_registry import member_name
from euclid_polish.eval.combiner import (
    ACTIVE_COMBINER_KINDS,
    SPATIAL_GATE_KIND,
    combiner_model_spec,
    normalize_model_kind,
)
from euclid_polish.eval.spatial_gate import MIX_SPACES
from euclid_polish.training.target_blur import validate_target_fwhm_arcsec
from euclid_polish.web import errors
from euclid_polish.web.fasrc_gate import requires_fasrc
from euclid_polish.web.fasrc_pipeline import TaskParamError
from euclid_polish.web.helpers.ensemble_viz import (
    _combiner_payload_path,
    _evals_payload_path,
    check_new_variant_name,
    combiner_variants,
    compare_reports,
    compute_combiner_payload,
    compute_evaluation_payload,
    ensemble_dir,
    ensemble_overview,
    ensemble_status,
    job_archive_member,
    job_combiner_compare,
    job_combiner_fit,
    job_combiner_promote,
    job_ensemble_evaluate,
    job_ensemble_pull,
    job_gate_variant_fit,
    job_knee_psnr,
    job_member_psnr,
    job_restore_member,
    knee_psnr_status,
    member_detail,
    members_payload,
    pixel_trace,
    read_compare_report,
    refresh_evaluation_diagnostics,
    train_command_preview,
    training_curves_payload,
    training_jobs,
    variant_dir,
)
from euclid_polish.web.jobs import REGISTRY
from euclid_polish.web.security import (
    fresh_requested,
    is_same_origin_request,
    refuse_cross_site_cache_fill,
)


def _mode_starless(default: str = "starfull") -> bool:
    """Star regime for a request (``?mode=`` / form ``mode=``). starfull and
    starless artifacts are fully detached; the client sends the active regime
    on every read so the page shows that regime's data. STARFULL is the
    default (the production regime since 3aa5c86); starless is opt-in."""
    src = request.args if request.args.get("mode") is not None else request.form
    return (src.get("mode", default) or default).lower() == "starless"


def _bad(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


def _active_member_names() -> set[str]:
    """The active members of the ensemble registry (``member_NN``)."""
    return set(ensemble_registry.load_registry(ensemble_dir())["active"])


def _form_bool(name: str, default: bool = False) -> bool:
    raw = request.form.get(name)
    if raw is None or raw == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _form_number(name: str, default, cast, *, lo=None, hi=None):
    """A form number within [lo, hi]; ValueError names the field."""
    raw = request.form.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = cast(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ValueError(f"{name} must be within [{lo}, {hi}]")
    return value


def _form_list(name: str) -> list[str]:
    values = request.form.getlist(name)
    raw = ",".join(values) if len(values) > 1 else (request.form.get(name) or "")
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def register(app):
    errors.json_errors_for(app, "/ensemble/")

    @app.route("/ensemble/status.json")
    def ensemble_status_json():
        """Everything the members table + summary render from — the JSON twin of
        the classic page's render context (members, archived, eval summary,
        data presence). Consumed by the React console. ``?mode=`` selects which
        regime's eval summary + staleness to report (default starfull)."""
        return jsonify(ensemble_status(_mode_starless()))

    @app.route("/ensemble/overview.json")
    def ensemble_overview_json():
        """Overview tab: headline numbers with their definitions + staleness."""
        return jsonify(ensemble_overview(_mode_starless()))

    @app.route("/ensemble/members.json")
    def ensemble_members_json():
        """Members tab: one joined row per active member of the regime."""
        return jsonify(members_payload(_mode_starless()))

    @app.route("/ensemble/member/<name>.json")
    def ensemble_member_json(name: str):
        """One member's inspector payload (active or archived)."""
        try:
            detail = member_detail(name)
        except ValueError as exc:
            return _bad(str(exc))
        if detail is None:
            return _bad(f"unknown member {name}", 404)
        return jsonify(detail)

    @app.route("/ensemble/training-jobs.json")
    def ensemble_training_jobs_json():
        """Every ensemble_train submission in the local job log (newest first):
        the Train tab's presets, "repeat last batch" and "clone a past job"."""
        return jsonify({"jobs": training_jobs()})

    @app.route("/ensemble/evaluate", methods=["POST"])
    def ensemble_evaluate():
        try:
            num_images = max(1, int(request.form.get("num_images", 100) or 100))
        except (TypeError, ValueError):
            num_images = 100
        # Star regime: starfull (reconstruct stars, hr target — the default)
        # vs starless (erase them, clean target; opt-in).
        starless = _mode_starless()
        force = _form_bool("force")
        try:
            target_fwhm = validate_target_fwhm_arcsec(
                float(request.form.get("target_psf_fwhm_arcsec",
                                       Config.TARGET_PSF_FWHM_ARCSEC)))
        except (TypeError, ValueError):
            abort(400, description="invalid target_psf_fwhm_arcsec")
        regime = "starless" if starless else "starfull"
        job_id = REGISTRY.spawn(
            f"ensemble: evaluate {regime} on {num_images} test fields"
            + (" (forced re-inference)" if force else ""),
            target=lambda cap: job_ensemble_evaluate(
                cap, num_images=num_images, starless=starless, force=force,
                target_fwhm_arcsec=target_fwhm),
        )
        return jsonify({"job_id": job_id})

    @app.route("/ensemble/combiner/fit", methods=["POST"])
    def ensemble_combiner_fit():
        """Fit an RBF combiner for the requested star regime locally on the
        validate split, in place. The spatial gate — production — is refused
        here (it would be overwritten without a backup): fit it as a named
        variant (``/ensemble/combiners/fit``) and promote that
        (``/ensemble/combiners/promote``, which backs production up)."""
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
        if model_kind == SPATIAL_GATE_KIND:
            return _bad("the spatial gate is production: fit a named variant "
                        "(POST /ensemble/combiners/fit) and promote it "
                        "(POST /ensemble/combiners/promote, which backs production up)")
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

    # ---- combiner variants: registry, compare, fit, promote -------------- #

    @app.route("/ensemble/combiners.json")
    def ensemble_combiners_json():
        """The combiner variant registry of a regime + the saved compare reports."""
        starless = _mode_starless()
        return jsonify({**combiner_variants(starless), "reports": compare_reports(starless)})

    @app.route("/ensemble/combiners/compare.json")
    def ensemble_compare_report_json():
        """One saved compare report (``?report=<id>``; default the latest)."""
        try:
            report = read_compare_report(_mode_starless(), request.args.get("report"))
        except ValueError as exc:
            return _bad(str(exc))
        if report is None:
            return _bad("no compare report yet — run a compare", 404)
        return jsonify(report)

    @app.route("/ensemble/combiners/compare", methods=["POST"])
    def ensemble_combiners_compare():
        """Score gate variants on the test cubes + blackout copies (local job)."""
        starless = _mode_starless()
        gates = _form_list("gates")
        try:
            for gate in gates:
                variant_dir(starless, gate)
            blackout_fields = _form_number("blackout_fields", 40, int, lo=0, hi=400)
            seed = _form_number("seed", 0, int, lo=0, hi=2**31 - 1)
        except ValueError as exc:
            return _bad(str(exc))
        include_rbf = _form_bool("include_rbf", True)
        knee = _form_bool("knee", True)
        regime = "starless" if starless else "starfull"
        what = ", ".join(gates) if gates else "every applicable gate"
        job_id = REGISTRY.spawn(
            f"combiner: compare {regime} ({what})",
            target=lambda cap: job_combiner_compare(
                cap, starless=starless, gates=gates or None,
                blackout_fields=blackout_fields, seed=seed,
                include_rbf=include_rbf, knee=knee),
            kind="ensemble-compare")
        return jsonify({"ok": True, "job_id": job_id})

    @app.route("/ensemble/combiners/fit", methods=["POST"])
    def ensemble_combiners_fit():
        """Fit a NAMED spatial-gate variant (never production; local job)."""
        starless = _mode_starless()
        out_name = (request.form.get("out_name") or "").strip()
        if out_name and not out_name.startswith("spatial_gate_"):
            out_name = f"spatial_gate_{out_name}"
        try:
            overwrite = _form_bool("overwrite")
            check_new_variant_name(starless, out_name, overwrite=overwrite)
            mix = (request.form.get("mix_space") or "linear").strip()
            if mix not in MIX_SPACES:
                raise ValueError(f"mix_space must be one of {', '.join(MIX_SPACES)}")
            loss_knees = (request.form.get("loss_knees") or "all").strip()
            knobs = {
                "width": _form_number("width", 32, int, lo=4, hi=256),
                "steps": _form_number("steps", 2000, int, lo=1, hi=200_000),
                "batch_size": _form_number("batch_size", 8, int, lo=1, hi=64),
                "crop": _form_number("crop", 192, int, lo=32, hi=1024),
                "learning_rate": _form_number("learning_rate", 2e-3, float, lo=1e-7, hi=1.0),
                "eval_every": _form_number("eval_every", 250, int, lo=1, hi=100_000),
                "holdout": _form_number("holdout", 15, int, lo=1, hi=1000),
                "blackout_fields": _form_number("blackout_fields", 40, int, lo=0, hi=400),
                "seed": _form_number("seed", 0, int, lo=0, hi=2**31 - 1),
                "num_images": _form_number("num_images", 100, int, lo=2, hi=2000),
            }
            target_fwhm = validate_target_fwhm_arcsec(
                _form_number("target_psf_fwhm_arcsec", Config.TARGET_PSF_FWHM_ARCSEC, float))
            members = _form_list("members")
            for token in members:
                member_name(token)
        except ValueError as exc:
            return _bad(str(exc))
        use_lr = _form_bool("use_lr")
        compare_after = _form_bool("compare_after", True)
        regime = "starless" if starless else "starfull"
        job_id = REGISTRY.spawn(
            f"combiner: fit gate variant {out_name} ({regime}, {knobs['steps']} steps)",
            target=lambda cap: job_gate_variant_fit(
                cap, starless=starless, out_name=out_name, use_lr=use_lr,
                members=members or None, loss_knees=loss_knees, mix_space=mix,
                target_fwhm_arcsec=target_fwhm, overwrite=overwrite,
                compare_after=compare_after, **knobs),
            kind="gate-fit")
        return jsonify({"ok": True, "job_id": job_id, "variant": out_name})

    @app.route("/ensemble/combiners/promote", methods=["POST"])
    def ensemble_combiners_promote():
        """Promote a variant to production (backs the current one up; job)."""
        starless = _mode_starless()
        variant = (request.form.get("variant") or "").strip()
        try:
            name = os.path.basename(variant_dir(starless, variant))
        except ValueError as exc:
            return _bad(str(exc))
        force = _form_bool("force")
        regime = "starless" if starless else "starfull"
        job_id = REGISTRY.spawn(
            f"combiner: promote {name} to production ({regime})",
            target=lambda cap: job_combiner_promote(
                cap, starless=starless, variant=name, force=force),
            kind="gate-promote")
        return jsonify({"ok": True, "job_id": job_id})

    # ---- knee, member PSNR, curves, diagnostics --------------------------- #

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
        """Per-member training series (rollback-deduped) for the in-browser
        charts — registry-active members only: joint + per-band PSNR, the
        loss series, gradient norm and step time, with the facets the lines
        colour by. Empty ``members`` → the client shows an empty state."""
        return jsonify({"members": training_curves_payload()})

    @app.route("/ensemble/evals.json")
    def ensemble_evals_json():
        """The Evaluations card's dataset: power-spectrum curves, diagnostic
        histograms, calibration stats and per-member loss/depth meta. The
        FRONTEND renders all figures from this JSON, so styling (member-line
        coloring, tab switches) never recomputes anything. ``?fresh=1``
        recomputes the payload from the cached cubes (one sweep, seconds) —
        for a same-origin request only. A cross-site request gets the cached
        payload as it is (no diagnostics upgrade), or 404 when none exists."""
        starless = _mode_starless()
        path = _evals_payload_path(starless)
        refuse_cross_site_cache_fill(path)
        if not is_same_origin_request():
            return send_file(path, mimetype="application/json", max_age=0)
        fresh = fresh_requested()          # same-origin only (a GET recomputes)
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
            abort(404, description="no evaluation cached for this regime — "
                                   "evaluate the ensemble first")
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

    # ---- members: archive, restore, pull, train preview ------------------- #

    @app.route("/ensemble/archive-member", methods=["POST"])
    def ensemble_archive_member():
        """Retire one member: zip → tracking campaign, registry tombstone,
        member dir deleted, cube cache purged. Reduces the ensemble. The name
        is validated (and must be active) before the job starts: 400 JSON."""
        try:
            name = member_name(request.form.get("member") or "")
        except ValueError as exc:
            return _bad(str(exc))
        if name not in _active_member_names():
            return _bad(f"{name} is not an active ensemble member")
        job_id = REGISTRY.spawn(
            f"ensemble: archive {name} → tracking",
            target=lambda cap: job_archive_member(cap, name=name),
        )
        return jsonify({"ok": True, "job_id": job_id})

    @app.route("/ensemble/restore-member", methods=["POST"])
    def ensemble_restore_member():
        """Restore an archived member from its tracking zip (local job)."""
        try:
            name = member_name(request.form.get("member") or "")
        except ValueError as exc:
            return _bad(str(exc))
        job_id = REGISTRY.spawn(
            f"ensemble: restore {name} from its archive zip",
            target=lambda cap: job_restore_member(cap, name=name),
            kind="member-restore")
        return jsonify({"ok": True, "job_id": job_id})

    @app.route("/ensemble/pull", methods=["POST"])
    @requires_fasrc
    def ensemble_pull():
        """Download changed members from FASRC (local job). ``members`` limits
        it to those members; ``dry_run=1`` only probes what changed."""
        try:
            members = [member_name(token) for token in _form_list("members")]
        except ValueError as exc:
            return _bad(str(exc))
        dry_run = _form_bool("dry_run")
        label = ("ensemble: check FASRC for changed members" if dry_run
                 else f"ensemble: download {', '.join(members)} from FASRC" if members
                 else "ensemble: download from FASRC")
        job_id = REGISTRY.spawn(
            label,
            target=lambda cap: job_ensemble_pull(
                cap, members=members or None, dry_run=dry_run),
        )
        return jsonify({"job_id": job_id})

    @app.route("/ensemble/train/preview", methods=["POST"])
    def ensemble_train_preview():
        """The member names + ``train_ensemble.py`` command an ensemble_train
        submit with this form would run (nothing reaches FASRC)."""
        form = request.form.to_dict()
        members = request.form.getlist("members")
        if len(members) > 1:
            form["members"] = ",".join(members)
        try:
            return jsonify(train_command_preview(form))
        except (ValueError, TaskParamError) as exc:
            return _bad(str(exc))
