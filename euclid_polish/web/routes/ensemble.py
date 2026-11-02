"""Ensemble routes: the registry's members and one model's detail, the
training jobs and curves, the production switch, archive / restore, the
universal models (payload, milestones, spectra, score, pull) and the
train-command preview. Every ``/ensemble/*`` route defaults ``mode`` to
STARFULL; errors under ``/ensemble/`` are JSON ``{error}``.
"""

from __future__ import annotations

from flask import jsonify, request

from euclid_polish import ensemble_registry
from euclid_polish.ensemble_registry import member_name
from euclid_polish.web import errors
from euclid_polish.web.fasrc_gate import requires_fasrc
from euclid_polish.web.fasrc_pipeline import TaskParamError
from euclid_polish.web.helpers import milestones
from euclid_polish.web.helpers.ensemble_viz import (
    ensemble_dir,
    gate_and_cube_member_names,
    job_archive_member,
    job_restore_member,
    ledger_universal_names,
    member_detail,
    members_payload,
    train_command_preview,
    training_curves_payload,
    training_jobs,
)
from euclid_polish.web.helpers.remote_curves import remote_training_curves
from euclid_polish.web.helpers.universal_models import (
    job_universal_pull,
    job_universal_score,
    universal_payload,
    universal_targets,
)
from euclid_polish.web.helpers.universal_spectra import spectra_map_payload, spectra_payload
from euclid_polish.web.jobs import REGISTRY


def _mode_starless(default: str = "starfull") -> bool:
    """Star regime for a request (``?mode=`` / form ``mode=``). starfull and
    starless artifacts are fully detached; the console sends ``mode=starfull``
    on every read (Models is starfull-only since 064fa13). STARFULL is the
    default (the production regime since 3aa5c86); starless is opt-in."""
    src = request.args if request.args.get("mode") is not None else request.form
    return (src.get("mode", default) or default).lower() == "starless"


def _bad(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


def _active_member_names() -> set[str]:
    """The active members of the ensemble registry (``member_NN``)."""
    return set(ensemble_registry.load_registry(ensemble_dir())["active"])


def _universal_member_names() -> set[str]:
    """The registered universal models (standalone, ``member_NN``)."""
    return set(ensemble_registry.load_registry(ensemble_dir())["universal"])


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
        Models › Train's presets, "Repeat last batch" and "Clone a past job"."""
        return jsonify({"jobs": training_jobs()})

    @app.route("/ensemble/production", methods=["POST"])
    def ensemble_production():
        """Make a registered universal model production (``model``: any
        ``universal:member_NNN`` / ``member_NNN`` / ``NNN`` spelling). Instant
        (no job)."""
        raw = (request.form.get("model") or "").strip()
        if not raw:
            return _bad("name the model (model=universal:member_NNN)")
        base = ensemble_dir()
        try:
            ensemble_registry.set_production(base, member_name(raw.removeprefix("universal:")))
        except ValueError as exc:
            return _bad(str(exc))
        name = ensemble_registry.production_name(base)
        return jsonify({"ok": True, "production": name,
                        "spec": f"universal:{name}" if name else "production"})

    # ---- training curves -------------------------------------------------- #

    @app.route("/ensemble/training-curves.json")
    def ensemble_training_curves_json():
        """Per-member training series (rollback-deduped) for the in-browser
        charts — registry-active members: joint + per-band PSNR, the loss
        series, gradient norm and step time, with the facets the lines colour
        by — then the members of recent training jobs that are not pulled yet
        (``remote``: training or finished on FASRC), read live from their
        event streams. Registered universal models count as local (never
        fetched); a remote entry of a model the job log created as a
        universal model carries ``universal: true``. Empty ``members`` → the
        client shows an empty state."""
        base = ensemble_dir()
        registry = ensemble_registry.load_registry(base)
        remote = remote_training_curves(
            [*registry["active"], *registry["universal"]],
            [str(t.get("name")) for t in registry["archived"]],
            universal_names=ledger_universal_names(base=base))
        return jsonify({"members": training_curves_payload() + remote})

    # ---- archive, restore, train preview ------------------------------------ #

    @app.route("/ensemble/archive-member", methods=["POST"])
    def ensemble_archive_member():
        """Retire one member: zip → tracking campaign, registry tombstone,
        member dir deleted, regime marked stale (its cached cubes go in the
        follow-up stale-cube purge). Reduces the ensemble. A registered
        universal model archives the same way (no regime goes stale). The
        name is validated (and must be active or a registered universal
        model) before the job starts: 400 JSON."""
        try:
            name = member_name(request.form.get("member") or "")
        except ValueError as exc:
            return _bad(str(exc))
        if name not in _active_member_names() and name not in _universal_member_names():
            return _bad(f"{name} is not an active ensemble member or a universal model")
        if ensemble_registry.production_name(ensemble_dir()) == name:
            return _bad(f"{name} is the production model — make another model production first")
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

    # ---- universal models ------------------------------------------------ #

    @app.route("/ensemble/universal.json")
    def ensemble_universal_json():
        """Models › Universal: every universal model the panel lists
        (single-image multi-knee) with its recipe, training time, hardware and
        test score, production's own score, and the universal models still
        only on FASRC (from the local job log). Local files only."""
        return jsonify(universal_payload(_mode_starless()))

    @app.route("/ensemble/universal/milestones.json")
    def ensemble_universal_milestones_json():
        """Models › Universal › Milestones: each long run's measured
        milestones (synthetic score per step, JWST resolution per band beside
        the reference models). Local files only."""
        return jsonify(milestones.report_payload())

    @app.route("/ensemble/universal/spectra.json")
    def ensemble_universal_spectra_json():
        """Models › Spectra: every local universal model's power spectrum
        against the truth's (coherence, transfer) and the direction structure
        of its error, from the records the score job writes. Local files
        only."""
        return jsonify(spectra_payload())

    @app.route("/ensemble/universal/spectra/<name>.json")
    def ensemble_universal_spectra_map_json(name: str):
        """One universal model's 2-D error-direction map (404 when the name is
        no local universal model or it has no spectra yet)."""
        try:
            name = member_name(name)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        payload = spectra_map_payload(name)
        if payload is None:
            return jsonify({"ok": False, "error": f"no spectra for {name} — not a local universal "
                                                  "model, or not computed yet"}), 404
        return jsonify(payload)

    @app.route("/ensemble/universal/score", methods=["POST"])
    def ensemble_universal_score():
        """Score the universal models (``members``: comma list, default every
        one the panel lists; ``force=1`` rescores current entries;
        ``spectra=1`` also re-runs the models whose spectra are missing or
        stale) on the test fields (local job, one at a time). Every model
        scored gets its spectra."""
        try:
            members = [member_name(token) for token in _form_list("members")]
        except ValueError as exc:
            return _bad(str(exc))
        local = universal_targets(ensemble_dir())
        unknown = [n for n in members if n not in local]
        if unknown:
            return _bad("not a local universal model with a checkpoint: " + ", ".join(unknown))
        force = _form_bool("force")
        spectra = _form_bool("spectra")
        what = ", ".join(members) if members else "every universal model"
        job, started = REGISTRY.spawn_exclusive(
            f"universal: score {what} on the test fields"
            + (" (forced)" if force else "") + (" + spectra" if spectra else ""),
            lambda cap: job_universal_score(cap, members=members or None, force=force,
                                            spectra=spectra),
            kind="universal-score")
        return jsonify({"ok": True, "job_id": job.job_id, "already_running": not started})

    @app.route("/ensemble/universal/pull", methods=["POST"])
    @requires_fasrc
    def ensemble_universal_pull():
        """Pull universal models from FASRC as standalone models (``members``:
        comma list, required; registered before the rsync, never activated).
        Active or archived names, and names a local gate or cube manifest
        reads, are refused up front (local job)."""
        try:
            members = list(dict.fromkeys(member_name(token) for token in _form_list("members")))
        except ValueError as exc:
            return _bad(str(exc))
        if not members:
            return _bad("name the universal models to pull (members)")
        registry = ensemble_registry.load_registry(ensemble_dir())
        tombstoned = {str(t.get("name")) for t in registry["archived"]}
        ensemble_names = gate_and_cube_member_names()
        for name in members:
            if name in registry["active"]:
                return _bad(f"{name} is an active ensemble member")
            if name in tombstoned:
                return _bad(f"{name} is archived (tombstoned)")
            if name in ensemble_names and name not in registry["universal"]:
                return _bad(f"{name} is an ensemble member (a local gate or cube manifest "
                            "reads it) — the ensemble pull restores it")
        job, started = REGISTRY.spawn_exclusive(
            f"universal: pull {', '.join(members)} from FASRC",
            lambda cap: job_universal_pull(cap, members=members),
            kind="universal-pull")
        return jsonify({"ok": True, "job_id": job.job_id, "already_running": not started})

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
