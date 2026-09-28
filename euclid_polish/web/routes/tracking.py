"""tracking routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

import os
import re
from typing import Any

from flask import jsonify, request

from euclid_polish.config import Config
from euclid_polish.observability.training_log import TrainingLog
from euclid_polish.tracking import TrackingError, dirty_warning
from euclid_polish.tracking import default_store as tracking_default_store
from euclid_polish.tracking import sync as tracking_sync
from euclid_polish.tracking import timetravel as tracking_timetravel
from euclid_polish.training.log_plot import plot_training_log
from euclid_polish.web import errors, fasrc_config, fasrc_jobs
from euclid_polish.web.fasrc_gate import requires_fasrc
from euclid_polish.web.helpers.paths import (
    _abort_json,
    _resolve_trackable_ckpt,
    _resolve_trackable_file,
)
from euclid_polish.web.remote import STATE

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

#: FITS / image backup kinds a time travel can start from → their list in
#: ``TrackingStore.backups_in``.
_FILE_BACKUP_KINDS = {"fits": "fits", "image": "images"}

#: Page size bounds of ``GET /api/tracking/jobs``.
_JOBS_DEFAULT_LIMIT = 50
_JOBS_MAX_LIMIT = 500


def _compact_job(record: dict[str, Any]) -> dict[str, Any]:
    """A tracking job record without its embedded payload blobs (population
    calibration / star-prior JSON, ~200 KB each): ``params`` compacted,
    ``params_omitted`` = ``{key: size}``."""
    out = dict(record)
    params = record.get("params")
    kept, omitted = fasrc_jobs.compact_params(params if isinstance(params, dict) else {})
    out["params"] = kept
    out["params_omitted"] = omitted
    return out


def _job_matches(record: dict[str, Any], needle: str) -> bool:
    hay = " ".join(str(record.get(k) or "") for k in ("jobid", "label", "step_id", "logged_at"))
    return needle in hay.lower()


#: A time-travel sandbox id is the commit's short hash (``git rev-parse
#: --short``): hex only, so it can never name the sandbox root ('' / '.'),
#: a parent or a nested path.
_SANDBOX_SHORT = re.compile(r"^[0-9a-f]{4,40}$")


def _sandbox_short() -> str:
    """The form's ``short``, refused (400 JSON) unless it is a well-formed id
    of an existing sandbox — before :mod:`timetravel` joins it into a path
    (an empty id used to make ``remove`` rmtree every sandbox)."""
    short = (request.form.get("short") or "").strip()
    if not _SANDBOX_SHORT.fullmatch(short):
        _abort_json(400, f"invalid time-travel sandbox id {short!r}")
    known = {str(meta.get("short") or "") for meta in tracking_timetravel.list_sandboxes()}
    if short not in known:
        _abort_json(400, f"no time-travel sandbox {short!r}")
    return short


def register(app):

    # =========================================================================
    # Tracking tab — the experiment "lab notebook". Titled campaigns collect
    # model/FITS/image backups (each with a comment + git-commit stamp), a
    # markdown log, and every FASRC job's parameters. The local store is
    # gitignored and mirrored to persistent holylabs storage on each backup.
    # See euclid_polish.tracking.
    # =========================================================================

    def _tracking_remote_dir() -> str:
        cfg = fasrc_config.load()
        return tracking_sync.remote_tracking_dir(
            cfg.repo_path, cfg.tracking_remote_dir,
        )

    def _tracking_try_sync(store) -> dict[str, Any]:
        """Best-effort push of the store → holylabs; never raises."""
        try:
            return tracking_sync.push(
                STATE.ssh, _tracking_remote_dir(), local_root=store.root,
            )
        except Exception as e:  # pragma: no cover - defensive
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def _tracking_state() -> dict[str, Any]:
        store = tracking_default_store()
        listing = store.list_campaigns()
        # Enrich each archived campaign with its model backups so the UI can
        # offer a per-model ⏱ time-travel button without an extra round-trip.
        for c in listing["archived"]:
            try:
                c["models"] = store.backups_in(c.get("_dir")).get("models", [])
            except Exception:
                c["models"] = []
        # The job records themselves are paged by GET /api/tracking/jobs:
        # each embeds the submit params, which can carry ~200 KB of
        # calibration JSON, so the state only reports how many there are.
        return {
            "active":        listing["active"],
            "archived":      listing["archived"],
            "backups":       store.list_backups(),
            "jobs_count":    store.count_fasrc_jobs(),
            "unassigned_count": store.count_fasrc_jobs("unassigned"),
            "log_md":        store.read_log() if listing["active"] else "",
            "remote_dir":    _tracking_remote_dir(),
            "tracking_dir":  store.root,
            "ssh_connected": bool(STATE.ssh and STATE.ssh.is_connected()),
            "sandboxes":     tracking_timetravel.list_sandboxes(),
        }

    @app.route("/api/tracking/state")
    def api_tracking_state():
        return jsonify(_tracking_state())

    @app.route("/api/tracking/jobs")
    def api_tracking_jobs():
        """One page of a campaign's FASRC job records, newest first, without
        the embedded payload blobs. ``campaign`` = ``current`` (default), an
        archived campaign's dir, or ``unassigned``; ``q`` filters on jobid,
        label, step and time; ``offset``/``limit`` (≤ 500) page. ``ids=1``
        answers every job id of the campaign instead (unpaged, deduped,
        newest first; Runs › History's campaign filter)."""
        store = tracking_default_store()
        campaign = (request.args.get("campaign") or "current").strip()
        try:
            records = store.read_fasrc_jobs(campaign)
        except TrackingError as e:
            return jsonify({"ok": False, "error": str(e)}), 404
        if request.args.get("ids") in ("1", "true", "yes"):
            jobids = list(dict.fromkeys(
                str(r.get("jobid")) for r in records if r.get("jobid") not in (None, "")))
            return jsonify({"ok": True, "campaign": campaign, "total": len(jobids),
                            "jobids": jobids})
        needle = (request.args.get("q") or "").strip().lower()
        if needle:
            records = [r for r in records if _job_matches(r, needle)]
        offset = errors.int_arg("offset", 0, lo=0, hi=10**9, clamp=True)
        limit = errors.int_arg("limit", _JOBS_DEFAULT_LIMIT, lo=1, hi=_JOBS_MAX_LIMIT, clamp=True)
        page = [_compact_job(r) for r in records[offset:offset + limit]]
        return jsonify({"ok": True, "campaign": campaign, "total": len(records),
                        "offset": offset, "limit": limit, "jobs": page})

    @app.route("/api/tracking/campaign/<name>")
    def api_tracking_campaign(name: str):
        """An archived (or the active, ``current``) campaign: metadata, its
        model/FITS/image backups, the notebook and the job count."""
        store = tracking_default_store()
        try:
            directory = store.campaign_dir(name)
            meta = store.campaign_meta(name) or {}
            backups = store.backups_in(name)
            log_md = store.read_log(name)
            jobs_count = store.count_fasrc_jobs(name)
        except TrackingError as e:
            return jsonify({"ok": False, "error": str(e)}), 404
        return jsonify({"ok": True, "dir": os.path.basename(directory),
                        "active": name == "current", "metadata": meta,
                        "backups": backups, "log_md": log_md,
                        "jobs_count": jobs_count})

    @app.route("/api/tracking/new", methods=["POST"])
    def api_tracking_new():
        title = request.form.get("title", "").strip()
        desc = request.form.get("description", "").strip()
        if not title:
            return jsonify({"ok": False, "error": "title is required"}), 400
        try:
            meta = tracking_default_store().create_campaign(title, desc)
        except TrackingError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        return jsonify({"ok": True, "metadata": meta})

    @app.route("/api/tracking/save", methods=["POST"])
    def api_tracking_save():
        try:
            res = tracking_default_store().save_campaign()
        except TrackingError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        sync = _tracking_try_sync(tracking_default_store())
        return jsonify({"ok": True, "archive_path": res["archive_path"],
                        "metadata": res["metadata"], "sync": sync,
                        "warning": dirty_warning(
                            res["metadata"].get("saved_commit"))})

    @app.route("/api/tracking/log", methods=["POST"])
    def api_tracking_log():
        text = request.form.get("text", "")
        mode = request.form.get("mode", "append")
        store = tracking_default_store()
        try:
            if mode == "replace":
                store.write_log(text)
            elif text.strip():
                store.append_log(text)
            else:
                return jsonify({"ok": False, "error": "empty note"}), 400
        except TrackingError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        return jsonify({"ok": True, "log_md": store.read_log()})

    @app.route("/api/tracking/backup", methods=["POST"])
    def api_tracking_backup():
        kind = request.form.get("kind", "").strip()
        comment = request.form.get("comment", "").strip()
        name = (request.form.get("name") or "").strip() or None
        store = tracking_default_store()
        try:
            if kind == "model":
                raw = (request.form.get("ckpt_dir") or "").strip() \
                    or Config.DEFAULT_CHECKPOINT_DIR
                rec = store.backup_model(
                    _resolve_trackable_ckpt(raw), comment, name,
                )
                # Bundle a rendered training-log plot into the backup so the
                # visualization travels with the checkpoint ("saved on the
                # checkpoint"). Best-effort — never fail the backup over it.
                try:
                    bdir = store.model_backup_dir("current", rec["name"])
                    csv = os.path.join(bdir, TrainingLog.FILENAME)
                    if os.path.isfile(csv):
                        plot_training_log(
                            csv, os.path.join(bdir, "training_log.png"))
                except Exception:
                    pass
            elif kind == "fits":
                rec = store.backup_fits(
                    _resolve_trackable_file(request.form.get("path", "")),
                    comment, name,
                )
            elif kind == "image":
                rec = store.backup_image(
                    _resolve_trackable_file(request.form.get("path", "")),
                    comment, name,
                )
            else:
                return jsonify({"ok": False,
                                "error": f"unknown kind {kind!r}"}), 400
        except TrackingError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        return jsonify({"ok": True, "record": rec,
                        "warning": dirty_warning(rec.get("commit")),
                        "sync": _tracking_try_sync(store)})

    @app.route("/api/tracking/sync", methods=["POST"])
    @requires_fasrc
    def api_tracking_sync():
        res = _tracking_try_sync(tracking_default_store())
        return jsonify(res), (200 if res.get("ok") else 400)

    # ---- time-travel: re-run a backup's exact code in a sandbox ----------

    def _sandbox_fasrc_cfg(short: str) -> dict[str, Any]:
        """Live FASRC config with all paths redirected to the remote sandbox,
        so a job submitted from the sandbox server can't touch live work."""
        cfg = fasrc_config.load()
        base = tracking_timetravel.remote_sandbox_base(cfg.data_dir, short)
        d = cfg.to_dict()
        d.update({
            "repo_path":           tracking_timetravel.remote_worktree_path(
                                       cfg.repo_path, short),
            "data_dir":            f"{base}/data",
            "ckpt_dir":            f"{base}/ckpt/wdsr",
            "tracking_remote_dir": f"{base}/tracking",
            "control_socket":      f"/tmp/euclid-polish-tt-{short}.sock",
        })
        return d

    @app.route("/api/tracking/timetravel/restore", methods=["POST"])
    def api_timetravel_restore():
        store = tracking_default_store()
        campaign = (request.form.get("campaign") or "current").strip()
        model = (request.form.get("model") or "").strip()
        backup = (request.form.get("backup") or "").strip()
        kind = (request.form.get("kind") or "").strip()
        want_remote = request.form.get("remote") in ("1", "true", "yes", "on")
        source: dict[str, Any] = {"campaign": campaign, "model": model or None}
        # Resolve the commit (+ checkpoint to seed) from a model backup, a
        # FITS / image backup (the code only: nothing to seed), else from
        # the campaign's saved/created commit.
        try:
            if backup:
                if kind not in _FILE_BACKUP_KINDS:
                    kinds = sorted(_FILE_BACKUP_KINDS)
                    return jsonify({"ok": False, "error": f"kind must be one of {kinds}"}), 400
                records = store.backups_in(campaign).get(_FILE_BACKUP_KINDS[kind], [])
                rec = next((r for r in records if r.get("name") == backup), None)
                if rec is None:
                    error = f"no {kind} backup {backup!r} in {campaign!r}"
                    return jsonify({"ok": False, "error": error}), 400
                commit_info = rec.get("commit")
                seed_dir = None
                source.update(backup=backup, kind=kind)
            elif model:
                mm = store.model_backup_meta(campaign, model) or {}
                commit_info = mm.get("commit")
                # A retired-model zip restores the code at its commit but is
                # no live checkpoint to seed the sandbox with.
                seed_dir = (None if model.endswith(".zip")
                            else store.model_backup_dir(campaign, model))
            else:
                cm = store.campaign_meta(campaign) or {}
                commit_info = cm.get("saved_commit") or cm.get("created_commit")
                seed_dir = None
        except TrackingError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        if not commit_info or not commit_info.get("hash"):
            return jsonify({"ok": False, "error": "no git commit recorded for "
                "this backup — it was made outside a git repo, so the exact "
                "code can't be restored."}), 400
        commit = commit_info["hash"]

        try:
            sb = tracking_timetravel.prepare_local_sandbox(
                commit, live_data_dir=Config.DATA_DIR, seed_ckpt_dir=seed_dir,
                source=source)
        except tracking_timetravel.TimeTravelError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        short = sb["short"]

        # Always isolate the sandbox's FASRC config (remote sandbox paths) so
        # even an accidental submit lands in the sandbox, not live work.
        tracking_timetravel.write_home_fasrc_config(short, _sandbox_fasrc_cfg(short))

        remote_res = None
        if want_remote:
            cfg = fasrc_config.load()
            push_cmd = ["git", "-C", _PROJECT_ROOT, "push",
                        "origin", f"{commit}:refs/heads/timetravel/{short}"]
            remote_res = tracking_timetravel.prepare_remote_sandbox(
                STATE.ssh, repo_path=cfg.repo_path, data_dir=cfg.data_dir,
                commit=commit, short=short, push_origin_cmd=push_cmd)
            tracking_timetravel.set_sandbox_remote(short, remote_res)

        spawn = tracking_timetravel.spawn_server(short)
        return jsonify({
            "ok":      bool(spawn.get("ok")),
            "short":   short,
            "url":     spawn.get("url"),
            "spawn":   spawn,
            "remote":  remote_res,
            "commit":  commit_info,
            "warning": dirty_warning(commit_info),
        })

    @app.route("/api/tracking/timetravel/open", methods=["POST"])
    def api_timetravel_open():
        short = _sandbox_short()
        try:
            return jsonify(tracking_timetravel.spawn_server(short))
        except tracking_timetravel.TimeTravelError as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/tracking/timetravel/stop", methods=["POST"])
    def api_timetravel_stop():
        short = _sandbox_short()
        try:
            return jsonify(tracking_timetravel.stop_server(short))
        except tracking_timetravel.TimeTravelError as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/tracking/timetravel/remove", methods=["POST"])
    def api_timetravel_remove():
        short = _sandbox_short()
        try:
            return jsonify(tracking_timetravel.remove_sandbox(short))
        except tracking_timetravel.TimeTravelError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
