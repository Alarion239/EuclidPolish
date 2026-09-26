"""fasrc routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

import contextlib
import json
import os
import shlex
import subprocess
import time
import traceback
from typing import Any, cast

from flask import jsonify, request

from euclid_polish.observability.training_log import TrainingLog
from euclid_polish.web import (
    fasrc_config,
    fasrc_jobs,
    fasrc_log_parser,
    fasrc_queue,
    job_config,
)
from euclid_polish.web.fasrc_gate import requires_fasrc
from euclid_polish.web.fasrc_mirror import MIRROR
from euclid_polish.web.fasrc_pipeline import REGISTRY as STEP_REGISTRY
from euclid_polish.web.fasrc_pipeline import StepResources, TaskParamError
from euclid_polish.web.job_status import JobStatusFetcher
from euclid_polish.web.jobs import REGISTRY as JOB_REGISTRY
from euclid_polish.web.remote import STATE, SSHError, SSHSession, connect_from_config

# Sentinel line appended to the remote env-update command so the local job
# learns the pipeline's exit status from a plain output stream.
_ENV_UPDATE_EXIT_MARKER = "__EP_EXIT__"
# Keep-alive line the remote env-update watchdog prints every
# ``_ENV_UPDATE_HEARTBEAT_S`` seconds (filtered from the job log). It lets a
# cancel land while mamba solves silently, and its failing write is how the
# remote side learns the channel is gone (see ``_env_update_watchdog``).
_ENV_UPDATE_ALIVE_MARKER = "__EP_ALIVE__"
_ENV_UPDATE_HEARTBEAT_S: float = 2.0


def _env_update_watchdog(period_s: float) -> str:
    """Bash prefix: a heartbeat that kills the remote job once nobody listens.

    ``SSHSession.stream`` runs without a pty, so closing it (a cancel) only
    ends the local ``ssh`` client; the remote processes get no SIGHUP and a
    silent ``mamba`` solve would run on until its next write. The background
    loop prints ``_ENV_UPDATE_ALIVE_MARKER`` every ``period_s`` with SIGPIPE
    ignored, so a write into the closed channel fails and it runs
    ``kill -TERM 0``: sshd starts each session command with ``setsid``, so
    process group 0 is exactly this command (``yes``, ``mamba``, the shell).
    The ``EXIT`` trap stops the loop when the update ends normally; its
    ``sleep`` writes to /dev/null so it never holds the channel open.
    A local stand-in for ``SSHSession`` must likewise start the command in
    its own session (``start_new_session=True``, as ``tests/_local_ssh.py``
    does), or ``kill -TERM 0`` would reach the caller's process group.
    """
    return (
        "( trap '' PIPE; "
        f"while sleep {period_s:g} >/dev/null 2>&1; do "
        f"echo {_ENV_UPDATE_ALIVE_MARKER} 2>/dev/null || kill -TERM 0; "
        "done ) & __ep_hb=$!; "
        "trap 'kill $__ep_hb 2>/dev/null' EXIT; "
    )


class _JobStatusSSHAdapter:
    """Expose the text-only subset of ``SSHSession`` used for job events."""

    def __init__(self, session: SSHSession) -> None:
        self._session = session

    def run(
        self, cmd: str, *, timeout: float = 60.0,
    ) -> tuple[int, str, str]:
        return_code, output, error = self._session.run(
            cmd, timeout=int(timeout))
        return return_code, cast(str, output), error

    def is_connected(self) -> bool:
        return self._session.is_connected()


def _job_status_ssh(
    session: SSHSession | None,
) -> _JobStatusSSHAdapter | None:
    return _JobStatusSSHAdapter(session) if session is not None else None


def _merge_squeue_fields(
    row: dict[str, Any], squeue_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Overlay the live squeue columns of ``row``'s job (or its RUNNING array
    task, else its first task) onto a JobDB row."""
    jid = str(row.get("jobid", "")).strip()
    live_rows = fasrc_jobs.array_squeue_rows(jid, squeue_rows)
    live = next((r for r in live_rows if r.get("state") == "RUNNING"), None)
    live = live or next(iter(live_rows), None)
    if live is not None:
        for key in ("start_time", "reason", "nodes", "time", "time_limit"):
            value = live.get(key)
            if value is not None and value != "":
                row[key] = value
    return row


def register(app):
    # =========================================================================
    # FASRC tab — Bitwarden-driven SSH ControlMaster, SLURM submission,
    # live log streaming, checkpoint auto-mirror.
    # =========================================================================

    # ---- config -----------------------------------------------------------

    @app.route("/api/fasrc/config", methods=["GET", "POST"])
    def api_fasrc_config():
        if request.method == "POST":
            patch = dict(request.form.items())
            cfg = fasrc_config.update(patch)
        else:
            cfg = fasrc_config.load()
        return jsonify(cfg.to_dict())

    # ---- auth -------------------------------------------------------------

    @app.route("/api/fasrc/status")
    def api_fasrc_status():
        """Connection state; ``last_error`` explains a disconnected state
        (startup auto-connect error or the last failed connect)."""
        return jsonify(STATE.public_status())

    @app.route("/api/fasrc/connect", methods=["POST"])
    def api_fasrc_connect():
        try:
            session = connect_from_config()
        except SSHError as e:
            return jsonify({"ok": False, "error": str(e),
                            "status": STATE.public_status()}), 400
        # Catch up on any jobs that finished while the server was offline:
        # squeue no longer lists them, so reconcile marks them DONE and
        # the ssh-passing path fetches their sacct accounting into the
        # CSV log. Best-effort — failures here must not block connect.
        with contextlib.suppress(Exception):
            fasrc_jobs.sync_pending_on_connect(session)
        return jsonify({"ok": True, "status": STATE.public_status()})

    @app.route("/api/fasrc/disconnect", methods=["POST"])
    def api_fasrc_disconnect():
        if STATE.ssh:
            STATE.ssh.disconnect()
        STATE.ssh = None
        STATE.connected_at = None
        STATE.last_error = None
        MIRROR.stop()
        return jsonify({"ok": True, "status": STATE.public_status()})

    # ---- remote info ------------------------------------------------------

    @app.route("/api/fasrc/git-status")
    @requires_fasrc
    def api_fasrc_git_status():
        ssh = STATE.ssh
        if ssh is None or not ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        cfg = fasrc_config.load()
        repo = cfg.repo_path
        cmds = (
            f"cd {repo} && "
            f"git rev-parse --abbrev-ref HEAD && "
            f"git fetch --quiet && "
            f"git rev-list --left-right --count HEAD...@{{u}} 2>/dev/null && "
            f"git log -1 --pretty=format:'%h%x09%s%x09%cr'"
        )
        rc, out, err = ssh.run(cmds, timeout=30)
        if rc != 0:
            return jsonify({"ok": False, "error": err.strip() or out.strip()}), 500
        lines = out.strip().splitlines()
        branch  = lines[0] if len(lines) > 0 else ""
        counts  = (lines[1].split() if len(lines) > 1 else ["0", "0"])
        ahead   = int(counts[0]) if counts and counts[0].isdigit() else 0
        behind  = int(counts[1]) if len(counts) > 1 and counts[1].isdigit() else 0
        last    = lines[2].split("\t", 2) if len(lines) > 2 else []
        last_commit = ({"hash": last[0], "subject": last[1], "relative": last[2]}
                       if len(last) == 3 else {})
        return jsonify({"ok": True, "repo": repo, "branch": branch,
                        "ahead": ahead, "behind": behind,
                        "last": last_commit})

    @app.route("/api/fasrc/git-pull", methods=["POST"])
    @requires_fasrc
    def api_fasrc_git_pull():
        """``git pull`` + auto-update conda env when ``environment.yml`` moved.

        Returns ``env_update_needed: True`` whenever the pull's diff
        touches ``environment.yml``; the UI then starts the
        ``POST /api/fasrc/env-update`` job automatically so the user
        doesn't have to remember.
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        cfg = fasrc_config.load()
        # Run ``git pull`` and ask in the same shell what just moved.
        # ``ORIG_HEAD..HEAD`` is everything fetched; if the pull was a
        # no-op the second command emits an empty list.
        rc, out, err = STATE.ssh.run(
            f"cd {shlex.quote(cfg.repo_path)} && "
            f"git pull --ff-only && "
            f"echo '__CHANGED__' && "
            f"git diff --name-only ORIG_HEAD..HEAD 2>/dev/null || true",
            timeout=60,
        )
        out_text = (out + err).strip()
        changed_files: list[str] = []
        if "__CHANGED__" in out:
            head, _, tail = out.partition("__CHANGED__")
            out_text = head.strip()
            changed_files = [line for line in tail.splitlines() if line.strip()]
        env_update_needed = any(
            f.endswith("environment.yml") for f in changed_files
        )
        return jsonify({
            "ok":                rc == 0,
            "stdout":            out_text,
            "changed_files":     changed_files,
            "env_update_needed": env_update_needed,
            "error":             "" if rc == 0 else
                                  (err.strip() or out.strip()),
        })

    @app.route("/api/fasrc/data-listing")
    @requires_fasrc
    def api_fasrc_data_listing():
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        cfg = fasrc_config.load()
        # Each section is guarded with `[ -d path ] &&` so a missing
        # directory (common on a fresh netscratch dir) doesn't sink the
        # whole listing. The trailing ``exit 0`` keeps the SSH call
        # green even if every section is empty.
        # ``du -shL`` dereferences symlinks: COSMOS2025 / euclid_psf are
        # typically symlinks into <repo>/data on holylabs, and the naked
        # ``du`` reports the link itself (~60 B, rounds to 0). ``-L``
        # follows the link and reports the contents. ``find -L`` mirrors
        # that semantic for the tfrecord / checkpoint sweeps below.
        # Each section is capped with a remote ``timeout`` so a slow ``du -L``
        # (it dereferences symlinks into big holylabs trees) can't hang the whole
        # call — it returns whatever completed and the sweeps still run. Without
        # this the request sat for 30 s then 500'd, so the Storage tab "never
        # loaded".
        cmd = (
            f"{{ "
            f"  [ -d {shlex.quote(cfg.data_dir)} ] && "
            f"    timeout 12 du -shL {shlex.quote(cfg.data_dir)}/* 2>/dev/null | sort -k2 ; "
            f"  echo '---' ; "
            f"  [ -d {shlex.quote(cfg.data_dir)} ] && "
            f"    timeout 8 find -L {shlex.quote(cfg.data_dir)} -maxdepth 3 -type f "
            f"      -name '*.tfrecord' -printf '%p\\t%s\\n' 2>/dev/null ; "
            f"  echo '---' ; "
            f"  [ -d {shlex.quote(cfg.ckpt_dir)} ] && "
            f"    timeout 8 find -L {shlex.quote(cfg.ckpt_dir)} -maxdepth 2 -type f "
            f"      -printf '%p\\t%s\\t%TY-%Tm-%Td %TH:%TM\\n' 2>/dev/null ; "
            f"}}; exit 0"
        )
        try:
            rc, out, err = STATE.ssh.run(cmd, timeout=32)
        except (subprocess.TimeoutExpired, SSHError):
            return jsonify({"ok": False, "error":
                            "remote listing timed out — netscratch is slow right "
                            "now; hit ↻ to retry."}), 200
        if rc != 0:
            return jsonify({"ok": False,
                            "error": f"remote du/find failed: {err.strip()}"}), 200
        sections = out.split("---")
        du_lines     = (sections[0].splitlines() if len(sections) > 0 else [])
        tfr_lines    = (sections[1].splitlines() if len(sections) > 1 else [])
        ckpt_lines   = (sections[2].splitlines() if len(sections) > 2 else [])

        def _split(line: str, n: int) -> list[str]:
            parts = line.split("\t" if "\t" in line else None, n - 1)
            return parts + [""] * (n - len(parts))

        return jsonify({
            "ok": True,
            "data_dir": cfg.data_dir,
            "ckpt_dir": cfg.ckpt_dir,
            "du": [line.split(None, 1) for line in du_lines if line.strip()],
            "tfrecords": [
                {"path": p, "size": int(s) if s.isdigit() else 0}
                for line in tfr_lines if line.strip()
                for p, s in [_split(line, 2)[:2]]
            ],
            "checkpoints": [
                {"path": p, "size": int(s) if s.isdigit() else 0, "mtime": m}
                for line in ckpt_lines if line.strip()
                for p, s, m in [_split(line, 3)[:3]]
            ],
        })

    @app.route("/api/fasrc/bootstrap-data", methods=["POST"])
    @requires_fasrc
    def api_fasrc_bootstrap_data():
        """Re-create the symlinks that point ``data_dir`` at the durable
        copy of the same data under ``{repo_path}/data/`` on holylabs.
        Idempotent: re-runnable after a netscratch purge, after committing
        new PSFs, or after Globus uploads a fresh COSMOS catalog —
        without manual cleanup.

        Targets (source → link name under ``data_dir``):
          - ``{repo_path}/data/euclid_psf``  → ``euclid_psf``   (ships via git)
          - ``{repo_path}/data/COSMOS2025``  → ``COSMOS2025``   (Globus-uploaded)
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        cfg = fasrc_config.load()
        repo_data = f"{cfg.repo_path}/data"
        targets = [
            ("euclid_psf", f"{repo_data}/euclid_psf"),
            ("COSMOS2025", f"{repo_data}/COSMOS2025"),
        ]
        link_cmds = []
        for name, src in targets:
            link_cmds.append(
                f"if [ -e {shlex.quote(src)} ]; then "
                f"  ln -sfn {shlex.quote(src)} {shlex.quote(name)} "
                f"    && echo 'linked: {name} -> {src}' "
                f"    || echo 'FAILED: ln -sfn {src} {name}'; "
                f"else "
                f"  echo 'MISSING source: {src} — upload via Globus first'; "
                f"fi"
            )
        cmd = (
            f"mkdir -p {shlex.quote(cfg.data_dir)} && "
            f"cd {shlex.quote(cfg.data_dir)} && {{ "
            + " ; ".join(link_cmds)
            + "; echo '---'; ls -l . | head -40; "
            + "}"
        )
        rc, out, err = STATE.ssh.run(cmd, timeout=20)
        return jsonify({
            "ok":     rc == 0,
            "output": out.strip(),
            "error":  err.strip() if rc != 0 else "",
        })

    @app.route("/api/fasrc/queue")
    @requires_fasrc
    def api_fasrc_queue():
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        rc, out, err = STATE.ssh.run(
            f"squeue -r -h -u $USER --format='{fasrc_jobs.SQUEUE_FMT}'",
            timeout=15,
        )
        if rc != 0:
            return jsonify({"ok": False, "error": err.strip()}), 500
        rows = fasrc_jobs.parse_squeue(out)
        # Single source of truth for "is this job still alive?": rows
        # not present in squeue get marked DONE (if we'd seen them run)
        # or UNKNOWN (if we never did — the user can then look at .err
        # to figure out what happened, instead of seeing RUNNING forever).
        fasrc_jobs.reconcile_with_squeue(rows, ssh=STATE.ssh)
        return jsonify({"ok": True, "rows": rows})

    # ---- submission -------------------------------------------------------

    def _require_confirm(form):
        """Shared confirm-token guard for the two FASRC submit endpoints.

        Returns ``None`` when the form carries the explicit-confirm
        token; otherwise returns a ``(flask response, 400)`` tuple the
        caller can return directly.
        """
        if str(form.get("confirm", "")).lower() in ("yes", "true", "1"):
            return None
        return jsonify({
            "ok": False,
            "error": (
                "missing explicit confirmation token. Refresh the page "
                "and click Submit again — the flow shows a dialog with "
                "the full payload before any FASRC submit."
            ),
        }), 400

    # ---- local submission queue (one cluster job at a time, fail-stops) ----
    #
    # When a job is submitted while another is still active it is queued
    # locally instead of sbatch'd. On the active job's SUCCESS the next is
    # submitted; on FAILURE (incl OOM) the queue halts. See fasrc_queue.

    def _spec_label(step_ref, form):
        explicit = (form.get("label") or "").strip()
        if explicit:
            return explicit
        try:
            return STEP_REGISTRY.get(step_ref).label
        except KeyError:
            return f"step {step_ref}"

    def _render_spec(kind, step_ref, form):
        """Render a stored spec's sbatch script without touching FASRC.

        → ``(cfg, step_ref, label, built, resources)``. Raises
        ``ValueError`` when the spec cannot be built (bad member list, an
        inactive population calibration, …). Every spec is a pipeline step
        (``kind="step"``). A spec still queued by the removed
        ``/api/fasrc/submit`` (``kind="synthetic"``, persisted in
        ``fasrc_queue.json``) is rendered as the ``synthetic_generate`` step
        with blank resources taken from the step defaults.
        """
        cfg = fasrc_config.load()
        legacy = kind == "synthetic"
        if legacy and step_ref not in STEP_REGISTRY.by_id:
            step_ref = "synthetic_generate"
        step = STEP_REGISTRY.get(step_ref)
        form2 = dict(form)
        # Partition is fixed per job type — force the step's value even on
        # queued specs built from an older form.
        form2["partition"] = step.defaults.partition
        if step.fixed_cpus is not None:
            form2["n_cpus"] = str(step.fixed_cpus)
        if step.fixed_gpus is not None:
            form2["n_gpus"] = str(step.fixed_gpus)
        resources = (StepResources.from_form(form2, step.defaults) if legacy
                     else StepResources.from_form_strict(form2))
        if step.fixed_cpus is not None:
            resources.n_cpus = int(step.fixed_cpus)
        if step.fixed_gpus is not None:
            resources.n_gpus = int(step.fixed_gpus)
        label = _spec_label(step_ref, form2)
        built = step.build_sbatch_body(
            params=form2, resources=resources, cfg=cfg, label=label)
        return cfg, step_ref, label, built, resources

    def _build_and_submit(kind, step_ref, form):
        """Render + sbatch a job from a stored spec → (slurm_id, payload)."""
        cfg, step_ref, label, built, resources = _render_spec(
            kind, step_ref, form)
        # Array steps resolve member names/base seeds while rendering, and
        # ``prepare_payload_files`` swaps embedded calibration JSON for
        # immutable sidecar paths + digests. Persist those prepared values so
        # monitoring can map task indices to members and the job DB records
        # exactly which calibrations the remote command consumed.
        params_for_db = dict(built.get("params", form))
        params_for_db.update(resources.to_dict())
        params_for_db["step_id"] = step_ref
        return fasrc_jobs.submit_sbatch_script(
            STATE.ssh, cfg=cfg, built=built, label=label,
            params=params_for_db, step_id=step_ref)

    def _submit_spec_now(spec):
        return _build_and_submit(spec["kind"], spec["step"], spec["form"])

    def _queue_tick():
        """Promote/halt the local queue — call after every squeue reconcile."""
        try:
            fasrc_queue.QUEUE.tick(
                fasrc_jobs.DB, fasrc_jobs.JOBLOG, STATE.ssh, _submit_spec_now)
        except Exception:
            traceback.print_exc()

    def _submit_or_queue(step_ref, form):
        """Submit immediately if the single lane is free, else enqueue."""
        label = _spec_label(step_ref, form)
        spec = {"kind": "step", "step": step_ref, "form": form}
        # Dry-run the exact render promotion will do (prepare_params, payload
        # staging, build_command — no SSH) before queueing: a queued spec is
        # only built when the lane frees, and a build failure there halts the
        # whole queue. This also checks the fitted population artifacts are
        # active; they are resolved and embedded again at promotion, so no
        # Config/legacy population fallback can enter.
        try:
            _render_spec(spec["kind"], step_ref, form)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if fasrc_queue.QUEUE.active_is_running(fasrc_jobs.DB):
            fasrc_queue.QUEUE.enqueue(spec, label)
            return jsonify({"ok": True, "queued": True, "label": label,
                            "queue": fasrc_queue.QUEUE.public()})
        # Lane free → a fresh submit also clears any prior halt (resume).
        if fasrc_queue.QUEUE.halted:
            fasrc_queue.QUEUE.resume()
        try:
            slurm_id, payload = _submit_spec_now(spec)
        except subprocess.TimeoutExpired:
            # The sbatch/scp over SSH timed out — usually the login node being
            # briefly slow, not a bad submit. Give a clear, retryable message
            # instead of a bare 500.
            return jsonify({"ok": False, "error":
                "FASRC connection timed out while submitting — the login node "
                "may be briefly slow. Try again in a moment (reconnect if it "
                "persists)."}), 503
        except SSHError as e:
            return jsonify({"ok": False, "error": f"FASRC SSH error: {e}"}), 503
        except ValueError as e:
            # e.g. a malformed per-member spec — surface the reason, not a 500.
            return jsonify({"ok": False, "error": str(e)}), 400
        if slurm_id is None:
            return jsonify(payload), 500
        fasrc_queue.QUEUE.on_direct_submit(slurm_id)
        payload["queue"] = fasrc_queue.QUEUE.public()
        return jsonify(payload)

    @app.route("/api/fasrc/queue/clear", methods=["POST"])
    def api_fasrc_queue_clear():
        return jsonify({"ok": True, "queue": fasrc_queue.QUEUE.clear()})

    @app.route("/api/fasrc/queue/remove", methods=["POST"])
    def api_fasrc_queue_remove():
        item_id = (request.form.get("id") or "").strip()
        return jsonify({"ok": True,
                        "queue": fasrc_queue.QUEUE.remove(item_id)})

    # =========================================================================
    # Pipeline steps (generic FASRC submissions)
    # =========================================================================

    @app.route("/api/fasrc/steps/status")
    def api_fasrc_steps_status():
        """Per-step: name, defaults, last-runtime median, on-disk status.

        Uses ``STATE.ssh`` to ``test`` for each artifact's existence; if
        SSH isn't connected we return only the static defaults so the UI
        can still render its forms.
        """
        cfg_loaded = fasrc_config.load()
        ssh = STATE.ssh
        ssh_ok = ssh is not None and ssh.is_connected()

        steps_payload = []
        for step in STEP_REGISTRY.all():
            steps_payload.append({
                "step_id":     step.step_id,
                "label":       step.label,
                "needs_gpu":   step.needs_gpu,
                "fixed_cpus":  step.fixed_cpus,
                "fixed_gpus":  step.fixed_gpus,
                "defaults":    step.defaults.to_dict(),
                # Contract C5: the schema the SPA renders generically, and
                # the task params of the newest successful run (prefill).
                "task_params": step.task_param_schema(),
                "last_params": step.last_task_params(
                    fasrc_jobs.JOBLOG.history_for_step(step.step_id)),
            })

        # Cheap probes for "does this artifact exist on FASRC?" — single
        # ``test -e`` per check, batched in one SSH round-trip. Keep
        # this list in sync with the ``produces`` map in fasrc.html
        # (the JS side maps each step_id to one of these keys).
        artifacts = {
            "ckpt": None,
            # Per-page Euclid star-cutout pipeline:
            #   euclid_cutouts — VIS cutout subdir, written by the
            #                    download_euclid_cutouts step.
            #   euclid_psf     — VIS empirical ePSF, written by the
            #                    extract_euclid_psf (all-band) step.
            "euclid_cutouts": None, "euclid_psf": None,
            # Synthetic generation (/sky page).
            "synthetic_records": None,
        }
        if ssh_ok and ssh is not None:
            paths = {
                "ckpt":    f"{cfg_loaded.ckpt_dir}/checkpoint",
                "euclid_cutouts":
                    f"{cfg_loaded.data_dir}/euclid_stars/cutouts/VIS",
                "euclid_psf":
                    f"{cfg_loaded.data_dir}/euclid_psf/euclid_psf_VIS.fits",
                "synthetic_records":
                    f"{cfg_loaded.data_dir}/images/records_v2/clean_train.tfrecord",
            }
            probe = " && ".join(
                f"(test -e {shlex.quote(p)} && echo {k}=1 || echo {k}=0)"
                for k, p in paths.items()
            )
            try:
                rc, out, _err = ssh.run(probe, timeout=10)
                if rc == 0:
                    for line in out.splitlines():
                        line = line.strip()
                        if "=" in line:
                            k, v = line.split("=", 1)
                            if k in artifacts:
                                artifacts[k] = (v == "1")
            except Exception:
                pass

        return jsonify({
            "ssh_connected": ssh_ok,
            "steps":         steps_payload,
            "artifacts":     artifacts,
            "remote_paths": {
                "data_dir":    cfg_loaded.data_dir,
                "ckpt_dir":    cfg_loaded.ckpt_dir,
                "logs_dir":    "logs/pipeline",
            },
        })

    @app.route("/api/fasrc/steps/<step_id>/submit", methods=["POST"])
    @requires_fasrc
    def api_fasrc_step_submit(step_id: str):
        """Generic submission for any pipeline step.

        Two defences, in order of evaluation:

        1. ``confirm=yes`` token — proves the frontend dialog was
           shown (catches stale-JS tabs and most accidental submits).
        2. SSH connected (sanity check before the work starts).

        Any failure returns 400 and DOES NOT touch the SSH session,
        so no sbatch call, no script write, nothing reaches FASRC."""

        ssh = STATE.ssh
        if ssh is None or not ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        try:
            step = STEP_REGISTRY.get(step_id)
        except KeyError:
            return jsonify({"ok": False, "error": f"unknown step: {step_id}"}), 404
        form = request.form.to_dict()
        # Multi-valued fields (the ensemble continue-mode member checkboxes)
        # flatten to a comma-joined string — to_dict() alone keeps only the
        # first value.
        for key in ("members",):
            vals = request.form.getlist(key)
            if len(vals) > 1:
                form[key] = ",".join(vals)

        confirm_err = _require_confirm(form)
        if confirm_err is not None:
            return confirm_err

        force_redownload = str(form.get("force_redownload", "")).strip().lower() in (
            "1", "true", "yes", "on",
        )
        if (
            step_id == "archive_field_sample"
            and force_redownload
            and str(form.get("confirm_force_redownload", "")).strip().lower()
            not in ("1", "true", "yes")
        ):
            return jsonify({
                "ok": False,
                "error": (
                    "forcing a multipoint archive re-download requires its "
                    "separate confirmation dialog"
                ),
            }), 400
        # This token proves the stronger prompt was shown; it is not a science
        # parameter and should not become part of job identity/history.
        form.pop("confirm_force_redownload", None)

        # The partition is determined by the job type (gpu for training,
        # shared for everything else) — it is not a form question. Force
        # the step's partition regardless of what the form sent, BEFORE
        # the strict parse (which requires the field).
        form["partition"] = step.defaults.partition

        # A step with a locked CPU count renders the field as read-only
        # text, so the form may not carry ``n_cpus`` at all. Inject the
        # locked value before the strict parse (which requires it) — the
        # value is forced again below regardless of what the form sent.
        if step.fixed_cpus is not None:
            form["n_cpus"] = str(step.fixed_cpus)
        if step.fixed_gpus is not None:
            form["n_gpus"] = str(step.fixed_gpus)

        try:
            resources = StepResources.from_form_strict(form)
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400

        # If the step locks the CPU count, force it regardless of what
        # the form sent. Prevents the user from over-allocating cores
        # for a single-threaded job.
        if step.fixed_cpus is not None:
            resources.n_cpus = int(step.fixed_cpus)
        if step.fixed_gpus is not None:
            resources.n_gpus = int(step.fixed_gpus)

        # Task params (C5): absent ones take the step's schema defaults (so an
        # euclid_query submit with no knobs asks for 10,000 stars, never the
        # old 200); an invalid one is refused before anything reaches FASRC.
        try:
            form = step.fill_task_params(form)
        except TaskParamError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        # Fill universal job-config values. Most steps always inherit /config
        # (including computed/locked values); React Train members deliberately
        # exposes experiment-local ensemble controls, so preserve values that
        # page explicitly submitted instead of silently replacing them.
        for param_name, value in job_config.fasrc_params_for(step_id).items():
            if step_id == "ensemble_train":
                form.setdefault(param_name, value)
            else:
                form[param_name] = value

        # All form values are passed as ``params``; the step picks out
        # what it needs.
        # Validation above (step, confirm, resources) has passed; hand off
        # to the local queue: submit now if the lane is free, else enqueue.
        return _submit_or_queue(step_id, form)

    @app.route("/api/fasrc/refresh-accounting", methods=["POST"])
    @requires_fasrc
    def api_fasrc_refresh_accounting():
        """One-shot: re-pull sacct for every finalised job and re-record.

        Use after a change to how a post-mortem stat is computed (e.g. the
        CPU-utilisation fix) to backfill existing history rows."""
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        return jsonify(fasrc_jobs.refresh_all_post_mortems(STATE.ssh))

    @app.route("/api/fasrc/steps/<step_id>/history", methods=["GET", "POST"])
    def api_fasrc_step_history(step_id: str):
        """Per-step run history + best-match prefill suggestion.

        Powers the "Previous runs" panel and the resources prefill under
        each step's form. The client posts the current task-params
        dict (everything in the form except resource fields and meta
        keys); we serialise it the same way the submitter does so the
        latest-match lookup uses string equality on ``params_json``.

        Response shape:

        ```json
        {
          "ok":       true,
          "step_id":  "extract_psf",
          "history":  [<row>, ...],          # newest first, all states
          "match":    <row> | null,          # latest exact-match row
          "task_params_json": "..."          # canonical serialisation
        }
        ```
        """
        try:
            STEP_REGISTRY.get(step_id)
        except KeyError:
            return jsonify({"ok": False, "error": f"unknown step: {step_id}"}), 404

        # Accept task params via form POST (UI flow) OR query string GET
        # (curl-friendly debugging). Strip the same meta keys the
        # submitter strips so the match lookup is symmetric.
        raw = request.form.to_dict() if request.method == "POST" else request.args.to_dict()
        task_params = {
            k: v for k, v in raw.items()
            if k not in ("partition", "n_cpus", "n_gpus", "memory",
                         "time_limit", "confirm", "label", "preset")
        }
        params_json = json.dumps(
            task_params, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True,
        ) if task_params else ""

        history = fasrc_jobs.JOBLOG.history_for_step(step_id)
        match   = fasrc_jobs.JOBLOG.latest_match(step_id, params_json)
        return jsonify({
            "ok":               True,
            "step_id":          step_id,
            "history":          history,
            "match":            match,
            "task_params_json": params_json,
        })

    @app.route("/api/fasrc/cancel", methods=["POST"])
    @requires_fasrc
    def api_fasrc_cancel():
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        jid = request.form.get("jobid", "").strip()
        if not jid.isdigit():
            return jsonify({"ok": False, "error": "bad job id"}), 400
        rc, _, err = STATE.ssh.run(f"scancel {jid}", timeout=10)
        if rc != 0:
            return jsonify({"ok": False, "error": err.strip()}), 500
        fasrc_jobs.DB.update_state(jid, state="CANCELLED",
                                   ended_at=time.time())
        return jsonify({"ok": True})

    @app.route("/api/fasrc/current-submission")
    @requires_fasrc
    def api_fasrc_current_submission():
        """Return the user's most-recent live submission + its event-stream status.

        Looks at the JobDB for the latest job in ``PENDING`` or
        ``RUNNING`` state, reconciles its row against ``squeue`` so the
        elapsed/limit/node columns are fresh, and folds its ``.events``
        JSONL into a :class:`JobStatus` (stage, full stage history,
        step progress, warnings, errors).

        Response::

            { "ok": true, "current": null }   # no active job
            { "ok": true,
              "current": { "job": { ... DB row + squeue overrides ... },
                           "status": { stage, stages, step, warnings,
                                       errors, has_events, ... } } }

        Every response also carries ``live``: all PENDING/RUNNING rows
        (newest first, same shape as ``current.job``) for the job tray.
        """
        ssh = STATE.ssh
        if ssh is None or not ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400

        # Reconcile DB rows against squeue first — without this, a job
        # that already finished still shows up as RUNNING in the DB and
        # the tab would lie. One squeue call per refresh; the same one
        # the Logs tab already makes. A slow login node can time this out —
        # skip the reconcile for this tick (flagging the data stale) rather
        # than 500 the poll; the next tick retries.
        stale = False
        try:
            rc_q, out_q, _err_q = ssh.run(
                f"squeue -r -h -u $USER --format='{fasrc_jobs.SQUEUE_FMT}'",
                timeout=15,
            )
        except (subprocess.TimeoutExpired, SSHError):
            rc_q, out_q, stale = 1, "", True
        squeue_rows: list[dict[str, Any]] = []
        if rc_q == 0:
            squeue_rows = fasrc_jobs.parse_squeue(out_q)
            fasrc_jobs.reconcile_with_squeue(squeue_rows, ssh=ssh)
            # Advance the local queue (promote on success / halt on failure)
            # off the same reconcile that just refreshed job states.
            _queue_tick()

        queue_public = fasrc_queue.QUEUE.public()

        # Every still-live row (``live``, contract C5), newest first —
        # ``list_live`` orders by submitted_at DESC with no row limit, so the
        # first one IS the current submission. Merge live squeue fields into each row so the UI
        # sees the current ``start_time`` (PENDING jobs only), ``reason`` (why
        # SLURM hasn't started it: Priority / Resources / …), updated elapsed
        # ``time`` and assigned ``nodes``. reconcile_with_squeue only persists
        # state + started_at, so these are merged in at the response layer.
        # A stale tick (slow login node) returns the last-known DB rows.
        live_jobs = [
            dict(row) if stale else _merge_squeue_fields(dict(row), squeue_rows)
            for row in fasrc_jobs.DB.list_live()
        ]
        current_row = live_jobs[0] if live_jobs else None
        if current_row is None:
            return jsonify({"ok": True, "current": None, "queue": queue_public,
                            "stale": stale, "live": []})
        if stale:
            # Login node slow this tick — return the last-known DB row without
            # the extra SSH calls (squeue merge + event fetch) that would also
            # hang and 500. The next poll fills the live fields back in.
            return jsonify({"ok": True, "stale": True, "queue": queue_public,
                            "current": {"job": current_row, "status": None},
                            "live": live_jobs})
        jid = str(current_row.get("jobid", "")).strip()
        live_rows = fasrc_jobs.array_squeue_rows(jid, squeue_rows)

        # Fold the live event stream into a JobStatus. Array submissions have
        # one Reporter stream per model; expose them separately rather than
        # inventing a misleading aggregate training curve.
        fetcher = JobStatusFetcher(ssh=_job_status_ssh(ssh))
        try:
            stored_params = json.loads(current_row.get("params_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            stored_params = {}
        array_count = int(stored_params.get("array_count", 1) or 1)
        array_tasks = None
        if array_count > 1:
            names_raw = (stored_params.get("members")
                         if stored_params.get("mode") == "continue"
                         else stored_params.get("member_names"))
            member_names = [n.strip() for n in str(names_raw or "").split(",")]
            event_paths = [fasrc_jobs.expand_array_path(
                current_row.get("events_path"), jid, i)
                for i in range(array_count)]
            task_statuses = fetcher.fetch_many(event_paths)
            child_by_index = {}
            for row in live_rows:
                child_id = str(row.get("jobid", ""))
                suffix = child_id.removeprefix(jid + "_")
                if suffix.isdigit():
                    child_by_index[int(suffix)] = row
            array_tasks = []
            for i, task_status in enumerate(task_statuses):
                child = child_by_index.get(i, {})
                completed = bool(
                    task_status.step and task_status.step.total > 0
                    and task_status.step.current >= task_status.step.total
                )
                array_tasks.append({
                    "index": i,
                    "member": member_names[i] if i < len(member_names) else f"task {i}",
                    "jobid": f"{jid}_{i}",
                    "state": child.get("state") or
                             ("COMPLETED" if completed else "NOT_IN_QUEUE"),
                    "reason": child.get("reason"),
                    "nodes": child.get("nodes"),
                    "time": child.get("time"),
                    "status": task_status.to_dict(),
                })
            status = None
        else:
            status = fetcher.fetch(
                events_path=current_row.get("events_path")).to_dict()
        # Jobstats is richer than the local event sampler, but the endpoint
        # polls frequently.  The helper applies a 30-second per-job TTL and
        # only gets called for a job that is actually running.
        live_accounting = (
            fasrc_jobs.fetch_live_jobstats(ssh, jid)
            if current_row.get("state") == "RUNNING" and array_count <= 1
            else None
        )
        return jsonify({
            "ok":      True,
            "stale":   False,
            "current": {
                "job":    current_row,
                "status": status,
                "array": ({"count": array_count,
                           "max_parallel": stored_params.get("array_max_parallel"),
                           "tasks": array_tasks}
                          if array_tasks is not None else None),
                "accounting": live_accounting,
            },
            "queue":   queue_public,
            "live":    live_jobs,
        })

    @app.route("/api/fasrc/jobs/<jobid>/status")
    def api_fasrc_job_status(jobid: str):
        """Return the structured status for one job.

        Reads the job's ``.events`` JSONL stream (written on FASRC by
        :class:`euclid_polish.observability.Reporter`) and folds it into
        the :class:`JobStatus` shape the ``JobStatusCard`` widget polls
        for. Returns an empty status (``has_events=False``) when the
        job is queued / pre-Reporter / disconnected — never 500s on a
        missing file."""
        row = fasrc_jobs.DB.get(jobid)
        if not row:
            return jsonify({"ok": False, "error": "unknown jobid"}), 404
        fetcher = JobStatusFetcher(ssh=_job_status_ssh(STATE.ssh))
        try:
            params = json.loads(row.get("params_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            params = {}
        array_count = int(params.get("array_count", 1) or 1)
        array_tasks = None
        if array_count > 1:
            names_raw = (params.get("members") if params.get("mode") == "continue"
                         else params.get("member_names"))
            names = [n.strip() for n in str(names_raw or "").split(",")]
            paths = [fasrc_jobs.expand_array_path(
                row.get("events_path"), jobid, i) for i in range(array_count)]
            statuses = fetcher.fetch_many(paths)
            array_tasks = [{
                "index": i,
                "member": names[i] if i < len(names) else f"task {i}",
                "jobid": f"{jobid}_{i}",
                "status": task_status.to_dict(),
            } for i, task_status in enumerate(statuses)]
            status = None
        else:
            status = fetcher.fetch(events_path=row.get("events_path")).to_dict()
        return jsonify({
            "ok":          True,
            "jobid":       jobid,
            "state":       row.get("state"),
            "events_path": row.get("events_path"),
            "status":      status,
            "array":       ({"count": array_count, "tasks": array_tasks}
                            if array_tasks is not None else None),
        })

    # ---- past-runs browser (Logs tab) ---------------------------------------
    #
    # Combines two sources so the user sees every run that left a log on
    # FASRC, regardless of how it was submitted:
    #   (1) ``JobDB`` — every job submitted from this UI, with its SLURM
    #       jobid, label, state, and timestamps.
    #   (2) Remote ``find <repo>/logs -name '*.out' -o -name '*.err'``
    #       — picks up jobs submitted directly via sbatch from the CLI,
    #       which the DB has no record of.
    # Rows are de-duplicated by base name (the ``euclid-YYYYMMDD-HHMMSS``
    # prefix that pairs an ``.out`` with its ``.err``); UI-submitted jobs
    # therefore get their full DB metadata, CLI-submitted jobs just get
    # the file timestamps + sizes.

    @app.route("/api/fasrc/runs")
    @requires_fasrc
    def api_fasrc_runs():
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        cfg = fasrc_config.load()
        log_dir = f"{cfg.repo_path}/{cfg.logs_subdir}"

        # 0. Reconcile DB state against the live queue *before* we read
        # rows out of sqlite. Without this, any job that finished
        # while the Logs tab wasn't open shows up as RUNNING forever,
        # and jobs that disappeared from squeue before ever starting
        # (sbatch rejected, queue purged, etc.) stay PENDING. One
        # extra cheap squeue call per Logs-tab load is well worth it.
        rc_q, out_q, _err_q = STATE.ssh.run(
            f"squeue -r -h -u $USER --format='{fasrc_jobs.SQUEUE_FMT}'",
            timeout=15,
        )
        squeue_rows: list[dict[str, Any]] = []
        if rc_q == 0:
            squeue_rows = fasrc_jobs.parse_squeue(out_q)
            fasrc_jobs.reconcile_with_squeue(
                squeue_rows, ssh=STATE.ssh,
            )

        # 1. Scan remote for every .out / .err — one cheap SSH call.
        # ``stat -c '%Y\t%s\t%n'`` works on GNU coreutils (FASRC); falls
        # through with empty output if the dir doesn't exist yet.
        cmd = (
            f"{{ [ -d {shlex.quote(log_dir)} ] && "
            # Modern pipeline logs live in ``logs/pipeline`` while older
            # generic submissions live in ``logs/jobs``. Search the configured
            # log tree, not just the legacy jobs directory.
            f"find {shlex.quote(log_dir)} -maxdepth 3 -type f "
            f"\\( -name '*.out' -o -name '*.err' \\) "
            f"-printf '%T@\\t%s\\t%p\\n' 2>/dev/null "
            f"| sort -rn -k1,1 | head -20000 ; }}; exit 0"
        )
        rc, out, _err = STATE.ssh.run(cmd, timeout=15)
        files: dict[str, dict[str, Any]] = {}     # keyed by base name
        if rc == 0:
            for line in out.splitlines():
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                try:
                    mtime = float(parts[0])
                    size  = int(parts[1])
                except ValueError:
                    continue
                full = parts[2]
                base = os.path.basename(full)
                if base.endswith(".out"):
                    stem, kind = base[:-4], "out"
                elif base.endswith(".err"):
                    stem, kind = base[:-4], "err"
                else:
                    continue
                rec = files.setdefault(stem, {"name": stem, "mtime": 0.0})
                rec[f"{kind}_path"] = full
                rec[f"{kind}_size"] = size
                rec["mtime"] = max(rec["mtime"], mtime)

        # 2. Build one run per DB submission.  Array submissions deliberately
        #    store SLURM's literal ``%A_%a`` template in the DB, while the
        #    files on disk contain concrete ``<parent>_<index>`` values.  The
        #    old name-based overlay could never join those records: it showed
        #    an empty template row and (when found) anonymous child-file rows.
        #    Keep the submission as one parent run and attach its concrete log
        #    targets as named tasks for the UI to select.
        runs: list[dict[str, Any]] = []
        consumed_stems: set[str] = set()
        live_by_jobid = {
            str(row.get("jobid")): row for row in squeue_rows
            if row.get("jobid")
        }
        for db_row in fasrc_jobs.DB.list_recent(5000):
            lp = str(db_row.get("log_path") or "")
            base = os.path.basename(lp)
            stem = base[:-4] if base.endswith(".out") else base
            if not stem:
                continue
            try:
                params = json.loads(db_row.get("params_json") or "{}")
            except (TypeError, ValueError):
                params = {}
            try:
                array_count = max(1, int(params.get("array_count", 1) or 1))
            except (TypeError, ValueError):
                array_count = 1

            common = {
                "name":         stem,
                "jobid":        db_row.get("jobid"),
                "label":        db_row.get("label"),
                "state":        db_row.get("state"),
                "submitted_at": db_row.get("submitted_at") or 0.0,
                "started_at":   db_row.get("started_at"),
                "ended_at":     db_row.get("ended_at"),
                "params":       params,
            }
            if array_count > 1:
                parent_jobid = str(db_row.get("jobid") or "")
                names_key = "members" if params.get("mode") == "continue" else "member_names"
                member_names = [
                    name.strip() for name in str(params.get(names_key) or "").split(",")
                    if name.strip()
                ]
                tasks: list[dict[str, Any]] = []
                latest_mtime = float(db_row.get("submitted_at") or 0.0)
                total_out_size = 0
                total_err_size = 0
                for index in range(array_count):
                    task_out = fasrc_jobs.expand_array_path(
                        db_row.get("log_path"), parent_jobid, index,
                    )
                    task_err = fasrc_jobs.expand_array_path(
                        db_row.get("err_path"), parent_jobid, index,
                    )
                    task_base = os.path.basename(task_out or "")
                    task_stem = (task_base[:-4]
                                 if task_base.endswith(".out") else task_base)
                    rec = files.get(task_stem, {})
                    if task_stem in files:
                        consumed_stems.add(task_stem)
                    task_jobid = f"{parent_jobid}_{index}"
                    live = live_by_jobid.get(task_jobid)
                    parent_state = str(db_row.get("state") or "").upper()
                    task_state = live.get("state") if live else (
                        parent_state if parent_state in fasrc_jobs.TERMINAL_STATES
                        else None
                    )
                    out_size = int(rec.get("out_size", 0) or 0)
                    err_size = int(rec.get("err_size", 0) or 0)
                    latest_mtime = max(latest_mtime, float(rec.get("mtime", 0.0) or 0.0))
                    total_out_size += out_size
                    total_err_size += err_size
                    tasks.append({
                        "index": index,
                        "member": (member_names[index]
                                   if index < len(member_names)
                                   else f"task {index}"),
                        "jobid": task_jobid,
                        "name": task_stem,
                        "state": task_state,
                        "out_path": rec.get("out_path") or task_out,
                        "err_path": rec.get("err_path") or task_err,
                        "out_size": out_size,
                        "err_size": err_size,
                        "mtime": rec.get("mtime", 0.0),
                        "missing": not bool(rec.get("out_path") or rec.get("err_path")),
                    })
                runs.append({
                    **common,
                    "array_count": array_count,
                    "tasks": tasks,
                    "out_path": None,
                    "err_path": None,
                    "out_size": total_out_size,
                    "err_size": total_err_size,
                    "mtime": latest_mtime,
                    "missing": all(task["missing"] for task in tasks),
                })
                continue

            rec = files.get(stem, {})
            if stem in files:
                consumed_stems.add(stem)
            runs.append({
                **common,
                "out_path":     rec.get("out_path") or db_row.get("log_path"),
                "err_path":     rec.get("err_path") or db_row.get("err_path"),
                "out_size":     rec.get("out_size", 0),
                "err_size":     rec.get("err_size", 0),
                "mtime":        rec.get("mtime") or db_row.get("submitted_at") or 0.0,
                "missing":      not bool(rec.get("out_path") or rec.get("err_path")),
            })

        # 3. Files with no DB submission are CLI/manual runs.  Array child
        #    stems consumed above must not leak out as eight anonymous runs.
        for stem, rec in files.items():
            if stem in consumed_stems:
                continue
            runs.append({
                "name": stem,
                "jobid": None,
                "label": None,
                "state": None,
                "submitted_at": rec["mtime"],
                "started_at": None,
                "ended_at": None,
                "out_path": rec.get("out_path"),
                "err_path": rec.get("err_path"),
                "out_size": rec.get("out_size", 0),
                "err_size": rec.get("err_size", 0),
                "mtime": rec["mtime"],
                "params": {},
            })
        runs.sort(key=lambda r: r["mtime"], reverse=True)

        # Paginate runs (newest first). page 0 = newest; older pages walk
        # back through the full history up to the very first run.
        try:
            page = max(0, int(request.args.get("page", 0)))
        except ValueError:
            page = 0
        try:
            page_size = int(request.args.get("page_size", 100))
        except ValueError:
            page_size = 100
        page_size = max(10, min(page_size, 500))
        total = len(runs)
        start = page * page_size
        page_runs = runs[start:start + page_size]
        return jsonify({
            "ok":          True,
            "log_dir":     log_dir,
            "runs":        page_runs,
            "total_runs":  total,
            "page":        page,
            "page_size":   page_size,
            "start_index": (start + 1) if page_runs else 0,
            "end_index":   (start + len(page_runs)) if page_runs else 0,
            "has_older":   start + page_size < total,
            "has_newer":   page > 0,
        })

    def _training_run_rows(started_at: float, ended_at: float, *, step_id: str = ""):
        """Windowed training-log records for one run, with the ensemble active-
        member fallback. Returns ``(rows, member_label)`` (rows empty if none in
        the window). Shared by the training-plot PNG + training-curve JSON
        endpoints. The trainer writes ``training_log.csv`` (append-only across
        sessions sharing the ckpt dir); we fetch it over SSH (cap 50k lines) and
        keep only rows inside ``[started_at, ended_at]``."""
        ssh = STATE.ssh
        if ssh is None:
            raise SSHError("not connected")
        cfg = fasrc_config.load()
        base = cfg.ckpt_dir.rstrip("/")
        csv_path   = f"{base}/{TrainingLog.FILENAME}"
        jsonl_path = f"{base}/training_log.jsonl"

        def _windowed(csv_p: str, jsonl_p: str = "") -> list[dict]:
            parts = [f" if [ -f {shlex.quote(csv_p)} ]; then "
                     f"head -n 50000 {shlex.quote(csv_p)};"]
            if jsonl_p:
                parts.append(f" elif [ -f {shlex.quote(jsonl_p)} ]; then "
                             f"head -n 50000 {shlex.quote(jsonl_p)};")
            parts.append(" fi")
            try:
                rc, text, _err = ssh.run(
                    "{" + "".join(parts) + " ; }; exit 0", timeout=30)
            except (subprocess.TimeoutExpired, SSHError):
                return []
            if rc != 0 or not text.strip():
                return []
            allr = fasrc_log_parser.parse_training_log(text, max_records=10_000_000)
            return [r for r in allr
                    if started_at <= r.get("wall_time", 0.0) <= ended_at]

        member_label = ""
        rows = _windowed(csv_path, jsonl_path)
        if not rows:
            # An ensemble_train run logs into <ckpt parent>/ensemble/member_NN/,
            # not the single-model ckpt dir — so the read above finds nothing in
            # this window. Fall back to the ACTIVE member (the most-recently
            # modified member log) so the live curve works during ensemble runs.
            ens_dir = f"{os.path.dirname(base)}/ensemble"
            pick = (f"ls -t {shlex.quote(ens_dir)}/member_*/"
                    f"{shlex.quote(TrainingLog.FILENAME)} 2>/dev/null | head -n1; "
                    f"exit 0")
            with contextlib.suppress(subprocess.TimeoutExpired, SSHError):
                _rc, picked, _e = ssh.run(pick, timeout=15)
                member_csv = (picked.strip().splitlines()[0].strip()
                              if picked.strip() else "")
                if member_csv:
                    rows = _windowed(member_csv)
                    if rows:
                        member_label = os.path.basename(os.path.dirname(member_csv))
        return rows, member_label

    def _run_window() -> tuple[float, float] | None:
        """Parse + validate the ``started_at``/``ended_at`` run-window query
        params (ongoing run → ended_at = now). ``None`` on a bad/missing start."""
        try:
            started_at = float(request.args.get("started_at", "0"))
            ended_at   = float(request.args.get("ended_at",   "0"))
        except ValueError:
            return None
        if ended_at <= 0:
            ended_at = time.time() + 1.0
        if started_at <= 0:
            return None
        return started_at, ended_at

    #: Curve fields surfaced to the browser plot (per-eval training records).
    _CURVE_FIELDS = ("step", "psnr_stretched", "psnr_raw", "loss",
                     "psnr_vis", "psnr_y_e", "psnr_j_e", "psnr_h_e")

    @app.route("/api/fasrc/runs/training-curve.json")
    @requires_fasrc
    def api_fasrc_runs_training_curve():
        """Per-step training records for one run's wall-time window, as JSON, so
        the browser draws the curves live (no server-side matplotlib). Empty
        ``records`` while the run hasn't logged an eval yet — not an error."""
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        win = _run_window()
        if win is None:
            return jsonify({"ok": False, "error": "bad/missing started_at"}), 400
        rows, member_label = _training_run_rows(
            *win, step_id=(request.args.get("step_id") or "").strip()
        )
        # Downsample to ~600 points so a long run stays a light payload + fast
        # redraw; the newest point is always kept.
        if len(rows) > 600:
            stride = (len(rows) + 599) // 600
            rows = rows[::stride] + ([rows[-1]] if (len(rows) - 1) % stride else [])
        records = [{k: r.get(k) for k in _CURVE_FIELDS} for r in rows]
        return jsonify({"ok": True, "member": member_label, "records": records})

    @app.route("/api/fasrc/runs/log")
    @requires_fasrc
    def api_fasrc_runs_log():
        """Tail of one log file on FASRC.

        Path is supplied by the client (echoed back from ``/api/fasrc/runs``).
        We verify it falls under the configured logs dir and ends in
        ``.out`` / ``.err`` before reading — a stronger guarantee than
        relying on the URL not containing ``..``.
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        path = (request.args.get("path") or "").strip()
        try:
            lines = int(request.args.get("lines", 1000))
        except ValueError:
            lines = 1000
        lines = max(50, min(lines, 10_000))
        if not path:
            return jsonify({"ok": False, "error": "missing path"}), 400
        if not (path.endswith(".out") or path.endswith(".err")):
            return jsonify({"ok": False, "error": "path must end in .out or .err"}), 400
        cfg = fasrc_config.load()
        log_root = f"{cfg.repo_path}/{cfg.logs_subdir}/"
        if not path.startswith(log_root):
            return jsonify({"ok": False, "error": f"path must live under {log_root}"}), 400
        # Defensive: reject any sneaky path components.
        if ".." in path.split("/"):
            return jsonify({"ok": False, "error": "bad path"}), 400

        # ---- paginated mode -------------------------------------------------
        # ``page`` counts windows of ``page_size`` lines from the END of the
        # file: page 0 = the newest lines, page 1 = the previous block, … up
        # to the very first lines. Lets the user walk all the way back.
        page_param = request.args.get("page")
        if page_param is not None:
            try:
                page = max(0, int(page_param))
            except ValueError:
                page = 0
            try:
                page_size = int(request.args.get("page_size", lines))
            except ValueError:
                page_size = lines
            page_size = max(50, min(page_size, 10_000))
            rc, out_wc, _e = STATE.ssh.run(
                f"[ -f {shlex.quote(path)} ] && wc -l < {shlex.quote(path)} "
                f"|| echo 0", timeout=15)
            try:
                total = int((out_wc or "0").strip().split()[0])
            except (ValueError, IndexError):
                total = 0
            end_line = total - page * page_size
            start_line = max(1, end_line - page_size + 1)
            if total <= 0 or end_line < 1:
                content, start_line, end_line = "", 0, 0
            else:
                # sed window; ``Nq`` quits after the last wanted line so we
                # don't scan the whole file for early pages.
                rc2, content, _e2 = STATE.ssh.run(
                    f"sed -n '{start_line},{end_line}p;{end_line}q' "
                    f"{shlex.quote(path)} 2>/dev/null || true", timeout=20)
            return jsonify({
                "ok":          True,
                "path":        path,
                "page":        page,
                "page_size":   page_size,
                "total_lines": total,
                "start_line":  start_line,
                "end_line":    end_line,
                "has_older":   start_line > 1,
                "has_newer":   page > 0,
                "content":     content,
            })

        # ---- legacy tail mode (no ``page``) --------------------------------
        cmd = (
            f"{{ [ -f {shlex.quote(path)} ] && "
            f"  tail -n {lines} {shlex.quote(path)} 2>/dev/null || true ; "
            f"}}; exit 0"
        )
        rc, out, _err = STATE.ssh.run(cmd, timeout=20)
        if rc != 0:
            return jsonify({"ok": False, "error": "ssh tail failed"}), 500
        return jsonify({"ok": True, "path": path, "lines": lines,
                        "content": out})

    # ---- checkpoint auto-mirror -------------------------------------------

    @app.route("/api/fasrc/mirror/status")
    def api_fasrc_mirror_status():
        s = MIRROR.status
        return jsonify({
            "enabled":     s.enabled,
            "last_run_at": s.last_run_at,
            "last_rc":     s.last_rc,
            "last_error":  s.last_error,
            "last_stdout": s.last_stdout,
            "remote_dir":  s.remote_dir,
            "local_dir":   s.local_dir,
            "period_seconds": MIRROR.period,
        })

    @app.route("/api/fasrc/mirror/trigger", methods=["POST"])
    @requires_fasrc
    def api_fasrc_mirror_trigger():
        """One-shot rsync from remote ckpt dir → local mirror.

        Used by the Training tab's "Sync now" button AND the Logs
        tab's "Pull checkpoints" button. ``MIRROR.trigger`` runs
        synchronously, so the caller learns the final ``last_rc`` /
        ``last_error`` straight from the response without polling.
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        MIRROR.trigger()
        s = MIRROR.status
        return jsonify({
            "ok":          (s.last_rc == 0),
            "last_rc":     s.last_rc,
            "last_error":  s.last_error,
            "last_stdout": s.last_stdout,
            "remote_dir":  s.remote_dir,
            "local_dir":   s.local_dir,
            "last_run_at": s.last_run_at,
        })

    # ---- conda env update -------------------------------------------------

    def _build_env_update_cmd(cfg) -> str:
        """`module load python` + `yes | mamba env update -p … -f environment.yml`.

        FASRC uses lmod, which is exposed as the ``module`` shell function
        once ``/etc/profile.d/lmod.sh`` is sourced — non-interactive SSH
        bash doesn't pull that automatically, so we do it ourselves.
        The ``yes |`` keeps mamba 2.x's "Proceed ([y]/n)?" prompt from
        stalling the stream.
        """
        return (
            "set -o pipefail; "
            "[ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh; "
            f"cd {shlex.quote(cfg.repo_path)} && "
            "module purge 2>/dev/null || true; "
            "module load python && "
            "echo '--- mamba: '$(which mamba) && "
            f"yes | mamba env update -p {shlex.quote(cfg.conda_env_path)} "
            "-f environment.yml 2>&1"
        )

    @app.post("/api/fasrc/env-update")
    @requires_fasrc
    def api_fasrc_env_update():
        """Run ``yes | mamba env update`` on FASRC as a local job.

        POST-only (it mutates the cluster env, so it must sit behind the
        cross-origin mutation guard). Returns ``{ok, job_id}``; the job
        (``kind="fasrc-env-update"``) streams the remote output into its
        log line by line, ends ``done`` with ``result={exit_code, lines}``,
        or ``failed`` when the remote pipeline exits non-zero.

        Cancel (``POST /api/jobs/<id>/cancel``) lands within one heartbeat
        (``_ENV_UPDATE_HEARTBEAT_S``, even while mamba prints nothing) and
        closes the stream, which stops the local ``ssh`` client. The remote
        side gets no signal from that (no pty); the heartbeat watchdog
        (``_env_update_watchdog``) sees its next write fail and kills the
        remote process group, so ``mamba`` stops within about one more
        heartbeat. A cancel can therefore leave the env half-updated; re-run
        the update to finish it.
        """
        ssh = STATE.ssh
        cfg = fasrc_config.load()
        # ``yes | mamba`` under ``pipefail`` always ends 141 (``yes`` dies of
        # SIGPIPE once mamba exits), so report the pipeline's own statuses:
        # the last PIPESTATUS entry is mamba's exit (or the failing earlier
        # step's, e.g. ``module load``, when the pipeline never ran).
        cmd = (_env_update_watchdog(_ENV_UPDATE_HEARTBEAT_S)
               + _build_env_update_cmd(cfg)
               + f'; echo "{_ENV_UPDATE_EXIT_MARKER}=${{PIPESTATUS[*]}}"')

        def run(cap):
            cap.write(f"$ remote: cd {cfg.repo_path}\n")
            cap.write("$ module load python\n")
            cap.write(f"$ yes | mamba env update -p {cfg.conda_env_path} "
                      "-f environment.yml\n")
            exit_code: int | None = None
            lines = 0
            with contextlib.closing(ssh.stream(cmd)) as stream:
                for raw in stream:
                    line = raw.replace("\r", "")
                    if _ENV_UPDATE_ALIVE_MARKER in line:
                        # A heartbeat, possibly glued to a partial line
                        # (mamba's prompt has no newline): drop the marker,
                        # tick so a pending cancel lands, keep any text.
                        line = line.replace(_ENV_UPDATE_ALIVE_MARKER, "")
                        if not line.strip():
                            cap.tick(lines, 0, "mamba env update")
                            continue
                    if line.startswith(f"{_ENV_UPDATE_EXIT_MARKER}="):
                        statuses = line.split("=", 1)[1].split()
                        with contextlib.suppress(ValueError, IndexError):
                            exit_code = int(statuses[-1])
                        continue
                    cap.write(line + "\n")
                    lines += 1
                    cap.tick(lines, 0, "mamba env update")
            if exit_code is None:
                raise RuntimeError(
                    "remote env update ended without reporting an exit code "
                    "(connection dropped?)")
            if exit_code != 0:
                raise RuntimeError(
                    f"remote env update failed with exit code {exit_code}")
            return {"exit_code": exit_code, "lines": lines}

        job_id = JOB_REGISTRY.spawn(
            "FASRC: update conda environment", run, kind="fasrc-env-update",
        )
        return jsonify({"ok": True, "job_id": job_id})
