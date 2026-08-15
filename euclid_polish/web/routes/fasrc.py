"""fasrc routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

import contextlib
import csv
import io as _io
import json
import os
import shlex
import subprocess
import tempfile
import threading as _t
import time
import traceback
from typing import Any

from flask import Response, abort, jsonify, render_template, request, stream_with_context

from euclid_polish.observability.training_log import TrainingLog
from euclid_polish.training.log_plot import plot_training_records
from euclid_polish.web import (
    experimental,
    fasrc_config,
    fasrc_jobs,
    fasrc_log_parser,
    fasrc_queue,
    job_config,
)
from euclid_polish.web.fasrc_mirror import MIRROR
from euclid_polish.web.fasrc_pipeline import REGISTRY as STEP_REGISTRY
from euclid_polish.web.fasrc_pipeline import StepResources
from euclid_polish.web.job_status import JobStatusFetcher
from euclid_polish.web.remote import STATE, SSHConfig, SSHError, SSHSession


def register(app):
    # =========================================================================
    # FASRC tab — Bitwarden-driven SSH ControlMaster, SLURM submission,
    # live log streaming, checkpoint auto-mirror.
    # =========================================================================

    @app.route("/fasrc")
    def fasrc_page():
        # Don't call STATE.public_status() here — it runs ``bw --version``
        # and ``ssh -O check`` subprocesses (up to ~600 ms combined on a
        # warm cache, several seconds on a cold one) and the template
        # doesn't even use the result. The page's own JS fetches connection
        # state via /api/fasrc/status after the DOM loads, which is async
        # and doesn't block render.
        cfg = fasrc_config.load()
        return render_template(
            "fasrc.html",
            cfg=cfg,
            recent=fasrc_jobs.DB.list_recent(20),
        )

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
        return jsonify(STATE.public_status())

    @app.route("/api/fasrc/connect", methods=["POST"])
    def api_fasrc_connect():
        cfg = fasrc_config.load()
        if not cfg.ssh_user:
            return jsonify({"ok": False,
                            "error": "set ssh_user in Settings first"}), 400
        STATE.ssh = SSHSession(SSHConfig(
            user=cfg.ssh_user, host=cfg.ssh_host,
            socket=cfg.control_socket,
            control_persist=cfg.control_persist,
        ))
        try:
            STATE.ssh.connect()
        except SSHError as e:
            STATE.ssh = None
            return jsonify({"ok": False, "error": str(e)}), 400
        STATE.connected_at = time.time()
        # Catch up on any jobs that finished while the server was offline:
        # squeue no longer lists them, so reconcile marks them DONE and
        # the ssh-passing path fetches their sacct accounting into the
        # CSV log. Best-effort — failures here must not block connect.
        with contextlib.suppress(Exception):
            fasrc_jobs.sync_pending_on_connect(STATE.ssh)
        return jsonify({"ok": True, "status": STATE.public_status()})

    @app.route("/api/fasrc/disconnect", methods=["POST"])
    def api_fasrc_disconnect():
        if STATE.ssh:
            STATE.ssh.disconnect()
        STATE.ssh = None
        STATE.connected_at = None
        MIRROR.stop()
        return jsonify({"ok": True, "status": STATE.public_status()})

    # ---- remote info ------------------------------------------------------

    @app.route("/api/fasrc/git-status")
    def api_fasrc_git_status():
        if not STATE.ssh or not STATE.ssh.is_connected():
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
        rc, out, err = STATE.ssh.run(cmds, timeout=30)
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
    def api_fasrc_git_pull():
        """``git pull`` + auto-update conda env when ``environment.yml`` moved.

        Returns ``env_update_needed: True`` whenever the pull's diff
        touches ``environment.yml``; the UI then kicks off the
        ``/api/fasrc/env-update`` SSE stream automatically so the user
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

    _parse_slurm_time = fasrc_jobs.parse_slurm_time

    # ---- submission -------------------------------------------------------

    @app.route("/api/fasrc/eta")
    def api_fasrc_eta():
        try:
            steps = int(request.args.get("steps", 0))
        except ValueError:
            steps = 0
        spt = fasrc_jobs.secs_per_step_history()
        return jsonify({
            "secs_per_step": spt,
            "history_n":     len(fasrc_jobs.DB.list_completed(8)),
            "eta_seconds":   fasrc_jobs.eta_for_submission(steps),
        })

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

    def _spec_label(kind, step_ref, form):
        explicit = (form.get("label") or "").strip()
        if explicit:
            return explicit
        try:
            step = STEP_REGISTRY.get(step_ref)
        except KeyError:
            step = None
        if kind == "synthetic":
            steps = form.get("steps") or fasrc_config.load().steps
            return f"{step.label if step else 'synthetic'}: {steps} steps"
        return step.label if step else f"step {step_ref}"

    def _build_and_submit(kind, step_ref, form):
        """Render + sbatch a job from a stored spec → (slurm_id, payload)."""
        cfg = fasrc_config.load()
        if kind == "synthetic":
            try:
                step = STEP_REGISTRY.get(step_ref)
            except KeyError:
                step = STEP_REGISTRY.get("synthetic_generate")
                step_ref = "synthetic_generate"
            resources = StepResources.from_form(form, step.defaults)
            # Partition is fixed per job type — never taken from the form.
            resources.partition = step.defaults.partition
            params = {
                # Generation is standalone — no training knobs (batch_size /
                # steps). Training runs separately via the ensemble step.
                "n_train":     int(form.get("n_train",    cfg.n_train)),
                "n_valid":     int(form.get("n_valid",    cfg.n_valid)),
                "n_test":      int(form.get("n_test",     cfg.n_test)),
                "image_size":  int(form.get("image_size", cfg.image_size)),
                "extra_flags": (form.get("extra_flags", "") or "").strip(),
                # "Override existing data" checkbox → run_pipeline --force
                # (regenerate from scratch instead of resuming prior shards).
                "force": str(form.get("force", "")).strip().lower() in (
                    "1", "true", "yes", "on"),
                # "on-the-fly training" checkbox → --onthefly-train (train
                # split generated clean-only; no hr/dirty — training builds
                # both live from clean_train).
                "onthefly_train": str(form.get("onthefly_train", "")
                                      ).strip().lower() in (
                    "1", "true", "yes", "on"),
            }
            params.update(resources.to_dict())
            label = _spec_label(kind, step_ref, form)
            built = step.build_sbatch_body(
                params=params, resources=resources, cfg=cfg, label=label)
            return fasrc_jobs.submit_sbatch_script(
                STATE.ssh, cfg=cfg, built=built, label=label,
                params=params, step_id=step.step_id)
        # Pipeline step
        step = STEP_REGISTRY.get(step_ref)
        form2 = dict(form)
        # Partition is fixed per job type — force the step's value even on
        # queued specs built from an older form.
        form2["partition"] = step.defaults.partition
        if step.fixed_cpus is not None:
            form2["n_cpus"] = str(step.fixed_cpus)
        if step.fixed_gpus is not None:
            form2["n_gpus"] = str(step.fixed_gpus)
        resources = StepResources.from_form_strict(form2)
        if step.fixed_cpus is not None:
            resources.n_cpus = int(step.fixed_cpus)
        if step.fixed_gpus is not None:
            resources.n_gpus = int(step.fixed_gpus)
        label = _spec_label(kind, step_ref, form2)
        built = step.build_sbatch_body(
            params=form2, resources=resources, cfg=cfg, label=label)
        # Array steps resolve member names/base seeds while rendering. Persist
        # those prepared values so monitoring can map task indices to members
        # and a queued/retried submission remains reproducible.
        params_for_db = dict(built.get("params", form2))
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

    def _submit_or_queue(kind, step_ref, form):
        """Submit immediately if the single lane is free, else enqueue."""
        label = _spec_label(kind, step_ref, form)
        spec = {"kind": kind, "step": step_ref, "form": form}
        # Validate that both fitted population artifacts are active before
        # creating a queue entry. They are resolved and embedded again at
        # promotion, so no Config/legacy population fallback can enter.
        try:
            resolved_step = step_ref
            if kind == "synthetic":
                try:
                    resolved_step = STEP_REGISTRY.get(step_ref).step_id
                except KeyError:
                    resolved_step = "synthetic_generate"
            if resolved_step == "synthetic_generate":
                STEP_REGISTRY.get(resolved_step).prepare_params(dict(form))
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

    @app.route("/api/fasrc/submit", methods=["POST"])
    def api_fasrc_submit():
        """``run_pipeline.py`` submission (API path).

        Submits a ``scripts/run_pipeline.py`` job through the shared
        sbatch helper. The only registered run_pipeline step is
        ``synthetic_generate``; an optional ``step`` form field can name
        another registered step, otherwise it defaults to that.
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        confirm_err = _require_confirm(request.form)
        if confirm_err is not None:
            return confirm_err

        cfg = fasrc_config.load()
        f = request.form
        step_name = f.get("step") or f.get("preset") or "synthetic_generate"
        try:
            step = STEP_REGISTRY.get(step_name)
        except KeyError:
            # Unknown names fall back to the synthetic generator so a stale
            # frontend can't 404 the submit.
            step = STEP_REGISTRY.get("synthetic_generate")
            step_name = "synthetic_generate"

        # Resources from the form, falling back to the step's defaults
        # (so the legacy form fields keep working even when fields are
        # left blank).
        try:
            resources = StepResources.from_form(f, step.defaults)
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400

        # Training params — passed through to ``build_command``. We
        # validate the numerics here so a bad form field 400s instead of
        # blowing up inside the renderer.
        try:
            params = {
                "n_train":     int(f.get("n_train",    cfg.n_train)),
                "n_valid":     int(f.get("n_valid",    cfg.n_valid)),
                "image_size":  int(f.get("image_size", cfg.image_size)),
                "batch_size":  int(f.get("batch_size", cfg.batch_size)),
                "steps":       int(f.get("steps",      cfg.steps)),
                "extra_flags": f.get("extra_flags", "").strip(),
            }
        except (TypeError, ValueError) as e:
            return jsonify({"ok": False, "error": f"bad form field: {e}"}), 400
        params.update(resources.to_dict())

        # Validation above (resources + params) has passed; hand off to the
        # local queue: submit now if the lane is free, else enqueue. The
        # sbatch body is (re)built from the form by _build_and_submit.
        return _submit_or_queue("synthetic", step_name, f.to_dict())

    # =========================================================================
    # Pipeline steps (generic FASRC submissions; URL prefix /api/fasrc/hst/
    # is historical — it serves every registered step, not just HST ones)
    # =========================================================================

    @app.route("/api/fasrc/hst/status")
    def api_fasrc_hst_status():
        """Per-step: name, defaults, last-runtime median, on-disk status.

        Uses ``STATE.ssh`` to ``test`` for each artifact's existence; if
        SSH isn't connected we return only the static defaults so the UI
        can still render its forms.
        """
        cfg_loaded = fasrc_config.load()
        ssh_ok = bool(STATE.ssh and STATE.ssh.is_connected())

        steps_payload = []
        for step in STEP_REGISTRY.all():
            # EXPERIMENTAL lanes (HST / star-anchor / round-trip): their
            # steps are hidden from the UI while the feature flag is off —
            # see euclid_polish.web.experimental.
            if step.experimental and not experimental.EXPERIMENTAL_LANES_ENABLED:
                continue
            steps_payload.append({
                "step_id":     step.step_id,
                "label":       step.label,
                "needs_gpu":   step.needs_gpu,
                "fixed_cpus":  step.fixed_cpus,
                "fixed_gpus":  step.fixed_gpus,
                "defaults":    step.defaults.to_dict(),
            })

        # Cheap probes for "does this artifact exist on FASRC?" — single
        # ``test -e`` per check, batched in one SSH round-trip. Keep
        # this list in sync with the ``produces`` map in fasrc.html
        # (the JS side maps each step_id to one of these keys).
        artifacts = {
            "tiles": None, "psf": None, "kernel": None,
            "records": None, "ckpt": None,
            # Round-trip pipeline artifacts (Chunk C3 + web wiring):
            #   euclid_sky      — sky-position catalog written by the
            #                     sky-download step (cutouts arrive in
            #                     subdirs whose names depend on the
            #                     requested size; gate on the catalog
            #                     instead, which exists as soon as the
            #                     position generation has run).
            #   roundtrip_records — LR-only TFRecord produced by the
            #                     stack/chop step. Single-file probe so
            #                     the UI can flip ✓ as soon as the
            #                     first shard lands.
            "euclid_sky": None, "roundtrip_records": None,
            # Per-page Euclid star-cutout pipeline:
            #   euclid_cutouts — VIS cutout subdir, written by the
            #                    download_euclid_cutouts step.
            #   euclid_psf     — VIS empirical ePSF, written by the
            #                    extract_euclid_psf (all-band) step.
            "euclid_cutouts": None, "euclid_psf": None,
            # Synthetic generation (/sky page).
            "synthetic_records": None,
        }
        if ssh_ok:
            paths = {
                "ckpt":    f"{cfg_loaded.ckpt_dir}/checkpoint",
                "euclid_cutouts":
                    f"{cfg_loaded.data_dir}/euclid_stars/cutouts/VIS",
                "euclid_psf":
                    f"{cfg_loaded.data_dir}/euclid_psf/euclid_psf_VIS.fits",
                "synthetic_records":
                    f"{cfg_loaded.data_dir}/images/records_v2/clean_train.tfrecord",
            }
            # EXPERIMENTAL-lane artifacts (HST / round-trip): only probed
            # when the lanes are enabled — no point spending SSH time on
            # features the UI hides.
            if experimental.EXPERIMENTAL_LANES_ENABLED:
                paths.update({
                    "tiles":   f"{cfg_loaded.data_dir}/hst_hlsp/download_summary.json",
                    "psf":     f"{cfg_loaded.data_dir}/hst_psf/F814W.fits",
                    "kernel":  f"{cfg_loaded.data_dir}/hst_psf/diff_kernel_VIS.fits",
                    "records": f"{cfg_loaded.data_dir}/images/records_v2_hst/clean_train.tfrecord",
                    "euclid_sky":
                        f"{cfg_loaded.data_dir}/euclid_sky/sky_positions.csv",
                    "roundtrip_records":
                        f"{cfg_loaded.data_dir}/images/records_v2_euclid_roundtrip/dirty_train.tfrecord",
                })
            probe = " && ".join(
                f"(test -e {shlex.quote(p)} && echo {k}=1 || echo {k}=0)"
                for k, p in paths.items()
            )
            try:
                rc, out, _err = STATE.ssh.run(probe, timeout=10)
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
                "logs_dir":    "logs/hst_pipeline",
            },
        })

    @app.route("/api/fasrc/hst/<step_id>/submit", methods=["POST"])
    def api_fasrc_hst_submit(step_id: str):
        """Generic submission for any HST-pipeline step.

        Two defences, in order of evaluation:

        1. ``confirm=yes`` token — proves the frontend dialog was
           shown (catches stale-JS tabs and most accidental submits).
        2. SSH connected (sanity check before the work starts).

        Any failure returns 400 and DOES NOT touch the SSH session,
        so no sbatch call, no script write, nothing reaches FASRC."""

        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        try:
            step = STEP_REGISTRY.get(step_id)
        except KeyError:
            return jsonify({"ok": False, "error": f"unknown step: {step_id}"}), 404
        # EXPERIMENTAL-lane steps (HST / star-anchor / round-trip) are
        # disabled for now — refuse the submit so a stale tab can't launch
        # one. See euclid_polish.web.experimental.
        if step.experimental and not experimental.EXPERIMENTAL_LANES_ENABLED:
            return jsonify({"ok": False, "error":
                            f"step {step_id!r} is experimental and currently "
                            "disabled"}), 404

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
        # what it needs (e.g. ``n_tiles``, ``hst_fraction``).
        # Validation above (step, confirm, resources) has passed; hand off
        # to the local queue: submit now if the lane is free, else enqueue.
        return _submit_or_queue("hst", step_id, form)

    @app.route("/api/fasrc/refresh-accounting", methods=["POST"])
    def api_fasrc_refresh_accounting():
        """One-shot: re-pull sacct for every finalised job and re-record.

        Use after a change to how a post-mortem stat is computed (e.g. the
        CPU-utilisation fix) to backfill existing history rows."""
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        return jsonify(fasrc_jobs.refresh_all_post_mortems(STATE.ssh))

    @app.route("/api/fasrc/hst/<step_id>/history", methods=["GET", "POST"])
    def api_fasrc_hst_step_history(step_id: str):
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

    @app.route("/api/fasrc/jobs")
    def api_fasrc_jobs_list():
        rows = fasrc_jobs.DB.list_recent(30)
        for r in rows:
            r["eta_seconds"] = (
                fasrc_jobs.eta_for_running(r)
                if r["state"] in ("RUNNING", "PENDING") else None
            )
        return jsonify({"jobs": rows})

    @app.route("/api/fasrc/current-submission")
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

        Used by the FASRC page's "Current Submission" tab. Replaces the
        WDSR-specific ``/api/fasrc/training-status`` for the general job
        case; the training-status endpoint stays for the trainer's own
        live-metrics view.
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400

        # Reconcile DB rows against squeue first — without this, a job
        # that already finished still shows up as RUNNING in the DB and
        # the tab would lie. One squeue call per refresh; the same one
        # the Logs tab already makes. A slow login node can time this out —
        # skip the reconcile for this tick (flagging the data stale) rather
        # than 500 the poll; the next tick retries.
        stale = False
        try:
            rc_q, out_q, _err_q = STATE.ssh.run(
                f"squeue -r -h -u $USER --format='{fasrc_jobs.SQUEUE_FMT}'",
                timeout=15,
            )
        except (subprocess.TimeoutExpired, SSHError):
            rc_q, out_q, stale = 1, "", True
        squeue_rows: list[dict[str, Any]] = []
        if rc_q == 0:
            squeue_rows = fasrc_jobs.parse_squeue(out_q)
            fasrc_jobs.reconcile_with_squeue(squeue_rows, ssh=STATE.ssh)
            # Advance the local queue (promote on success / halt on failure)
            # off the same reconcile that just refreshed job states.
            _queue_tick()

        queue_public = fasrc_queue.QUEUE.public()

        # Pick the newest still-live row. ``list_recent`` orders by
        # submitted_at DESC, so the first matching row IS the newest.
        recent = fasrc_jobs.DB.list_recent(limit=10)
        current_row = next(
            (r for r in recent if r.get("state") in ("PENDING", "RUNNING")),
            None,
        )
        if current_row is None:
            return jsonify({"ok": True, "current": None, "queue": queue_public,
                            "stale": stale})
        if stale:
            # Login node slow this tick — return the last-known DB row without
            # the extra SSH calls (squeue merge + event fetch) that would also
            # hang and 500. The next poll fills the live fields back in.
            return jsonify({"ok": True, "stale": True, "queue": queue_public,
                            "current": {"job": current_row, "status": None}})

        # Merge live squeue fields into the row so the UI sees the
        # current ``start_time`` (PENDING jobs only), ``reason`` (why
        # SLURM hasn't started it: Priority / Resources / …), updated
        # elapsed ``time``, and assigned ``nodes``. reconcile_with_squeue
        # only persists state + started_at, so these have to be merged
        # in at the response layer.
        jid = str(current_row.get("jobid", "")).strip()
        live_rows = fasrc_jobs.array_squeue_rows(jid, squeue_rows)
        live = next((r for r in live_rows
                     if r.get("state") == "RUNNING"), None)
        live = live or next(iter(live_rows), None)
        if live is not None:
            for k in ("start_time", "reason", "nodes", "time", "time_limit"):
                v = live.get(k)
                if v is not None and v != "":
                    current_row[k] = v

        # Fold the live event stream into a JobStatus. Array submissions have
        # one Reporter stream per model; expose them separately rather than
        # inventing a misleading aggregate training curve.
        fetcher = JobStatusFetcher(ssh=STATE.ssh)
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
            fasrc_jobs.fetch_live_jobstats(STATE.ssh, jid)
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
        fetcher = JobStatusFetcher(ssh=STATE.ssh)
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
            # Modern pipeline logs live in ``logs/hst_pipeline`` while older
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

    @app.route("/api/fasrc/runs/ckpt-bundle.tar")
    def api_fasrc_runs_ckpt_bundle():
        """Stream a tar of the FASRC ckpt dir so the user can download
        the trained model after a job completes.

        We tar on the remote (one ssh + tar pipeline) and pipe bytes
        back through the ControlMaster — no temp file involved. Bundle
        size is bounded by ``max_to_keep=3`` in the trainer, so usually
        a few × 7 MB plus the training_log.
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        cfg = fasrc_config.load()
        ckpt_dir = cfg.ckpt_dir
        # Defensive: refuse if cfg.ckpt_dir points anywhere weird.
        if not ckpt_dir or ".." in ckpt_dir.split("/"):
            return jsonify({"ok": False, "error": "invalid ckpt_dir"}), 400
        parent = os.path.dirname(ckpt_dir.rstrip("/")) or "/"
        leaf   = os.path.basename(ckpt_dir.rstrip("/"))
        # ``tar -C parent leaf`` makes the tarball self-contained: it
        # unpacks into ``leaf/`` regardless of where the user extracts
        # it. ``--ignore-failed-read`` keeps going if a single ckpt file
        # is being rewritten while we tar.
        cmd = (
            f"tar -C {shlex.quote(parent)} -cf - "
            f"  --ignore-failed-read {shlex.quote(leaf)} 2>/dev/null"
        )
        rc, out, err = STATE.ssh.run(cmd, timeout=300, binary=True)
        if rc != 0 or not out:
            return jsonify({
                "ok": False,
                "error": f"tar failed: {err[:200] if err else 'no output'}",
            }), 500
        filename = f"{leaf}.tar"
        return Response(
            out, mimetype="application/x-tar",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length":      str(len(out)),
            },
        )

    def _training_run_rows(started_at: float, ended_at: float, *, step_id: str = ""):
        """Windowed training-log records for one run, with the ensemble active-
        member fallback. Returns ``(rows, member_label)`` (rows empty if none in
        the window). Shared by the training-plot PNG + training-curve JSON
        endpoints. The trainer writes ``training_log.csv`` (append-only across
        sessions sharing the ckpt dir); we fetch it over SSH (cap 50k lines) and
        keep only rows inside ``[started_at, ended_at]``."""
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
                rc, text, _err = STATE.ssh.run("{" + "".join(parts) + " ; }; exit 0",
                                               timeout=30)
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
            ens_dir = (
                f"{cfg.data_dir.rstrip('/')}/experiments/lens_isolation/ensemble"
                if step_id == "lens_isolation_train"
                else f"{os.path.dirname(base)}/ensemble"
            )
            pick = (f"ls -t {shlex.quote(ens_dir)}/member_*/"
                    f"{shlex.quote(TrainingLog.FILENAME)} 2>/dev/null | head -n1; "
                    f"exit 0")
            with contextlib.suppress(subprocess.TimeoutExpired, SSHError):
                _rc, picked, _e = STATE.ssh.run(pick, timeout=15)
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

    @app.route("/api/fasrc/runs/training-plot.png")
    def api_fasrc_runs_training_plot():
        """PNG of the validation log restricted to one run's wall-time window.
        (Kept for the classic page; the SPA renders the curve client-side.)"""
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        win = _run_window()
        if win is None:
            return jsonify({"ok": False, "error": "bad/missing started_at"}), 400
        started_at, ended_at = win
        rows, member_label = _training_run_rows(started_at, ended_at)
        if not rows:
            return jsonify({"ok": False,
                            "error": f"no training-log rows in window "
                                     f"[{started_at:.0f}, {ended_at:.0f}]"}), 404

        # Render to a throwaway tempfile OUTSIDE data/vis — this is a
        # per-request scratch render (the page re-polls it every minute),
        # and a stable path under data/ used to trip the test suite's
        # data-dir immutability guard whenever a live WebUI overwrote it
        # mid-pytest-run.
        fd, tmp_png = tempfile.mkstemp(suffix=".png",
                                       prefix="euclid_training_plot_")
        os.close(fd)
        try:
            plot_training_records(
                rows, tmp_png,
                title_suffix=(
                    f"\n(ensemble {member_label}, this run: {len(rows)} evals)"
                    if member_label else
                    f"\n(this run only: {len(rows)} evals)"
                ),
            )
            with open(tmp_png, "rb") as fh:
                data = fh.read()
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_png)
        return Response(data, mimetype="image/png",
                        headers={"Cache-Control": "no-cache"})

    @app.route("/api/fasrc/runs/log")
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

    # ---- parsed live status (.out + .err + training_log) -----------------
    #
    # The sidebar polls this every couple of seconds AND every page in the
    # app pulls it on initial render — without a cache that's a fresh
    # squeue + multi-file tail per poll, which is what makes the UI feel
    # sluggish. A 2-second TTL coalesces bursts and keeps live progress
    # visibly fresh (next poll arrives just after expiry).

    _TRAINING_STATUS_CACHE: dict[str, Any] = {"at": 0.0, "resp": None}
    _TRAINING_STATUS_TTL_S: float = 2.0

    @app.route("/api/fasrc/training-status")
    def api_fasrc_training_status():
        """Single JSON dict the UI polls every few seconds.

        Identifies the currently running job (RUNNING state in squeue,
        cross-referenced against local sqlite), reads the tail of its
        ``.out`` / ``.err`` / ``training_log`` over SSH, and returns a
        parsed summary. Errors are reported in-band as
        ``{"ok": False, "error": ...}`` rather than raising — a 5xx
        here would just look like the dashboard "disconnecting" to
        the user, when really the SSH is fine and only one log read
        misbehaved.
        """
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400

        # Serve from the cache if a recent (<2 s) response is available.
        now = time.monotonic()
        if (_TRAINING_STATUS_CACHE["resp"] is not None
                and now - _TRAINING_STATUS_CACHE["at"] < _TRAINING_STATUS_TTL_S):
            return _TRAINING_STATUS_CACHE["resp"]

        try:
            resp = _build_training_status()
        except subprocess.TimeoutExpired:
            # FASRC login-node / ControlMaster lag — transient and expected.
            # Don't dump a full traceback on every ~3 s heartbeat; one quiet line.
            print("[training-status] FASRC poll timed out — retry next tick")
            resp = jsonify({"ok": False, "error": "fasrc poll timed out",
                            "transient": True}), 200
        except Exception as e:
            traceback.print_exc()
            resp = jsonify({
                "ok":    False,
                "error": f"{type(e).__name__}: {e}",
            }), 200
        _TRAINING_STATUS_CACHE["at"]   = now
        _TRAINING_STATUS_CACHE["resp"] = resp
        return resp

    def _build_training_status():
        cfg = fasrc_config.load()

        # 1. Identify the running job. Trust live squeue > sqlite.
        rc, sq_out, _err = STATE.ssh.run(
            f"squeue -r -h -u $USER --format='{fasrc_jobs.SQUEUE_FMT}'",
            timeout=10,
        )
        # Drive the local submit queue off this poll too — this endpoint is
        # the global ~3 s heartbeat, so the queue promotes/halts even when
        # the user isn't on the Current-Submission tab.
        if rc == 0:
            fasrc_jobs.reconcile_with_squeue(
                fasrc_jobs.parse_squeue(sq_out), ssh=STATE.ssh)
            _queue_tick()
        running_rows = []
        if rc == 0:
            for row in fasrc_jobs.parse_squeue(sq_out):
                if row.get("state") == "RUNNING":
                    running_rows.append(row)
        if not running_rows:
            return jsonify({"ok": True, "running": False,
                            "queue_rows": fasrc_jobs.parse_squeue(sq_out)
                                          if rc == 0 else []})

        # Prefer a row belonging to a submission from this UI. With ``squeue
        # -r`` an array appears as parent_index rows while sqlite stores the
        # numeric parent id, so map the selected live task back to its parent.
        known_ids = [r["jobid"] for r in fasrc_jobs.DB.list_recent(20)]
        live = next((r for r in running_rows
                     if any(r["jobid"] == parent
                            or r["jobid"].startswith(parent + "_")
                            for parent in known_ids)), running_rows[0])
        live_jobid = live["jobid"]
        jobid = next((parent for parent in known_ids
                      if live_jobid == parent
                      or live_jobid.startswith(parent + "_")), live_jobid)
        task_suffix = live_jobid.removeprefix(jobid + "_")
        task_index = int(task_suffix) if task_suffix.isdigit() else None
        stored = fasrc_jobs.DB.get(jobid)
        log_path = (stored or {}).get("log_path") \
                   or f"{cfg.repo_path}/logs/jobs/{live['name']}.out"
        err_path = (stored or {}).get("err_path") \
                   or log_path.replace(".out", ".err")
        events_path = (stored or {}).get("events_path")
        if task_index is not None:
            log_path = fasrc_jobs.expand_array_path(log_path, jobid, task_index)
            err_path = fasrc_jobs.expand_array_path(err_path, jobid, task_index)
            events_path = fasrc_jobs.expand_array_path(
                events_path, jobid, task_index)
        # 2. Fold the job's structured event stream into a JobStatus —
        # the SAME Reporter events the JobStatusCard polls. No log
        # scraping: stage, progress, per-evaluate metrics (loss/PSNR) and
        # the checkpoint marker all come from the events file the trainer
        # writes via :class:`Reporter`.
        elapsed_s = fasrc_jobs.parse_slurm_time(live.get("time"))
        status = JobStatusFetcher(ssh=STATE.ssh).fetch(
            events_path=events_path)

        # Map JobStatus → the dashboard's existing field shape.
        progress = None
        if status.step is not None and status.step.total > 0:
            cur, tot = status.step.current, status.step.total
            progress = {"current": cur, "total": tot,
                        "pct": round(100.0 * cur / tot, 2)}
        # ``stage_index`` is just how far through the stage sequence we are
        # (0-based); ``pipeline_done`` is always False here — this branch
        # only runs while a job is RUNNING in squeue.
        stage_index = max(0, len(status.stages) - 1)

        # 3. Side-effect: keep sqlite up to date so /api/fasrc/jobs is
        # accurate for the recent-submissions panel.
        if progress:
            fasrc_jobs.DB.update_progress(
                jobid, progress["current"], progress["total"])
        if stored and stored["started_at"] is None:
            fasrc_jobs.DB.update_state(
                jobid, state="RUNNING",
                started_at=time.time() - elapsed_s,
            )

        # 4. Activate the auto-mirror during the training stage and trigger
        # an immediate sync whenever a fresh checkpoint marker arrives on
        # the event stream (the trainer emits ``saved`` on each checkpoint
        # eval; fold_events surfaces it as ``last_checkpoint``).
        # ``MIRROR.trigger()`` rsyncs synchronously and can block for
        # minutes on large ckpt dirs — dispatch on a daemon thread so the
        # status poll stays snappy.
        in_training = bool(status.stage
                           and status.stage.lower().startswith("training"))
        if in_training:
            if not MIRROR.status.enabled:
                MIRROR.start()
            if (status.last_checkpoint
                    and MIRROR.status.last_checkpoint_line
                        != status.last_checkpoint):
                MIRROR.status.last_checkpoint_line = status.last_checkpoint
                _t.Thread(target=MIRROR.trigger, daemon=True,
                          name="mirror-trigger").start()

        return jsonify({
            "ok":       True,
            "running":  True,
            "job": {
                "jobid":           jobid,
                "array_task_jobid": live_jobid if task_index is not None else None,
                "name":            live.get("name", ""),
                "state":           live.get("state", ""),
                "elapsed_seconds": elapsed_s,
                "elapsed":         live.get("time", ""),
                "time_limit":      live.get("time_limit", ""),
                "node":            live.get("reason", ""),
                "start_time":      live.get("start_time", ""),
                "log_path":        log_path,
                "err_path":        err_path,
                "label":           (stored or {}).get("label", ""),
                "params":          json.loads((stored or {}).get("params_json") or "null"),
            },
            "stage":             status.stage,
            "stage_index":       stage_index,
            "pipeline_done":     False,
            "progress":          progress,
            "latest_metrics":    status.latest_metrics,
            "last_checkpoint":   status.last_checkpoint,
            "validations":       list(status.metrics),
            "latest_validation": status.latest_metrics,
            "eta_seconds":       status.step_eta_s,
            "queue_rows":        running_rows,
        })

    # ---- live log stream (SSE) -------------------------------------------

    @app.route("/api/fasrc/log/<jobid>")
    def api_fasrc_log_stream(jobid: str):
        if not jobid.isdigit():
            abort(400)
        row = fasrc_jobs.DB.get(jobid)
        if not row:
            abort(404)
        log_path = row["log_path"]
        # Stream both files in case the user wants stderr (`?which=err`).
        which = request.args.get("which", "out")
        if which == "err":
            log_path = row["err_path"]

        def _gen():
            if not STATE.ssh or not STATE.ssh.is_connected():
                yield "event: error\ndata: not connected\n\n"
                return
            # tail with retry — file may not exist until SLURM starts the job.
            cmd = (f"tail -F -n 200 {log_path} 2>/dev/null || "
                   f"(while [ ! -f {log_path} ]; do sleep 2; done && "
                   f" tail -F -n 200 {log_path})")
            try:
                for line in STATE.ssh.stream(cmd):
                    # This stream is the raw-log VIEWER only. Progress is no
                    # longer scraped from log lines — it comes from the
                    # Reporter event stream (folded in JobStatus); the DB
                    # progress is updated by the events-based status poll.
                    # SSE framing: one event per line, multiline data uses
                    # repeated ``data:`` lines.
                    safe = line.replace("\r", "")
                    yield f"data: {safe}\n\n"
            except SSHError as e:
                yield f"event: error\ndata: {e}\n\n"
        return Response(stream_with_context(_gen()),
                        mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

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

    @app.route("/api/fasrc/mirror/start", methods=["POST"])
    def api_fasrc_mirror_start():
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        with contextlib.suppress(ValueError):
            MIRROR.period = max(15, int(request.form.get("period", 60)))
        MIRROR.start()
        return jsonify({"ok": True, "status": api_fasrc_mirror_status().json})

    @app.route("/api/fasrc/mirror/stop", methods=["POST"])
    def api_fasrc_mirror_stop():
        MIRROR.stop()
        return jsonify({"ok": True})

    @app.route("/api/fasrc/mirror/trigger", methods=["POST"])
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

    # ---- per-stage timings (CSV from run_pipeline.py's StageTimer) -------

    @app.route("/api/fasrc/stages/<jobid>")
    def api_fasrc_stages(jobid: str):
        """Parse the remote ``stages_<jobid>.csv`` into JSON rows so the
        UI can render the per-stage breakdown. The CSV lives next to the
        TFRecords on netscratch (see ``run_pipeline.py``'s ``--stages-csv``
        default)."""
        if not jobid.isdigit() and jobid != "local":
            abort(400)
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        cfg = fasrc_config.load()
        path = f"{cfg.data_dir}/images/records_v2/stages_{jobid}.csv"
        rc, out, err = STATE.ssh.run(
            f"if [ -f {shlex.quote(path)} ]; then "
            f"  cat {shlex.quote(path)}; "
            f"else "
            f"  echo MISSING; "
            f"fi", timeout=10,
        )
        if rc != 0:
            return jsonify({"ok": False, "error": err.strip()}), 500
        if out.strip() == "MISSING":
            return jsonify({"ok": True, "path": path, "rows": []})

        reader = csv.DictReader(_io.StringIO(out))
        rows = []
        for r in reader:
            try:
                rows.append({
                    "stage":             r.get("stage", ""),
                    "started_at":        float(r.get("started_at",  "0") or 0),
                    "ended_at":          float(r.get("ended_at",    "0") or 0),
                    "duration_seconds":  float(r.get("duration_seconds", "0") or 0),
                    "params_dependent":  bool(int(r.get("params_dependent", "0") or 0)),
                    "n_train":           r.get("n_train", ""),
                    "n_valid":           r.get("n_valid", ""),
                    "image_size":        r.get("image_size", ""),
                    "batch_size":        r.get("batch_size", ""),
                    "steps":             r.get("steps", ""),
                })
            except (ValueError, TypeError):
                # Skip malformed rows (e.g. a partial write captured mid-flight).
                continue
        return jsonify({"ok": True, "path": path, "rows": rows})

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

    @app.route("/api/fasrc/env-update")
    def api_fasrc_env_update():
        if not STATE.ssh or not STATE.ssh.is_connected():
            return Response(
                "event: error\ndata: not connected\n\n",
                mimetype="text/event-stream", status=400,
            )
        cfg = fasrc_config.load()
        cmd = _build_env_update_cmd(cfg)

        def _gen():
            yield f"data: $ remote: cd {cfg.repo_path}\n\n"
            yield "data: $ module load python\n\n"
            yield (f"data: $ yes | mamba env update -p "
                   f"{cfg.conda_env_path} -f environment.yml\n\n")
            try:
                for line in STATE.ssh.stream(cmd):
                    yield f"data: {line.replace(chr(13), '')}\n\n"
                yield "event: done\ndata: complete\n\n"
            except SSHError as e:
                yield f"event: error\ndata: {e}\n\n"
        return Response(stream_with_context(_gen()),
                        mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})
