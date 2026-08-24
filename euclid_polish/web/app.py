"""
Flask app factory for the EuclidPolish web UI.

Routes live in :mod:`euclid_polish.web.routes` (one module per group, each
exposing ``register(app)``); shared helpers live in
:mod:`euclid_polish.web.helpers`. This module just wires them together and
owns the FASRC SSH gate + the root redirect.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import os as _os
import time

# Force the non-interactive matplotlib backend BEFORE any submodule imports
# pyplot. The job registry plots from worker threads; macOS's default GUI
# backend only works on the main thread and would otherwise crash with
# "Cannot create a GUI FigureManager outside the main thread".
import matplotlib

matplotlib.use("Agg")

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from euclid_polish.web import experimental, fasrc_config, fasrc_jobs
from euclid_polish.web.remote import STATE, SSHConfig, SSHError, SSHSession
from euclid_polish.web.routes import (
    auth,
    catalog,
    config,
    cutouts,
    ensemble,
    evaluation,
    fasrc,
    files,
    galaxy_distributions,
    git,
    hst,
    hstpairs,
    jwst_euclid,
    model,
    population_comparison,
    poster,
    psfs,
    sky,
    star_distribution,
    tng,
    tracking,
    viewer,
    views,
)
from euclid_polish.web.security import register_mutation_guard, validate_bind_host


def _try_startup_ssh_connect() -> str | None:
    """Attempt one SSH connect during app startup. Return None on success.

    Stores the active session on :data:`STATE`. Failures are returned as
    a short error string for the connection-error page to display.

    Honours the ``EUCLID_POLISH_DISABLE_AUTO_SSH=1`` env var as a hard
    kill-switch — when set, this function is a no-op (returns a
    diagnostic string but never opens a socket or touches ``STATE.ssh``).
    This is **load-bearing for tests**: pytest imports ``create_app``
    which used to silently dial out to the user's real FASRC via their
    ControlMaster socket, blowing past any ``STATE.ssh = stub``
    monkeypatch the test had installed and submitting real SLURM jobs
    through every test that posted to a submit endpoint. The env var
    lets the test harness disable the auto-connect entirely so the
    stub stays in effect.
    """
    if _os.environ.get("EUCLID_POLISH_DISABLE_AUTO_SSH", "").strip() in (
        "1", "true", "yes", "on",
    ):
        return ("auto-connect disabled by "
                "EUCLID_POLISH_DISABLE_AUTO_SSH env var (test mode)")
    cfg = fasrc_config.load()
    if not cfg.ssh_user:
        return "ssh_user is unset — open Settings and configure it"
    try:
        STATE.ssh = SSHSession(SSHConfig(
            user=cfg.ssh_user, host=cfg.ssh_host,
            socket=cfg.control_socket,
            control_persist=cfg.control_persist,
        ))
        STATE.ssh.connect()
    except (SSHError, Exception) as e:
        STATE.ssh = None
        return f"{type(e).__name__}: {e}"
    STATE.connected_at = time.time()
    # Same catch-up as in /api/fasrc/connect — jobs that finished
    # while the server was offline get their state + sacct post-mortem
    # recorded now. Best-effort; failures here are swallowed so the
    # auto-connect message stays clean.
    with contextlib.suppress(Exception):
        fasrc_jobs.sync_pending_on_connect(STATE.ssh)
    return None


def create_app() -> Flask:
    here = os.path.dirname(os.path.abspath(__file__))
    app = Flask(
        __name__,
        template_folder=os.path.join(here, "templates"),
        static_folder=os.path.join(here, "static"),
    )

    register_mutation_guard(app)

    # EXPERIMENTAL lanes (HST / star-anchor / round-trip supervision):
    # features for the future, disabled for now. Templates read this
    # global to hide their nav links and step-card mounts — see
    # euclid_polish.web.experimental. A context processor (not a bare
    # jinja_env global) so the flag is read per-request and tests can
    # flip it.
    @app.context_processor
    def _inject_experimental_flags():
        return {"experimental_lanes": experimental.EXPERIMENTAL_LANES_ENABLED}

    # ---------------------------------------------------------------- #
    # Auto-connect to FASRC on launch. If we can't connect, every
    # non-connection-related page redirects to /connection-error where
    # the user can edit ssh_user / ssh_host and retry. We deliberately
    # store the last error message so the error page can show *why*.
    # ---------------------------------------------------------------- #
    app.config["FASRC_STARTUP_ERROR"] = _try_startup_ssh_connect()

    # Paths that remain reachable even when SSH is down — the settings
    # form, the retry endpoint, static assets, and the connection-error
    # page itself. Everything else gets gated below.
    _ALWAYS_REACHABLE_PREFIXES = (
        "/connection-error",
        "/api/fasrc/",            # connect/settings/login/etc.
        "/static/",
        "/api/status",
        "/auth/",                # laptop-side Euclid archive session; does
                                 # not depend on the FASRC SSH connection
        "/tracking",             # lab notebook is local-first; works offline
        "/api/tracking/",        # (only /api/tracking/sync needs SSH, and it
                                 #  degrades gracefully when disconnected)
        "/config",               # universal job-config tab is local-first
        "/api/config",           # (persists to ~/.euclid_polish; no SSH)
        "/api/jobs",             # the LOCAL background-job registry (spawn +
                                 # poll). Polling job progress must never be
                                 # gated on SSH — the job itself may need FASRC
                                 # and its failure is reported IN the job, but
                                 # the poll is in-process. Without this every
                                 # local job showed "job not found" when SSH was
                                 # down/flaky (the XHR got redirected to HTML).
        "/evaluation",           # results gallery reads local eval_results/
        "/api/evaluation/",      # (only .../sync needs SSH; it 400s when down)
        "/api/inspect",          # local FITS metadata for the React inspector
        "/ensemble",             # local: runs ensemble checkpoints on local test
        "/ensemble/",            # records; render + evaluate jobs need no SSH
        "/api/inference/",       # cached real-field workspace is local
        "/inference/",           # local recache/reapply jobs reuse archive data
        "/api/population-comparison",
        "/population-comparison",
        "/galaxy-distributions",
        "/api/galaxy-distributions",
        "/api/star-distribution",
        "/star-distribution",
        "/view/",                # cached, read-only presentation/diagnostic PNGs
        "/api/vis/",             # local data/vis gallery metadata
        "/eval-files/",          # serve already-pulled PNG/FITS offline
        "/viewer/",              # unified cutout viewer reads local caches
        "/jwst-euclid",
        "/api/jwst-euclid",
    )

    # The React console is now the only page UI. Keep the list explicit so
    # Flask data endpoints such as /ensemble/status.json and
    # /inspect/preview.png continue to reach their normal handlers.
    _REACT_PAGE_PATHS = frozenset({
        "/",
        "/config",
        "/catalog",
        "/psfs",
        "/sky",
        "/cutouts",
        "/tng",
        "/synthetic-real",
        "/population-comparison",
        "/galaxy-distributions",
        "/star-distribution",
        "/inference",
        "/ensemble",
        "/ensemble/starfull",
        "/ensemble/starless",
        "/train-members",
        "/evaluation",
        "/tracking",
        "/visualization",
        "/fasrc",
        "/git",
        "/inspect",
        "/jwst-euclid",
        "/connection-error",
    })

    # These pages are all rendered by the React shell even while their Flask
    # handlers remain registered as deprecated compatibility code. The HST
    # lanes are normally disabled, but including them here prevents a feature
    # flag change from silently bringing the old Jinja UI back.
    _DEPRECATED_PAGE_PATHS = frozenset({
        "/hst-psf", "/hst-cutouts", "/hst-tiles", "/hst-pairs", "/roundtrip",
        "/cutouts/VIS", "/cutouts/Y_E", "/cutouts/J_E", "/cutouts/H_E",
    })

    def _is_react_page_path(path: str) -> bool:
        normalized = path.rstrip("/") or "/"
        return normalized in _REACT_PAGE_PATHS or normalized in _DEPRECATED_PAGE_PATHS

    @app.before_request
    def _redirect_deprecated_app_prefix():
        """Move old /app bookmarks to the same route without the prefix."""
        if request.path == "/app" or request.path.startswith("/app/"):
            suffix = request.path[4:] or "/"
            query = f"?{request.query_string.decode('utf-8')}" if request.query_string else ""
            return redirect(f"{suffix}{query}", code=308)
        return None

    @app.before_request
    def _enforce_ssh_gate():
        # Quickest possible check first.
        if STATE.ssh is not None and STATE.ssh.is_connected():
            return None
        if request.path == "/connection-error":
            return None
        if _is_react_page_path(request.path):
            return None
        if any(request.path.startswith(p) for p in _ALWAYS_REACHABLE_PREFIXES):
            return None
        # API (XHR) callers can't follow an HTML redirect — the SPA would parse
        # the connection-error page as JSON and fail confusingly. Give them a
        # clean JSON 503 instead.
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": "FASRC not connected"}), 503
        # Full-page navigations: redirect to the error page so the user can
        # fix settings + retry without poking through the UI for it.
        return redirect(url_for("connection_error_page"))

    @app.route("/connection-error", methods=["GET", "POST"])
    def connection_error_page():
        """Settings-edit + retry-connect page shown when SSH is down."""
        if request.method == "POST":
            # User edited settings; persist + retry.
            patch = {
                "ssh_user":     request.form.get("ssh_user", "").strip(),
                "ssh_host":     request.form.get("ssh_host", "").strip(),
                "control_socket": request.form.get("control_socket", "").strip(),
            }
            patch = {k: v for k, v in patch.items() if v}
            if patch:
                fasrc_config.update(patch)
            err = _try_startup_ssh_connect()
            app.config["FASRC_STARTUP_ERROR"] = err
            if err is None:
                # Connected — bounce to the FASRC hub.
                return redirect(url_for("index"))
            # Fall through to re-render the page with the new error.

        cfg_loaded = fasrc_config.load()
        return render_template(
            "connection_error.html",
            error=app.config.get("FASRC_STARTUP_ERROR") or "not connected",
            cfg=cfg_loaded,
        )

    @app.route("/api/connection/retry", methods=["POST"])
    def api_connection_retry():
        """POST-only retry hook so the existing /fasrc tab can also trigger reconnect."""
        err = _try_startup_ssh_connect()
        app.config["FASRC_STARTUP_ERROR"] = err
        if err is None:
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": err}), 502

    # ---------------- Root ----------------
    @app.route("/")
    def index():
        # Keep this endpoint named ``index`` for existing url_for("index")
        # call sites. GET / is intercepted by the React shell above; this
        # fallback is only for code paths that explicitly dispatch the view.
        return redirect(url_for("fasrc_page"))

    # ---- new React console (SPA) ----
    # The redesigned UI is a Vite/React single-page app built into
    # static/dist/. The shell is served at the canonical page URLs and Flask's
    # own static handler serves /static/dist/* assets. The old Jinja handlers
    # stay registered only as compatibility code for backend route ownership;
    # page requests are intercepted below before dispatch.
    _spa_index = os.path.join(here, "static", "dist", "index.html")

    @app.before_request
    def react_console():
        if request.path in {"/cutouts/VIS", "/cutouts/Y_E", "/cutouts/J_E", "/cutouts/H_E"}:
            return redirect("/cutouts", code=308)
        if request.method not in ("GET", "HEAD") or not _is_react_page_path(request.path):
            return None
        if not os.path.isfile(_spa_index):
            return (
                "React console not built. Run:\n"
                "  cd euclid_polish/web/frontend && npm install && npm run build\n",
                503,
                {"Content-Type": "text/plain"},
            )
        return send_file(_spa_index)

    # ---- modular route groups (extracted from this file) ----
    config.register(app)
    catalog.register(app)
    auth.register(app)
    cutouts.register(app)
    hst.register(app)
    psfs.register(app)
    sky.register(app)
    tng.register(app)
    poster.register(app)
    population_comparison.register(app)
    galaxy_distributions.register(app)
    star_distribution.register(app)
    model.register(app)
    ensemble.register(app)
    evaluation.register(app)
    views.register(app)
    hstpairs.register(app)
    jwst_euclid.register(app)
    files.register(app)
    git.register(app)
    tracking.register(app)
    viewer.register(app)
    fasrc.register(app)

    return app


def main() -> None:
    """Run the Flask app on 127.0.0.1:8765."""
    ap = argparse.ArgumentParser(description="EuclidPolish localhost web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    try:
        validate_bind_host(args.host)
    except ValueError as exc:
        ap.error(str(exc))
    app = create_app()
    print(f"\nEuclidPolish web UI on http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug,
            use_reloader=False)


if __name__ == "__main__":
    main()
