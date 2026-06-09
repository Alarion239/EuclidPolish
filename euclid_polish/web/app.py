"""
Flask app factory for the EuclidPolish web UI.

Routes live in :mod:`euclid_polish.web.routes` (one module per group, each
exposing ``register(app)``); shared helpers live in
:mod:`euclid_polish.web.helpers`. This module just wires them together and
owns the FASRC SSH gate + the root redirect.
"""
from __future__ import annotations

# Force the non-interactive matplotlib backend BEFORE any submodule imports
# pyplot. The job registry plots from worker threads; macOS's default GUI
# backend only works on the main thread and would otherwise crash with
# "Cannot create a GUI FigureManager outside the main thread".
import matplotlib
matplotlib.use("Agg")

import argparse
import os
import os as _os
import time
from typing import Optional

from flask import (
    Flask, jsonify, redirect, render_template, request, url_for,
)

from euclid_polish.web import fasrc_config, fasrc_jobs
from euclid_polish.web.remote import SSHConfig, SSHError, SSHSession, STATE
from euclid_polish.web.routes import (
    auth, catalog, config, cutouts, fasrc, files, git, hst, hstpairs, model,
    poster, psfs, sky, tng, tracking, views,
)


def _try_startup_ssh_connect() -> Optional[str]:
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
    try:
        fasrc_jobs.sync_pending_on_connect(STATE.ssh)
    except Exception:
        pass
    return None


def create_app() -> Flask:
    here = os.path.dirname(os.path.abspath(__file__))
    app = Flask(
        __name__,
        template_folder=os.path.join(here, "templates"),
        static_folder=os.path.join(here, "static"),
    )

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
        "/tracking",             # lab notebook is local-first; works offline
        "/api/tracking/",        # (only /api/tracking/sync needs SSH, and it
                                 #  degrades gracefully when disconnected)
        "/config",               # universal job-config tab is local-first
        "/api/config",           # (persists to ~/.euclid_polish; no SSH)
    )

    @app.before_request
    def _enforce_ssh_gate():
        # Quickest possible check first.
        if STATE.ssh is not None and STATE.ssh.is_connected():
            return None
        if request.path == "/connection-error":
            return None
        if any(request.path.startswith(p) for p in _ALWAYS_REACHABLE_PREFIXES):
            return None
        # All other paths: redirect to the error page so the user can
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
        # The status dashboard was removed; the FASRC tab is the hub. Keep
        # this endpoint named ``index`` so existing ``url_for("index")``
        # call sites (e.g. the connection-error bounce) still resolve, and
        # the root URL lands somewhere useful instead of 404ing.
        return redirect(url_for("fasrc_page"))

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
    model.register(app)
    views.register(app)
    hstpairs.register(app)
    files.register(app)
    git.register(app)
    tracking.register(app)
    fasrc.register(app)

    return app


def main() -> None:
    """Run the Flask app on 127.0.0.1:8765."""
    ap = argparse.ArgumentParser(description="EuclidPolish localhost web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    app = create_app()
    print(f"\nEuclidPolish web UI on http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=args.debug,
            use_reloader=False)


if __name__ == "__main__":
    main()

