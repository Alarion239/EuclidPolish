"""
Flask app factory for the EuclidPolish web UI.

Routes live in :mod:`euclid_polish.web.routes` (one module per group, each
exposing ``register(app)``, listed in ``routes.MODULES``); shared helpers live
in :mod:`euclid_polish.web.helpers`. This module just wires them together: the
request hooks (Host allowlist + mutation guard from :mod:`.security`, the SPA
shell/redirects from :mod:`.spa_routes`, the per-route FASRC gate from
:mod:`.fasrc_gate`), the startup auto-connect and the connection-retry route.
"""
from __future__ import annotations

import argparse
import contextlib
import os

# Force the non-interactive matplotlib backend BEFORE any submodule imports
# pyplot. The job registry plots from worker threads; macOS's default GUI
# backend only works on the main thread and would otherwise crash with
# "Cannot create a GUI FigureManager outside the main thread".
import matplotlib

matplotlib.use("Agg")

from flask import Flask, jsonify, redirect, request, send_file

from euclid_polish.web import fasrc_jobs
from euclid_polish.web.fasrc_gate import register_fasrc_gate
from euclid_polish.web.remote import STATE, SSHError, connect_from_config
from euclid_polish.web.routes import MODULES as ROUTE_MODULES
from euclid_polish.web.security import (
    register_host_allowlist,
    register_mutation_guard,
    validate_bind_host,
)
from euclid_polish.web.spa_routes import is_page_path, redirect_target
from euclid_polish.web.version import process_tracker


def _try_startup_ssh_connect() -> str | None:
    """Attempt one SSH connect during app startup. Return None on success.

    Stores the active session on :data:`STATE`. Failures are returned as a
    short error string and kept on ``STATE.last_error`` so
    ``GET /api/fasrc/status`` can show why FASRC is disconnected.

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
    if os.environ.get("EUCLID_POLISH_DISABLE_AUTO_SSH", "").strip() in (
        "1", "true", "yes", "on",
    ):
        STATE.last_error = ("auto-connect disabled by "
                            "EUCLID_POLISH_DISABLE_AUTO_SSH env var (test mode)")
        return STATE.last_error
    try:
        session = connect_from_config()
    except SSHError as e:
        return str(e)
    # Same catch-up as in /api/fasrc/connect — jobs that finished
    # while the server was offline get their state + sacct post-mortem
    # recorded now. Best-effort; failures here are swallowed so the
    # auto-connect message stays clean.
    with contextlib.suppress(Exception):
        fasrc_jobs.sync_pending_on_connect(session)
    return None


_SPA_NOT_BUILT = (
    "React console not built. Run:\n"
    "  cd euclid_polish/web/frontend && npm install && npm run build\n"
)


def _register_spa_shell(app: Flask, index_path: str) -> None:
    """Serve the SPA shell for manifest page paths; 308 legacy page URLs.

    Page paths and redirects come from ``spa_routes.json`` (contract C1).
    Only GET/HEAD navigations are affected: data endpoints that share a
    prefix with a page (``/ensemble/status.json``) never match, and other
    methods always reach their handlers. ``SPA_INDEX_PATH`` is read per
    request so a test (or a fresh build) can point it elsewhere.
    """
    app.config.setdefault("SPA_INDEX_PATH", index_path)

    @app.before_request
    def spa_shell():
        if request.method not in ("GET", "HEAD"):
            return None
        target = redirect_target(request.path, request.query_string)
        if target is not None:
            return redirect(target, code=308)
        if not is_page_path(request.path):
            return None
        index = app.config["SPA_INDEX_PATH"]
        if not os.path.isfile(index):
            return _SPA_NOT_BUILT, 503, {"Content-Type": "text/plain; charset=utf-8"}
        return send_file(index, mimetype="text/html")


def create_app() -> Flask:
    here = os.path.dirname(os.path.abspath(__file__))
    app = Flask(
        __name__,
        static_folder=os.path.join(here, "static"),
    )

    # Hook order matters: Host allowlist first (nothing may answer an
    # untrusted Host), then the cross-origin mutation guard, then the SPA
    # shell/redirects, then the per-route FASRC gate (below).
    register_host_allowlist(app)
    register_mutation_guard(app)
    _register_spa_shell(app, os.path.join(here, "static", "dist", "index.html"))

    # ---------------------------------------------------------------- #
    # Auto-connect to FASRC on launch. The console is offline-first: a
    # failed connect only disables the handlers marked ``@requires_fasrc``
    # (they answer 503 ``fasrc_offline``); the error is kept so
    # ``/api/fasrc/status`` can show *why* (``STATE.last_error``, the single
    # record of the startup / last connect error).
    # ---------------------------------------------------------------- #
    _try_startup_ssh_connect()
    register_fasrc_gate(app)
    # Fix the boot commit now (``GET /api/version`` compares it with HEAD).
    process_tracker()

    @app.route("/api/connection/retry", methods=["POST"])
    def api_connection_retry():
        """POST-only retry hook so the existing /fasrc tab can also trigger reconnect."""
        err = _try_startup_ssh_connect()
        if err is None:
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": err}), 502

    # ---- modular route groups (routes.MODULES; one line per group) ----
    for module in ROUTE_MODULES:
        module.register(app)

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
