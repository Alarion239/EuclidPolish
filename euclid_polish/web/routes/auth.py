"""Archive credentials for the web UI (System › Connections).

- ONE laptop-side Euclid archive session (:mod:`euclid_polish.web.euclid_session`):
  ``/auth/status`` / ``/auth/login`` / ``/auth/logout``. Every local feature
  that queries the archive (galaxy and star distributions, the population
  comparison, the catalog-eval galaxy query) reads this one session;
  ``/auth/status`` lists them in ``used_by`` so the UI can say where it
  matters. The password is never stored.
- The FASRC-side credentials file the cutout download reads there
  (``/euclid-auth/*``).
"""
from __future__ import annotations

import contextlib
import threading
from datetime import UTC, datetime

from flask import jsonify, request

from euclid_polish.web import euclid_session
from euclid_polish.web.fasrc_gate import requires_fasrc
from euclid_polish.web.remote import STATE

#: When the current laptop session logged in (``None`` while logged out).
_SESSION: dict[str, str | None] = {"logged_in_at": None}
_SESSION_LOCK = threading.Lock()

#: The console features that read the laptop session (``id``, label, page).
CONSUMERS = (
    {"id": "galaxies", "label": "Synthetic › Galaxies (Euclid galaxy query)", "to": "/synthetic/galaxies"},
    {"id": "stars", "label": "Synthetic › Stars (Euclid star query)", "to": "/synthetic/stars"},
    {"id": "pixels", "label": "Synthetic › Fields (population comparison)",
     "to": "/synthetic/fields?view=stats"},
    {"id": "catalog-eval", "label": "Sky › Targets (query galaxies)", "to": "/sky/targets?set=galaxies"},
)


def session_status() -> dict:
    """The one laptop-side archive session (``/auth/status``)."""
    authenticated = euclid_session.is_authenticated()
    with _SESSION_LOCK:
        logged_in_at = _SESSION["logged_in_at"] if authenticated else None
    return {
        "authenticated": authenticated,
        "user": euclid_session.current_user(),
        "logged_in_at": logged_in_at,
        "used_by": [dict(item) for item in CONSUMERS],
    }


def register(app):

    # The catalog query + photometry verify are now two separate FASRC
    # pipeline steps (``euclid_query``, then download, then
    # ``euclid_verify_photometry``; both on Synthetic › PSF) submitted through
    # the standard ``/api/fasrc/steps/<step_id>/submit`` route — editable
    # resources, run history and Cancel-job all come for free. The bespoke
    # ``/catalog/query-brightest`` + ``/cutouts/verify-photometry`` routes
    # were removed.

    # ---------------- Authentication ----------------
    @app.route("/auth/status")
    def auth_status():
        return jsonify(session_status())

    @app.route("/auth/login", methods=["POST"])
    def auth_login():
        user = request.form.get("username", "").strip()
        # Passwords are opaque: trimming them can turn a valid archive
        # credential into a different password.
        pwd = request.form.get("password", "")
        if not user or not pwd:
            return jsonify({"ok": False, "error": "Missing username or password"}), 400
        try:
            euclid_session.login(user, pwd)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        with _SESSION_LOCK:
            _SESSION["logged_in_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        # The reply keeps its legacy shape; /auth/status has the full session.
        return jsonify({"ok": True, "user": euclid_session.current_user()})

    @app.route("/auth/logout", methods=["POST"])
    def auth_logout():
        with contextlib.suppress(Exception):
            euclid_session.logout()
        with _SESSION_LOCK:
            _SESSION["logged_in_at"] = None
        return jsonify({"ok": True})

    # ---------------- Euclid archive credentials (for FASRC download) -----
    # The cutout-download job runs on FASRC and logs into the Euclid
    # archive there. We write the credentials to the remote
    # ``~/.euclid_credentials`` (``scripts/download_all_bands.py`` bridges it
    # into ``EUCLID_USER``/``EUCLID_PASSWORD`` for ``EuclidCatalog``) via
    # the SSH channel — the password streams over the channel's stdin
    # (``SSHSession.write_text``; never in the local ssh argv or the remote
    # command line), is stored mode-600 on the remote, and never touches
    # the laptop disk or the job DB.
    _EUCLID_CREDS_PATH = "~/.euclid_credentials"
    # The same file as a remote shell word, for the status probe.
    _EUCLID_CREDS_REMOTE = '"$HOME/.euclid_credentials"'

    @app.route("/euclid-auth/save", methods=["POST"])
    @requires_fasrc
    def euclid_auth_save():
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected to FASRC"}), 400
        user = request.form.get("euclid_user", "").strip()
        pw   = request.form.get("euclid_password", "")
        if not user or not pw:
            return jsonify({"ok": False,
                            "error": "username and password are required"}), 400
        if "\n" in user or "\n" in pw:
            return jsonify({"ok": False, "error": "invalid characters"}), 400
        # Line 1 user, line 2 password; stdin carries both verbatim (no shell
        # sees them), and private=True keeps the file owner-only.
        rc, _out, err = STATE.ssh.write_text(
            _EUCLID_CREDS_PATH, f"{user}\n{pw}\n", private=True, timeout=15)
        if rc != 0:
            return jsonify({"ok": False,
                            "error": f"failed to write credentials: {err.strip()}"}), 500
        return jsonify({"ok": True, "user": user})

    @app.route("/euclid-auth/status")
    def euclid_auth_status():
        """Is a credentials file present on FASRC? Returns the username
        (line 1) for display — never the password."""
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"present": False, "connected": False})
        rc, out, _err = STATE.ssh.run(
            f"test -e {_EUCLID_CREDS_REMOTE} && head -1 {_EUCLID_CREDS_REMOTE} || true",
            timeout=10,
        )
        lines = [ln for ln in out.splitlines() if ln.strip()]
        user = lines[0].strip() if lines else ""
        return jsonify({"present": bool(user), "connected": True,
                        "user": user or None})
