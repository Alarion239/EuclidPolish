"""auth routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

import contextlib

from flask import jsonify, request

from euclid_polish.web import euclid_session
from euclid_polish.web.remote import STATE


def register(app):

    # The catalog query + photometry verify are now two separate FASRC
    # pipeline steps (``euclid_query`` on the catalog page, then download,
    # then ``euclid_verify_photometry`` on the cutouts page) submitted through
    # the standard ``/api/fasrc/hst/<step_id>/submit`` route — editable
    # resources, run history and Cancel-job all come for free. The bespoke
    # ``/catalog/query-brightest`` + ``/cutouts/verify-photometry`` routes
    # were removed.

    # ---------------- Authentication ----------------
    @app.route("/auth/login", methods=["POST"])
    def auth_login():
        user = request.form.get("username", "").strip()
        pwd  = request.form.get("password", "").strip()
        if not user or not pwd:
            return jsonify({"ok": False, "error": "Missing username or password"}), 400
        try:
            euclid_session.login(user, pwd)
            return jsonify({"ok": True, "user": euclid_session.current_user()})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/auth/logout", methods=["POST"])
    def auth_logout():
        with contextlib.suppress(Exception):
            euclid_session.logout()
        return jsonify({"ok": True})

    # ---------------- Euclid archive credentials (for FASRC download) -----
    # The cutout-download job runs on FASRC and logs into the Euclid
    # archive there. We write the credentials to the remote
    # ``~/.euclid_credentials`` (the file ``auth.login`` falls back to) via
    # the SSH channel — the password is sent as heredoc stdin (never in a
    # process argv), is stored mode-600 on the remote, and never touches
    # the laptop disk or the job DB.
    _EUCLID_CREDS_REMOTE = '"$HOME/.euclid_credentials"'

    @app.route("/euclid-auth/save", methods=["POST"])
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
        # Quoted heredoc → body is literal (no shell expansion of the
        # password); umask 077 + chmod 600 keep it private on the remote.
        write_cmd = (
            f"umask 077; cat > {_EUCLID_CREDS_REMOTE} <<'__EUCLID_CREDS_EOF__'\n"
            f"{user}\n{pw}\n"
            "__EUCLID_CREDS_EOF__\n"
            f"chmod 600 {_EUCLID_CREDS_REMOTE}"
        )
        rc, _out, err = STATE.ssh.run(write_cmd, timeout=15)
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
