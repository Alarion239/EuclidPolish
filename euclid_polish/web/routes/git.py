"""git routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

from euclid_polish.web import git_ops
from flask import jsonify
from flask import render_template
from flask import request


def register(app):

    # =========================================================================
    # Git tab — local commit / push / pull, no remote auth needed.
    # =========================================================================

    @app.route("/git")
    def git_page():
        return render_template(
            "git.html",
            status=git_ops.status(),
            log_entries=git_ops.log(15),
        )

    @app.route("/api/git/status")
    def api_git_status():
        return jsonify({"status": git_ops.status(),
                        "log": git_ops.log(15)})

    @app.route("/api/git/diff")
    def api_git_diff():
        staged = request.args.get("staged", "0") in ("1", "true", "yes")
        return jsonify({"diff": git_ops.diff(staged=staged)})

    @app.route("/git/commit", methods=["POST"])
    def git_commit():
        msg = request.form.get("message", "").strip()
        out = git_ops.commit(msg)
        code = 200 if out.get("ok") else 400
        return jsonify(out), code

    @app.route("/git/push", methods=["POST"])
    def git_push():
        out = git_ops.push()
        code = 200 if out.get("ok") else 400
        return jsonify(out), code

    @app.route("/git/pull", methods=["POST"])
    def git_pull():
        out = git_ops.pull()
        code = 200 if out.get("ok") else 400
        return jsonify(out), code

    @app.route("/git/fetch", methods=["POST"])
    def git_fetch():
        out = git_ops.fetch()
        code = 200 if out.get("ok") else 400
        return jsonify(out), code
