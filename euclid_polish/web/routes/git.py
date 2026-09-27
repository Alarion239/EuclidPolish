"""git routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

from flask import jsonify, request

from euclid_polish.web import errors, git_ops


def _flag(name: str) -> bool:
    return str(request.form.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


def _paths() -> list[str]:
    """``paths`` (repeatable; one value may hold several newline-separated
    paths), taken literally as ``/api/git/status`` lists them."""
    return [part.strip()
            for raw in request.form.getlist("paths")
            for part in raw.splitlines() if part.strip()]


def register(app):

    # =========================================================================
    # Git tab — local commit / push / pull, no remote auth needed.
    # =========================================================================

    @app.route("/api/git/status")
    def api_git_status():
        return jsonify({"status": git_ops.status(),
                        "log": git_ops.log(15)})

    @app.route("/api/git/diff")
    def api_git_diff():
        """``?staged=1`` for the index; ``?path=`` for one file/directory (an
        untracked file's unstaged diff is its content)."""
        staged = request.args.get("staged", "0") in ("1", "true", "yes")
        path = request.args.get("path")
        return jsonify({"diff": git_ops.diff(staged=staged, path=path),
                        "staged": staged, "path": path})

    @app.route("/api/git/log")
    def api_git_log():
        """One page of the history: ``?skip=&limit=`` (1–500, default 50)."""
        return jsonify(git_ops.log_page(skip=errors.int_arg("skip", 0),
                                        limit=errors.int_arg("limit", 50)))

    @app.route("/api/git/commit/<rev>")
    def api_git_show(rev: str):
        out = git_ops.show(rev)
        return jsonify(out), (200 if out.get("ok") else 400)

    @app.route("/git/stage", methods=["POST"])
    def git_stage():
        """Stage exactly the changed files covered by ``paths``."""
        out = git_ops.stage(_paths())
        return jsonify(out), (200 if out.get("ok") else 400)

    @app.route("/git/unstage", methods=["POST"])
    def git_unstage():
        """Take ``paths`` out of the index; working-tree edits are kept."""
        out = git_ops.unstage(_paths())
        return jsonify(out), (200 if out.get("ok") else 400)

    @app.route("/git/commit", methods=["POST"])
    def git_commit():
        """Commit the chosen files: ``paths`` (repeatable; one value may hold
        several newline-separated paths) or ``all=1``; ``force=1`` overrides
        the size guard. Paths are taken literally — as ``/api/git/status``
        lists them (commas, spaces and non-ASCII are part of a name).
        400 without a selection, 409 ``refused_files`` with the list."""
        msg = request.form.get("message", "").strip()
        paths = _paths()
        out = git_ops.commit(msg, paths or None, all_files=_flag("all"),
                             force=_flag("force"))
        if out.get("ok"):
            return jsonify(out), 200
        return jsonify(out), (409 if out.get("code") == "refused_files" else 400)

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
