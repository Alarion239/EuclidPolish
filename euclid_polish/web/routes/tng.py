"""TNG50 SKIRT-atlas download page + API-token endpoints for the web UI.

Hosts the ``download_tng_skirt`` FASRC step card (bulk-fetch the whole
IllustrisTNG TNG50-1 SKIRT atlas as dusty Euclid VIS+NISP FITS), a small
"what's on FASRC" summary derived from the per-galaxy ``.done`` markers, and a
token form that writes the IllustrisTNG API key to ``~/.tng_api_key`` on FASRC
— mirroring the Euclid-archive login. The token is sent over the SSH channel as
file content (never a process argv, never the job DB, never the laptop disk),
stored mode-600, and is the exact file the download job reads on the node.
"""
from __future__ import annotations

import os

from flask import jsonify, render_template, request, send_file

from euclid_polish.config import Config
from euclid_polish.web import fasrc_config
from euclid_polish.web.fasrc_fetcher import fetch_one_file, list_remote_dir
from euclid_polish.web.remote import STATE

# Job-rendered infographic artifacts on FASRC (written by
# scripts/fasrc_tng_infographic.py --save). Mirrors INFOGRAPHIC_SUBDIR /
# OUTPUT_NAMES in that script.
_INFOGRAPHIC_SUBDIR = "_infographics"
_INFOGRAPHIC_NAMES = {"histograms": "histograms.png", "grid": "grid.png",
                      "stack": "stack.fits"}

# Remote path of the token file, matching the script's default
# (``Config.Tng.API_KEY_FILE`` = ``~/.tng_api_key``). Quoted so a literal
# ``$HOME`` expands on the remote shell, mirroring ``_EUCLID_CREDS_REMOTE``.
_TNG_KEY_REMOTE = '"$HOME/' + os.path.basename(Config.Tng.API_KEY_FILE) + '"'


def register(app):

    @app.route("/tng")
    def tng_page():
        cfg = fasrc_config.load()
        tng_dir = f"{cfg.data_dir}/{Config.Tng.SKIRT_SUBDIR}"
        # Completed galaxies = ``.done`` sentinels one level under tng_skirt.
        # Depth 2 so ``tng_skirt/<subhalo_id>/.done`` is reached; a missing dir
        # degrades to an empty list (ok=True) rather than an error.
        ok, entries, err = list_remote_dir(
            tng_dir,
            glob_pattern=Config.Tng.DONE_MARKER,
            max_entries=2000,
            max_depth=2,
        )
        return render_template(
            "tng.html",
            tng_dir=tng_dir,
            n_done=(len(entries) if ok else None),
            list_err=(None if ok else err),
        )

    # ---------------- IllustrisTNG API token (for the FASRC job) ----------
    # The download job runs on FASRC and authenticates to the TNG API there.
    # We write the token to the remote ``~/.tng_api_key`` (the file the script
    # falls back to) via the SSH channel — the token is sent as heredoc stdin
    # (never in a process argv), stored mode-600, and never touches the laptop
    # disk or the job DB.

    @app.route("/tng-auth/save", methods=["POST"])
    def tng_auth_save():
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected to FASRC"}), 400
        token = request.form.get("tng_token", "").strip()
        if not token:
            return jsonify({"ok": False, "error": "token is required"}), 400
        if "\n" in token or "\r" in token:
            return jsonify({"ok": False, "error": "invalid characters"}), 400
        # Quoted heredoc → body is literal (no shell expansion of the token);
        # umask 077 + chmod 600 keep it private on the remote.
        write_cmd = (
            f"umask 077; cat > {_TNG_KEY_REMOTE} <<'__TNG_KEY_EOF__'\n"
            f"{token}\n"
            "__TNG_KEY_EOF__\n"
            f"chmod 600 {_TNG_KEY_REMOTE}"
        )
        rc, _out, err = STATE.ssh.run(write_cmd, timeout=15)
        if rc != 0:
            return jsonify({"ok": False,
                            "error": f"failed to write token: {err.strip()}"}), 500
        return jsonify({"ok": True, "chars": len(token)})

    @app.route("/tng-auth/status")
    def tng_auth_status():
        """Is a token file present on FASRC? Reports only presence + length —
        never the token bytes."""
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"present": False, "connected": False})
        rc, out, _err = STATE.ssh.run(
            f"test -s {_TNG_KEY_REMOTE} && "
            f"(head -1 {_TNG_KEY_REMOTE} | tr -d '\\n' | wc -c) || echo 0",
            timeout=10,
        )
        try:
            n = int((out or "0").strip().split()[0])
        except (ValueError, IndexError):
            n = 0
        return jsonify({"present": n > 0, "connected": True, "chars": n})

    # ---------------- Galaxy infographic results (job artifacts) ----------
    # The infographics are produced by the FASRC jobs ``tng_histograms`` /
    # ``tng_grid`` / ``tng_stack`` (resource-allocated step cards), which write
    # their artifact to ``tng_skirt/_infographics/<name>`` on the node. These
    # routes fetch the latest such artifact for display / download — they do
    # NOT render anything (that's the job's work).

    def _artifact_remote(kind: str) -> str:
        cfg = fasrc_config.load()
        return (f"{cfg.data_dir}/{Config.Tng.SKIRT_SUBDIR}/"
                f"{_INFOGRAPHIC_SUBDIR}/{_INFOGRAPHIC_NAMES[kind]}")

    def _serve_artifact(kind: str, mimetype: str, *, as_attachment=False,
                        download_name=None, max_bytes=None):
        # force=True so a freshly-rendered job result isn't masked by the
        # fetcher's TTL cache. The PNGs are tiny; the FITS needs the larger cap.
        kw = {"force": True}
        if max_bytes is not None:
            kw["max_bytes"] = max_bytes
        result = fetch_one_file(_artifact_remote(kind), **kw)
        if not result.ok or not result.local_path:
            hint = ("no result yet — submit the job above, then load the result "
                    "once it completes.")
            if result.error:
                hint += f" [{result.error}]"
            return jsonify({"ok": False, "error": hint}), 404
        return send_file(result.local_path, mimetype=mimetype, max_age=0,
                         as_attachment=as_attachment,
                         download_name=download_name)

    @app.route("/tng/result/histograms.png")
    def tng_result_histograms():
        return _serve_artifact("histograms", "image/png")

    @app.route("/tng/result/grid.png")
    def tng_result_grid():
        return _serve_artifact("grid", "image/png")

    @app.route("/tng/result/stack.fits")
    def tng_result_stack():
        # ~51 MB — pull with the larger cap (the default 50 MB cap is too
        # small) and hand it to the browser as a download.
        return _serve_artifact(
            "stack", "application/fits", as_attachment=True,
            download_name="TNG_stack.fits",
            max_bytes=Config.WebFetch.MAX_PSF_PULL_BYTES)
