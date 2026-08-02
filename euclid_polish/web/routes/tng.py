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

import glob
import io
import json
import os
import shlex
import time

from flask import jsonify, render_template, request, send_file

from euclid_polish.config import Config
from euclid_polish.tng.properties import render_histograms_for_ids
from euclid_polish.web import fasrc_config
from euclid_polish.web.fasrc_fetcher import fetch_one_file, list_remote_dir
from euclid_polish.web.remote import STATE

# Job-rendered image infographics on FASRC (written by
# scripts/fasrc_tng_infographic.py --save, grid/stack only). The histogram is
# rendered LOCALLY (see /tng/histograms.png) — it needs no FITS, just the id
# list pulled from FASRC + the TNG API.
_INFOGRAPHIC_SUBDIR = "_infographics"
_INFOGRAPHIC_NAMES = {"grid": "grid.png", "stack": "stack.fits"}
# Local working dir for the histogram's property cache (CSV + groupcat arrays).
_LOCAL_TNG_DIR = os.path.join(Config.DATA_DIR, "_tng_infographics")

# TNG Atlas images (the FASRC-pulled grid + the histogram drawn from the
# FASRC-pulled id list) are also archived under here so they appear in the
# Visualization gallery (data/vis/) and get the 📌-track button. ``os.walk``
# in ``_list_vis_pngs`` recurses, so a ``tng/`` subdir shows up automatically.
_VIS_TNG_DIR = os.path.join(Config.VIS_DIR, "tng")


def _archive_png_to_vis(kind: str, png_bytes: bytes) -> None:
    """Best-effort copy of a TNG Atlas image into ``data/vis/tng/``.

    Saves a timestamped ``tng_<kind>_<YYYYmmdd-HHMMSS>.png`` so every
    distinct pulled image is kept. Skips the write when the most recent
    archived image of this kind is byte-identical, so reloading the same
    result doesn't pile up duplicates. Never raises — archiving must not
    break serving the image to the page.
    """
    try:
        os.makedirs(_VIS_TNG_DIR, exist_ok=True)
        prior = sorted(glob.glob(os.path.join(_VIS_TNG_DIR, f"tng_{kind}_*.png")))
        if prior:
            try:
                with open(prior[-1], "rb") as fh:
                    if fh.read() == png_bytes:
                        return  # unchanged since last pull — already archived
            except OSError:
                pass
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(_VIS_TNG_DIR, f"tng_{kind}_{ts}.png")
        if os.path.exists(path):  # >1 save in the same second
            path = os.path.join(
                _VIS_TNG_DIR, f"tng_{kind}_{ts}_{int(time.time() * 1000) % 1000:03d}.png")
        with open(path, "wb") as fh:
            fh.write(png_bytes)
    except Exception:
        pass

# Remote path of the token file, matching the script's default
# (``Config.Tng.API_KEY_FILE`` = ``~/.tng_api_key``). Quoted so a literal
# ``$HOME`` expands on the remote shell, mirroring ``_EUCLID_CREDS_REMOTE``.
_TNG_KEY_REMOTE = '"$HOME/' + os.path.basename(Config.Tng.API_KEY_FILE) + '"'


def register(app):

    @app.route("/api/tng/radii/status")
    def tng_radii_status():
        """Validate the remote, versioned TNG effective-radius manifest."""
        if not STATE.ssh or not STATE.ssh.is_connected():
            return jsonify({"valid": False, "connected": False,
                            "reasons": ["not connected to FASRC"]})
        cfg = fasrc_config.load()
        tng_dir = os.path.join(cfg.data_dir, Config.Tng.SKIRT_SUBDIR)
        props = os.path.join(cfg.data_dir, _INFOGRAPHIC_SUBDIR,
                             "tng_properties.csv")
        manifest = os.path.join(cfg.data_dir, _INFOGRAPHIC_SUBDIR,
                                "tng_radius_manifest.json")
        cmd = (
            f"cd {shlex.quote(cfg.repo_path)} && "
            "python scripts/validate_tng_radius_manifest.py "
            f"--tng-dir {shlex.quote(tng_dir)} "
            f"--properties {shlex.quote(props)} "
            f"--manifest {shlex.quote(manifest)}"
        )
        try:
            rc, out, err = STATE.ssh.run(cmd, timeout=90)
            payload = json.loads((out or "").strip().splitlines()[-1])
        except Exception as exc:
            return jsonify({"valid": False, "connected": True,
                            "reasons": [str(exc)]})
        payload["connected"] = True
        if rc != 0 and not payload.get("reasons"):
            payload["reasons"] = [(err or "manifest validation failed").strip()]
        return jsonify(payload)

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

    # ---------------- Histogram (rendered LOCALLY) ------------------------
    # The histogram needs no FITS — only the downloaded-galaxy id list (a tiny
    # SSH `find`) plus the TNG API. So we pull just the ids from FASRC and
    # render the plot in this process; nothing heavy is transferred.

    def _downloaded_ids() -> list:
        """Subhalo ids of finished galaxies on FASRC (dirs holding a .done)."""
        if not STATE.ssh or not STATE.ssh.is_connected():
            return []
        cfg = fasrc_config.load()
        tng_dir = f"{cfg.data_dir}/{Config.Tng.SKIRT_SUBDIR}"
        cmd = (f"find {shlex.quote(tng_dir)} -mindepth 2 -maxdepth 2 "
               f"-name {shlex.quote(Config.Tng.DONE_MARKER)} -printf '%h\\n' "
               "2>/dev/null | head -n 20000")
        try:
            rc, out, _err = STATE.ssh.run(cmd, timeout=20)
        except Exception:
            return []
        if rc != 0 or not out:
            return []
        ids = {os.path.basename(ln.strip()) for ln in out.splitlines()
               if ln.strip()}
        try:
            return sorted(ids, key=int)
        except ValueError:
            return sorted(ids)

    def _tng_api_key() -> str:
        """TNG token for the local API call: local $TNG_API_KEY, else read the
        key saved on FASRC (the token form's file) over SSH. In-memory only."""
        env = os.environ.get("TNG_API_KEY", "").strip()
        if env:
            return env
        if STATE.ssh and STATE.ssh.is_connected():
            try:
                rc, out, _err = STATE.ssh.run(
                    f"cat {_TNG_KEY_REMOTE} 2>/dev/null", timeout=10)
                if rc == 0 and out:
                    return out.splitlines()[0].strip()
            except Exception:
                pass
        return ""

    @app.route("/tng/histograms.png")
    def tng_histograms_png():
        os.makedirs(_LOCAL_TNG_DIR, exist_ok=True)
        ids = _downloaded_ids()
        png = render_histograms_for_ids(_LOCAL_TNG_DIR, ids, _tng_api_key())
        # Archive to the Visualization gallery — but only when there are real
        # galaxies behind it, so we don't save empty-state placeholders.
        if ids:
            _archive_png_to_vis("histograms", png)
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=0)

    # ---------------- Image infographic results (grid/stack job artifacts) -
    # The grid + stacked-FITS jobs (``tng_grid`` / ``tng_stack`` step cards)
    # write their artifact to ``tng_skirt/_infographics/<name>`` on the node;
    # these routes fetch the latest one for display / download.

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
        # Archive the FASRC-pulled grid image into the Visualization gallery.
        # The stack is a FITS download (not a gallery image), so it's skipped.
        if kind == "grid":
            try:
                with open(result.local_path, "rb") as fh:
                    _archive_png_to_vis("grid", fh.read())
            except OSError:
                pass
        return send_file(result.local_path, mimetype=mimetype, max_age=0,
                         as_attachment=as_attachment,
                         download_name=download_name)

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
