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
import threading
import time

from flask import jsonify, request, send_file

from euclid_polish.config import Config
from euclid_polish.tng.properties import render_histograms_for_ids
from euclid_polish.web import fasrc_config, fasrc_jobs
from euclid_polish.web.fasrc_fetcher import fetch_one_file
from euclid_polish.web.fasrc_gate import requires_fasrc
from euclid_polish.web.jobs import REGISTRY as JOB_REGISTRY
from euclid_polish.web.remote import STATE

# Job-rendered image infographics on FASRC (written by
# scripts/fasrc_tng_infographic.py --save, grid/stack only). The histogram is
# rendered LOCALLY (see /tng/histograms.png) — it needs no FITS, just the id
# list pulled from FASRC + the TNG API.
_INFOGRAPHIC_SUBDIR = "_infographics"
_INFOGRAPHIC_NAMES = {"grid": "grid.png", "stack": "stack.fits"}
# Radius/property calibration artifacts live beside the other population
# caches, not inside the display-only infographic directory.
_CALIBRATION_SUBDIR = "_tng_infographics"
# Local working dir for the histogram's property cache (CSV + groupcat arrays).
_LOCAL_TNG_DIR = os.path.join(Config.DATA_DIR, _CALIBRATION_SUBDIR)

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


#: A cached radius validation older than this is reported ``stale`` by the
#: status GET (the client then asks for ``POST /api/tng/radii/refresh``).
_RADII_TTL_S = 3600.0
#: A cached *failure* (often a transient SSH timeout) goes stale much sooner,
#: so a retry is offered after minutes rather than an hour.
_RADII_FAILED_TTL_S = 300.0
_RADII_JOB_KIND = "tng-radii"


def _radii_cache_path() -> str:
    return os.path.join(Config.DATA_DIR, _CALIBRATION_SUBDIR,
                        "tng_radius_manifest_status.json")


def _read_radii_cache() -> dict | None:
    try:
        with open(_radii_cache_path()) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _validate_radius_manifest() -> dict:
    """Run the remote validator once; the payload the status route serves."""
    cfg = fasrc_config.load()
    tng_dir = os.path.join(cfg.data_dir, Config.Tng.SKIRT_SUBDIR)
    props = os.path.join(cfg.data_dir, _CALIBRATION_SUBDIR, "tng_properties.csv")
    manifest = os.path.join(cfg.data_dir, _CALIBRATION_SUBDIR,
                            "tng_radius_manifest.json")
    rc, out, err = fasrc_jobs.run_remote_python(
        STATE.ssh,
        cfg=cfg,
        argv=[
            "scripts/validate_tng_radius_manifest.py",
            "--tng-dir", tng_dir,
            "--properties", props,
            "--manifest", manifest,
        ],
        timeout=180,
    )
    lines = [line for line in (out or "").splitlines() if line.strip()]
    if not lines:
        raise ValueError(
            (err or "radius-manifest validator returned no output").strip())
    payload = json.loads(lines[-1])
    if rc != 0 and not payload.get("reasons"):
        payload["reasons"] = [(err or "manifest validation failed").strip()]
    return payload


def _write_radii_cache(payload: dict) -> None:
    path = _radii_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle)
    os.replace(tmp, path)


def _radii_cache_stale(cached: dict | None) -> bool:
    """Missing, or older than its TTL (a failure's TTL is the short one)."""
    if cached is None:
        return True
    ttl = _RADII_FAILED_TTL_S if cached.get("failed") else _RADII_TTL_S
    try:
        checked_at = float(cached.get("checked_at", 0))
    except (TypeError, ValueError):
        return True
    return time.time() - checked_at > ttl


def _radii_refresh_job(cap) -> dict:
    """Validate once and cache the answer — a failure is cached too (with
    ``checked_at`` and ``failed``, stale after the short failure TTL)."""
    cap.tick(0, 1, "validating the remote TNG radius manifest")
    try:
        payload = {**_validate_radius_manifest(), "checked_at": time.time()}
    except Exception as exc:
        _write_radii_cache({"valid": False, "reasons": [str(exc)],
                            "failed": True, "checked_at": time.time()})
        raise
    _write_radii_cache(payload)
    cap.tick(1, 1, "validated")
    return payload


_RADII_SPAWN_LOCK = threading.Lock()


def _radii_refresh_running() -> str | None:
    for job in JOB_REGISTRY.list(summary=True):
        if job.get("kind") == _RADII_JOB_KIND and job.get("status") == "running":
            return str(job["job_id"])
    return None


def _spawn_radii_refresh() -> str:
    """Start the validation job unless one already runs (one at a time)."""
    with _RADII_SPAWN_LOCK:
        running = _radii_refresh_running()
        if running is not None:
            return running
        return JOB_REGISTRY.spawn("TNG: validate radius manifest",
                                  _radii_refresh_job, kind=_RADII_JOB_KIND)


def register(app):

    @app.route("/api/tng/radii/status")
    def tng_radii_status():
        """The last radius-manifest validation, answered immediately.

        Read-only: validation runs the remote checker (up to 180 s over
        SSH) in a job started by ``POST /api/tng/radii/refresh``. This GET
        serves the cached result plus ``stale`` (cache missing, older than
        :data:`_RADII_TTL_S`, or a failure older than
        :data:`_RADII_FAILED_TTL_S`) and ``refresh_job`` (the running
        validation job, if any); a client refreshes when ``stale`` and
        ``connected`` and no job runs. Works offline.
        """
        cached = _read_radii_cache()
        connected = bool(STATE.ssh and STATE.ssh.is_connected())
        base = cached or {"valid": False, "reasons": [
            "not validated yet" + ("" if connected else " — connect to FASRC")]}
        return jsonify({**base, "cached": cached is not None,
                        "stale": _radii_cache_stale(cached),
                        "connected": connected,
                        "refresh_job": _radii_refresh_running()})

    @app.post("/api/tng/radii/refresh")
    @requires_fasrc
    def tng_radii_refresh():
        """Re-validate the remote radius manifest in a local job."""
        return jsonify({"ok": True, "job_id": _spawn_radii_refresh()})

    # ---------------- IllustrisTNG API token (for the FASRC job) ----------
    # The download job runs on FASRC and authenticates to the TNG API there.
    # We write the token to the remote ``~/.tng_api_key`` (the file the script
    # falls back to) via the SSH channel — the token is sent as heredoc stdin
    # (never in a process argv), stored mode-600, and never touches the laptop
    # disk or the job DB.

    @app.route("/tng-auth/save", methods=["POST"])
    @requires_fasrc
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

    def _serve_artifact(kind: str, mimetype: str, *, as_attachment: bool = False,
                        download_name: str | None = None,
                        max_bytes: int | None = None):
        # force=True so a freshly-rendered job result isn't masked by the
        # fetcher's TTL cache. The PNGs are tiny; the FITS needs the larger cap.
        if max_bytes is None:
            result = fetch_one_file(_artifact_remote(kind), force=True)
        else:
            result = fetch_one_file(
                _artifact_remote(kind), force=True, max_bytes=max_bytes,
            )
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
    @requires_fasrc
    def tng_result_grid():
        return _serve_artifact("grid", "image/png")

    @app.route("/tng/result/stack.fits")
    @requires_fasrc
    def tng_result_stack():
        # ~51 MB — pull with the larger cap (the default 50 MB cap is too
        # small) and hand it to the browser as a download.
        return _serve_artifact(
            "stack", "application/fits", as_attachment=True,
            download_name="TNG_stack.fits",
            max_bytes=Config.WebFetch.MAX_PSF_PULL_BYTES)
