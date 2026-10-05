"""TNG routes of Synthetic › Galaxies (templates): the API token on FASRC, the radius-manifest
validation, the property explorer and the grid/stack job results.

* The token form writes the IllustrisTNG API key to ``~/.tng_api_key`` on
  FASRC — sent over the SSH channel as file content (never a process argv,
  never the job DB, never the laptop disk), stored mode-600, the exact file
  the download job reads on the node.
* ``GET /api/tng/properties`` is the interactive property explorer over the
  local calibration CSVs (``helpers/tng_explorer.py``); ``POST
  /api/tng/properties/refresh`` re-queries missing galaxies from the TNG API
  in a job (the only thing that writes that cache).
* The ``tng_grid`` / ``tng_stack`` job artifacts are pulled from FASRC by an
  explicit ``POST /api/tng/result/pull`` job; ``GET /tng/result/grid.png``
  and ``/tng/result/stack.fits`` serve the last pulled copy (cache-only: a
  GET never writes into ``data/``).
"""
from __future__ import annotations

import glob
import json
import os
import shlex
import threading
import time

from flask import jsonify, request, send_file

from euclid_polish.config import Config
from euclid_polish.tng.properties import gather_properties
from euclid_polish.web import fasrc_config, fasrc_jobs
from euclid_polish.web.fasrc_fetcher import _local_path_for, fetch_one_file
from euclid_polish.web.fasrc_gate import requires_fasrc
from euclid_polish.web.helpers import tng_explorer
from euclid_polish.web.jobs import REGISTRY as JOB_REGISTRY
from euclid_polish.web.remote import STATE

# Job-rendered image infographics on FASRC (written by
# scripts/fasrc_tng_infographic.py --save, grid/stack only).
_INFOGRAPHIC_SUBDIR = "_infographics"
_INFOGRAPHIC_NAMES = {"grid": "grid.png", "stack": "stack.fits"}
# Radius/property calibration artifacts live beside the other population
# caches, not inside the display-only infographic directory.
_CALIBRATION_SUBDIR = "_tng_infographics"

# Pulled grid images are also archived under here so they appear in the
# Figures › Plates PNG gallery (data/vis/). ``os.walk``
# in ``_list_vis_pngs`` recurses, so a ``tng/`` subdir shows up automatically.
# Written by the pull JOB only (never by a GET); resolved per call so the
# configured VIS_DIR (tests redirect it) is honoured.
def _vis_tng_dir() -> str:
    return os.path.join(Config.VIS_DIR, "tng")


_RESULT_JOB_KIND = "tng-result"
_PROPERTIES_JOB_KIND = "tng-properties"


def _archive_png_to_vis(kind: str, png_bytes: bytes) -> None:
    """Best-effort copy of a TNG Atlas image into ``data/vis/tng/``.

    Saves a timestamped ``tng_<kind>_<YYYYmmdd-HHMMSS>.png`` so every
    distinct pulled image is kept. Skips the write when the most recent
    archived image of this kind is byte-identical, so re-pulling the same
    result doesn't pile up duplicates. Never raises.
    """
    vis_tng_dir = _vis_tng_dir()
    try:
        os.makedirs(vis_tng_dir, exist_ok=True)
        prior = sorted(glob.glob(os.path.join(vis_tng_dir, f"tng_{kind}_*.png")))
        if prior:
            try:
                with open(prior[-1], "rb") as fh:
                    if fh.read() == png_bytes:
                        return  # unchanged since last pull — already archived
            except OSError:
                pass
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(vis_tng_dir, f"tng_{kind}_{ts}.png")
        if os.path.exists(path):  # >1 save in the same second
            path = os.path.join(
                vis_tng_dir, f"tng_{kind}_{ts}_{int(time.time() * 1000) % 1000:03d}.png")
        with open(path, "wb") as fh:
            fh.write(png_bytes)
    except Exception:
        pass


# Remote path of the token file, matching the script's default
# (``Config.Tng.API_KEY_FILE`` = ``~/.tng_api_key``): ``_TNG_KEY_PATH`` for
# ``SSHSession.write_text`` (which resolves the ``~/``), ``_TNG_KEY_REMOTE``
# as a shell word whose ``$HOME`` expands on the remote shell, mirroring
# ``_EUCLID_CREDS_REMOTE``.
_TNG_KEY_PATH = "~/" + os.path.basename(Config.Tng.API_KEY_FILE)
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


def _running(kind: str) -> str | None:
    for job in JOB_REGISTRY.list(summary=True):
        if job.get("kind") == kind and job.get("status") == "running":
            return str(job["job_id"])
    return None


def _radii_refresh_running() -> str | None:
    return _running(_RADII_JOB_KIND)


def _artifact_remote(kind: str) -> str:
    cfg = fasrc_config.load()
    return (f"{cfg.data_dir}/{Config.Tng.SKIRT_SUBDIR}/"
            f"{_INFOGRAPHIC_SUBDIR}/{_INFOGRAPHIC_NAMES[kind]}")


def _artifact_local(kind: str) -> str:
    """Where the last pull of ``kind`` sits (the fetcher's cache path)."""
    return _local_path_for(_artifact_remote(kind))


def _artifact_status(kind: str) -> dict:
    path = _artifact_local(kind)
    try:
        st = os.stat(path)
    except OSError:
        return {"present": False, "pulled_at": None, "size_bytes": None}
    return {"present": True, "pulled_at": float(st.st_mtime), "size_bytes": int(st.st_size)}


def _job_pull_results(cap, kinds: list[str]) -> dict:
    """Pull the latest grid / stack artifacts from FASRC (force: a fresh job
    result is never masked by the fetcher's TTL); archive a new grid image
    into the Figures › Plates PNG gallery (``data/vis/tng/``)."""
    out: dict[str, dict] = {}
    for position, kind in enumerate(kinds):
        cap.tick(position, len(kinds), f"pulling the {kind}")
        kwargs = {"max_bytes": Config.WebFetch.MAX_PSF_PULL_BYTES} if kind == "stack" else {}
        result = fetch_one_file(_artifact_remote(kind), force=True, **kwargs)
        if not result.ok or not result.local_path:
            out[kind] = {"ok": False, "error": result.error or "not on FASRC — run the job first"}
            cap.write(f"{kind}: {out[kind]['error']}\n")
            continue
        out[kind] = {"ok": True, "size_bytes": result.size_bytes}
        cap.write(f"{kind}: {(result.size_bytes or 0) / 1e6:.1f} MB\n")
        if kind == "grid":
            try:
                with open(result.local_path, "rb") as fh:
                    _archive_png_to_vis("grid", fh.read())
            except OSError:
                pass
    cap.tick(len(kinds), len(kinds), "done")
    if not any(entry["ok"] for entry in out.values()):
        raise RuntimeError("; ".join(f"{k}: {v['error']}" for k, v in out.items()))
    return out


class _CapReporter:
    """``tng.properties`` progress reporter → a job's log/progress."""

    def __init__(self, cap) -> None:
        self.cap = cap

    def set_stage(self, text: str) -> None:
        self.cap.write(text + "\n")

    def set_step(self, done: int, total: int, label: str) -> None:
        self.cap.tick(done, total, label)


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
    # falls back to) via the SSH channel — the token streams over the
    # channel's stdin (``SSHSession.write_text``; never in the local ssh argv
    # or the remote command line), is stored mode-600, and never touches the
    # laptop disk or the job DB.

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
        # stdin carries the token verbatim (no shell sees it), and
        # private=True keeps the file owner-only.
        rc, _out, err = STATE.ssh.write_text(
            _TNG_KEY_PATH, f"{token}\n", private=True, timeout=15)
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

    # ---------------- Property explorer (local CSVs) ---------------------

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

    @app.route("/api/tng/properties")
    def tng_properties():
        """The property explorer's rows (local CSVs only; see
        ``helpers/tng_explorer``): per galaxy SFR, stellar/halo mass, the
        group-catalogue radius and the measured VIS R_e per viewpoint."""
        return jsonify(tng_explorer.properties_payload())

    @app.post("/api/tng/properties/refresh")
    @requires_fasrc
    def tng_properties_refresh():
        """Re-query the TNG API for downloaded galaxies missing from the
        property cache (a job, ``kind="tng-properties"``): ids from FASRC,
        the token from ``$TNG_API_KEY`` or the FASRC token file. Writes
        ``tng_properties.csv``; result ``{n_ids, n_resolved}``."""
        running = _running(_PROPERTIES_JOB_KIND)
        if running:
            return jsonify({"ok": True, "job_id": running, "already_running": True})

        def _job(cap):
            cap.tick(0, 1, "listing the downloaded galaxies on FASRC")
            ids = _downloaded_ids()
            key = _tng_api_key()
            if not ids:
                raise RuntimeError("no downloaded galaxies found on FASRC")
            if not key:
                raise RuntimeError("no TNG API token (System › Connections)")
            work = tng_explorer.calibration_dir()
            os.makedirs(work, exist_ok=True)
            props = gather_properties(work, ids, key, reporter=_CapReporter(cap))
            return {"n_ids": len(ids), "n_resolved": len(props)}

        return jsonify({"ok": True, "job_id": JOB_REGISTRY.spawn(
            "TNG: refresh galaxy properties", _job, kind=_PROPERTIES_JOB_KIND)})

    # ---------------- Grid / stack job results -----------------------------
    # The grid + stacked-FITS jobs (``tng_grid`` / ``tng_stack`` step cards)
    # write their artifact to ``tng_skirt/_infographics/<name>`` on the node.

    @app.route("/api/tng/results")
    def tng_results_status():
        """Which job results were pulled, and when (local): ``{grid:
        {present, pulled_at, size_bytes}, stack: {…}, pull_job}``."""
        return jsonify({kind: _artifact_status(kind) for kind in _INFOGRAPHIC_NAMES}
                       | {"pull_job": _running(_RESULT_JOB_KIND)})

    @app.post("/api/tng/result/pull")
    @requires_fasrc
    def tng_result_pull():
        """Pull the latest ``grid`` / ``stack`` (``kind=grid|stack|all``,
        default all) from FASRC in a job (``kind="tng-result"``)."""
        raw = str(request.values.get("kind", "all") or "all").strip()
        kinds = list(_INFOGRAPHIC_NAMES) if raw == "all" else [raw]
        if any(kind not in _INFOGRAPHIC_NAMES for kind in kinds):
            return jsonify({"ok": False, "error": "kind must be grid|stack|all"}), 400
        running = _running(_RESULT_JOB_KIND)
        if running:
            return jsonify({"ok": True, "job_id": running, "already_running": True})
        return jsonify({"ok": True, "job_id": JOB_REGISTRY.spawn(
            f"TNG: pull the {' + '.join(kinds)} result", lambda cap: _job_pull_results(cap, kinds),
            kind=_RESULT_JOB_KIND)})

    def _serve_cached(kind: str, mimetype: str, **kwargs):
        path = _artifact_local(kind)
        if not os.path.isfile(path):
            return jsonify({"ok": False, "error": (
                f"no {kind} pulled yet — run the job, then pull its result "
                "(POST /api/tng/result/pull).")}), 404
        return send_file(path, mimetype=mimetype, max_age=0, **kwargs)

    @app.route("/tng/result/grid.png")
    def tng_result_grid():
        """The last pulled ``tng_grid`` image (cache-only)."""
        return _serve_cached("grid", "image/png")

    @app.route("/tng/result/stack.fits")
    def tng_result_stack():
        """The last pulled ``tng_stack`` FITS as a download (cache-only)."""
        return _serve_cached("stack", "application/fits", as_attachment=True,
                             download_name="TNG_stack.fits")
