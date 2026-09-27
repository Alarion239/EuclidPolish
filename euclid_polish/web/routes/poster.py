"""Poster cutout results — the latest ``poster_cutout`` job artifact.

The ``poster_cutout`` FASRC step (Figures › Plates) runs
``scripts/fasrc_poster_cutout.py --save`` on the node, writing a clean 4-band
FITS + a preview PNG to ``$EUCLID_POLISH_DATA_DIR/_poster/`` (fixed names, so
each submit overwrites the previous result).

* ``POST /poster/result/pull`` (FASRC) pulls both into the local FASRC cache
  and archives a changed preview PNG into ``data/vis/poster/`` (timestamped,
  so each generated cutout is kept).
* ``GET /poster/result/status`` / ``cutout.png`` / ``cutout.fits`` serve the
  last pulled copy — local, so they work offline and a GET never writes.
"""
from __future__ import annotations

import glob
import os
import time
from typing import Any

from flask import jsonify, send_file

from euclid_polish.config import Config
from euclid_polish.web import fasrc_config
from euclid_polish.web.fasrc_fetcher import _local_path_for, fetch_one_file
from euclid_polish.web.fasrc_gate import requires_fasrc

# Must match scripts/fasrc_poster_cutout.py (OUTPUT_SUBDIR / FITS_NAME / PNG_NAME).
_POSTER_SUBDIR = "_poster"
_FITS_NAME = "poster_cutout.fits"
_PNG_NAME = "poster_cutout.png"
_ARTIFACTS = {"png": _PNG_NAME, "fits": _FITS_NAME}

_NO_RESULT = ("no poster cutout pulled yet — submit the poster_cutout step, then "
              "pull its result once the job completes")


def _vis_poster_dir() -> str:
    """``data/vis/poster/`` (resolved per call; tests re-point ``Config``)."""
    return os.path.join(Config.VIS_DIR, "poster")


def _remote(name: str) -> str:
    return f"{fasrc_config.load().data_dir}/{_POSTER_SUBDIR}/{name}"


def _local(name: str) -> str:
    """Where :func:`fetch_one_file` keeps the pulled copy (no SSH)."""
    return _local_path_for(_remote(name))


def _file_state(path: str) -> dict[str, Any] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return {"size": stat.st_size, "mtime": stat.st_mtime,
            "pulled_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime))}


def archive_png_to_vis(png_bytes: bytes) -> str | None:
    """Timestamped copy into ``data/vis/poster/``; ``None`` when the newest
    archived PNG is byte-identical (a re-pull adds no duplicate) or on error."""
    directory = _vis_poster_dir()
    try:
        os.makedirs(directory, exist_ok=True)
        prior = sorted(glob.glob(os.path.join(directory, "poster_cutout_*.png")))
        if prior:
            try:
                with open(prior[-1], "rb") as fh:
                    if fh.read() == png_bytes:
                        return None
            except OSError:
                pass
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(directory, f"poster_cutout_{ts}.png")
        if os.path.exists(path):  # >1 pull in the same second
            path = os.path.join(
                directory, f"poster_cutout_{ts}_{int(time.time() * 1000) % 1000:03d}.png")
        with open(path, "wb") as fh:
            fh.write(png_bytes)
        return path
    except OSError:
        return None


def register(app):

    def _serve(kind: str, mimetype: str, *, as_attachment=False):
        path = _local(_ARTIFACTS[kind])
        if not os.path.isfile(path):
            return jsonify({"ok": False, "error": _NO_RESULT}), 404
        return send_file(path, mimetype=mimetype, max_age=0, as_attachment=as_attachment,
                         download_name=_ARTIFACTS[kind])

    @app.get("/poster/result/status")
    def poster_result_status():
        files = {kind: _file_state(_local(name)) for kind, name in _ARTIFACTS.items()}
        return jsonify({"ok": True, "available": files["png"] is not None, **files,
                        "archive_dir": os.path.relpath(_vis_poster_dir(), os.getcwd())})

    @app.post("/poster/result/pull")
    @requires_fasrc
    def poster_result_pull():
        """Pull the latest cutout (force: a fresh job result is never masked by
        the fetcher's TTL cache) and archive a changed preview PNG."""
        out: dict[str, Any] = {"ok": True, "errors": {}}
        for kind, name in _ARTIFACTS.items():
            result = fetch_one_file(_remote(name), force=True)
            if not result.ok or not result.local_path:
                out["errors"][kind] = result.error or "not found on FASRC"
                out[kind] = None
                continue
            out[kind] = _file_state(result.local_path)
            if kind == "png":
                try:
                    with open(result.local_path, "rb") as fh:
                        archived = archive_png_to_vis(fh.read())
                except OSError:
                    archived = None
                out["archived"] = os.path.relpath(archived, os.getcwd()) if archived else None
        if out["png"] is None and out["fits"] is None:
            return jsonify({"ok": False, "error": f"{_NO_RESULT} [{out['errors'].get('png', '')}]",
                            "errors": out["errors"]}), 404
        return jsonify(out)

    @app.get("/poster/result/cutout.png")
    def poster_result_png():
        return _serve("png", "image/png")

    @app.get("/poster/result/cutout.fits")
    def poster_result_fits():
        return _serve("fits", "application/fits", as_attachment=True)
