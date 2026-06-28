"""Poster cutout results — fetch the latest ``poster_cutout`` job artifact.

The ``poster_cutout`` FASRC step card (mounted on the Visualization tab) runs
``scripts/fasrc_poster_cutout.py --save`` on the node, writing a clean 4-band
FITS + a preview PNG to ``$EUCLID_POLISH_DATA_DIR/_poster/`` (fixed names, so
each submit overwrites the previous result). These routes pull the latest one
for preview (PNG) / download (FITS), mirroring the TNG grid/stack result routes.
"""
from __future__ import annotations

import glob
import os
import time

from flask import jsonify, send_file

from euclid_polish.config import Config
from euclid_polish.web import fasrc_config
from euclid_polish.web.fasrc_fetcher import fetch_one_file

# Must match scripts/fasrc_poster_cutout.py (OUTPUT_SUBDIR / FITS_NAME / PNG_NAME).
_POSTER_SUBDIR = "_poster"
_FITS_NAME = "poster_cutout.fits"
_PNG_NAME = "poster_cutout.png"

# Archive pulled preview PNGs into the Visualization gallery (data/vis/poster/),
# so each generated cutout is kept + gets the 📌-track button.
_VIS_POSTER_DIR = os.path.join(Config.VIS_DIR, "poster")


def _archive_png_to_vis(png_bytes: bytes) -> None:
    """Best-effort timestamped copy into ``data/vis/poster/`` (never raises).

    Skips the write when the most recent archived PNG is byte-identical, so
    reloading the same result doesn't pile up duplicates."""
    try:
        os.makedirs(_VIS_POSTER_DIR, exist_ok=True)
        prior = sorted(glob.glob(os.path.join(_VIS_POSTER_DIR, "poster_cutout_*.png")))
        if prior:
            try:
                with open(prior[-1], "rb") as fh:
                    if fh.read() == png_bytes:
                        return
            except OSError:
                pass
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(_VIS_POSTER_DIR, f"poster_cutout_{ts}.png")
        if os.path.exists(path):  # >1 save in the same second
            path = os.path.join(
                _VIS_POSTER_DIR,
                f"poster_cutout_{ts}_{int(time.time() * 1000) % 1000:03d}.png")
        with open(path, "wb") as fh:
            fh.write(png_bytes)
    except Exception:
        pass


def register(app):

    def _artifact_remote(name: str) -> str:
        cfg = fasrc_config.load()
        return f"{cfg.data_dir}/{_POSTER_SUBDIR}/{name}"

    def _serve(name: str, mimetype: str, *, as_attachment=False,
               download_name=None, archive_png=False):
        # force=True so a freshly-rendered job result isn't masked by the
        # fetcher's TTL cache.
        result = fetch_one_file(_artifact_remote(name), force=True)
        if not result.ok or not result.local_path:
            hint = ("no result yet — pick a mode, submit the job above, then "
                    "load the result once it completes.")
            if result.error:
                hint += f" [{result.error}]"
            return jsonify({"ok": False, "error": hint}), 404
        if archive_png:
            try:
                with open(result.local_path, "rb") as fh:
                    _archive_png_to_vis(fh.read())
            except OSError:
                pass
        return send_file(result.local_path, mimetype=mimetype, max_age=0,
                         as_attachment=as_attachment,
                         download_name=download_name)

    @app.route("/poster/result/cutout.png")
    def poster_result_png():
        return _serve(_PNG_NAME, "image/png", archive_png=True)

    @app.route("/poster/result/cutout.fits")
    def poster_result_fits():
        return _serve(_FITS_NAME, "application/fits", as_attachment=True,
                      download_name=_FITS_NAME)
