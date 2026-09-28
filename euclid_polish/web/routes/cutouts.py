"""Star catalogue + star cutout routes (Synthetic › PSF, catalogue and cutouts).

Everything here reads the synchronised FASRC-mirror ``stars.csv`` and the
local cutout cache only: no SSH, works offline. The explicit catalogue pull
is ``POST /api/status/refresh-catalog`` (``routes/files.py``); the cutouts
viewer (collection ``cutouts``) pulls single star cutouts on demand.
"""
from __future__ import annotations

import io
import os
import re

from flask import abort, jsonify, request, send_file

from euclid_polish.config import Config
from euclid_polish.web.helpers import star_catalog
from euclid_polish.web.helpers.fits_render import (
    _list_band_cutouts,
    _render_fits_to_png,
    _render_fits_to_png_adaptive,
    _resolve_cutout_path,
)
from euclid_polish.web.helpers.paths import _abort_json, root_of
from euclid_polish.web.helpers.status import _cached_valid_4band_stars, catalog_cache_info

_CUTOUT_NAME = re.compile(r"^star_(\d+)_(\d+)\.fits$", re.IGNORECASE)


def _output_dir() -> str:
    """The request's ``output_dir`` (blank → ``Config.DEFAULT_OUTPUT_DIR``),
    jailed to the inspectable data roots (:func:`helpers.paths.root_of`,
    after symlink resolution): 403 JSON outside them, like ``/api/inspect``."""
    raw = (request.args.get("output_dir") or "").strip() or Config.DEFAULT_OUTPUT_DIR
    real = os.path.realpath(raw)
    if root_of(real) is None:
        _abort_json(403, f"{raw} is outside the inspectable data roots")
    return raw


def gallery_items(files: list[str], stars: dict[int, dict]) -> list[dict]:
    """``star_<id>_<size>.fits`` names → ``{file, id, size, ra, dec, mag}``."""
    items = []
    for name in files:
        match = _CUTOUT_NAME.match(name)
        sid = int(match.group(1)) if match else None
        size = int(match.group(2)) if match else None
        star = stars.get(sid, {}) if sid is not None else {}
        items.append({"file": name, "id": sid, "size": size, "ra": star.get("ra"),
                      "dec": star.get("dec"), "mag": star.get("mag")})
    return items


def register(app):

    @app.route("/api/catalog/stars")
    def api_catalog_stars():
        """The star catalogue explorer's payload (see ``helpers/star_catalog``):
        compact rows of the FASRC-mirror ``stars.csv`` + summary, per-band
        validity and the mirror's freshness. Cache-only."""
        return jsonify(star_catalog.stars_payload())

    @app.route("/api/cutouts/<band_name>/list.json")
    def api_cutouts_list(band_name: str):
        """Paginated per-band cutout files of the local cache, each with its
        star id, size and catalogue position / magnitude. Thumbnails load from
        /cutout-image/<band>/<filename>?size=…&output_dir=…"""
        try:
            Config.get_band(band_name)
        except Exception:
            abort(404)
        out_dir = _output_dir()
        try:
            page = max(1, int(request.args.get("page", 1)))
        except ValueError:
            page = 1
        try:
            per_page = max(12, min(int(request.args.get("per_page", 60)), 240))
        except ValueError:
            per_page = 60
        files = _list_band_cutouts(band_name, out_dir)
        total = len(files)
        n_pages = max(1, (total + per_page - 1) // per_page)
        page = min(page, n_pages)
        start = (page - 1) * per_page
        shown = files[start:start + per_page]
        return jsonify({
            "band": band_name, "files": shown,
            "items": gallery_items(shown, star_catalog.stars_by_id()),
            "total": total, "page": page, "n_pages": n_pages,
            "per_page": per_page, "output_dir": out_dir,
        })

    @app.route("/cutout-image/<band_name>/<path:filename>")
    def cutout_image(band_name: str, filename: str):
        """One cached cutout as a gray_r PNG. ``stretch=band`` (default): the
        band's fixed asinh knee, clipped at the 1 / 99.7 percentiles;
        ``stretch=star``: the cutout's own min–max range under a soft asinh
        (the Synthetic › PSF gallery), so a bright star's core keeps its
        shape."""
        out_dir = _output_dir()
        try:
            size = int(request.args.get("size", 0)) or None
        except ValueError:
            size = None
        if size is not None and (size < 16 or size > 2048):
            abort(400)
        stretch = request.args.get("stretch", "band")
        if stretch not in ("band", "star"):
            abort(400)
        try:
            band = Config.get_band(band_name)
        except ValueError:
            abort(404)
        fits_path = _resolve_cutout_path(band_name, filename, out_dir)
        png = (_render_fits_to_png_adaptive(fits_path, size or 256) if stretch == "star"
               else _render_fits_to_png(fits_path, band, size=size))
        return send_file(io.BytesIO(png), mimetype="image/png",
                         max_age=3600)

    # ---------------- Star cutouts (valid in all 4 bands) ----------------
    # The viewer collection ``cutouts`` serves the stars themselves; this
    # counts them for the Cutouts tab — from the synchronised mirror, so it
    # answers offline (a stale mirror is flagged by ``age_s``).
    @app.route("/api/star-cutouts/totals")
    def api_star_cutouts_totals():
        size, ids = _cached_valid_4band_stars()
        info = catalog_cache_info()
        return jsonify({"count": len(ids), "size": size, "cached": True,
                        "catalog": {k: info[k] for k in ("present", "path", "mtime", "age_s")}})
