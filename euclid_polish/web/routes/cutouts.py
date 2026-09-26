"""cutouts routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

import io

from flask import abort, jsonify, request, send_file

from euclid_polish.config import Config
from euclid_polish.web.fasrc_gate import requires_fasrc
from euclid_polish.web.helpers.fits_render import (
    _list_band_cutouts,
    _render_fits_to_png,
    _resolve_cutout_path,
)
from euclid_polish.web.helpers.status import _valid_4band_stars


def register(app):

    @app.route("/api/cutouts/<band_name>/list.json")
    def api_cutouts_list(band_name: str):
        """Paginated per-band cutout filenames as JSON for the React gallery.
        Thumbnails load from /cutout-image/<band>/<filename>?size=…&output_dir=…"""
        try:
            Config.get_band(band_name)
        except Exception:
            abort(404)
        out_dir = request.args.get("output_dir", Config.DEFAULT_OUTPUT_DIR)
        try:
            page = max(1, int(request.args.get("page", 1)))
        except ValueError:
            page = 1
        per_page = 60
        files = _list_band_cutouts(band_name, out_dir)
        total = len(files)
        n_pages = max(1, (total + per_page - 1) // per_page)
        page = min(page, n_pages)
        start = (page - 1) * per_page
        return jsonify({
            "band": band_name, "files": files[start:start + per_page],
            "total": total, "page": page, "n_pages": n_pages,
            "per_page": per_page, "output_dir": out_dir,
        })

    @app.route("/cutout-image/<band_name>/<path:filename>")
    def cutout_image(band_name: str, filename: str):
        out_dir = request.args.get("output_dir", Config.DEFAULT_OUTPUT_DIR)
        try:
            size = int(request.args.get("size", 0)) or None
        except ValueError:
            size = None
        if size is not None and (size < 16 or size > 2048):
            abort(400)
        try:
            band = Config.get_band(band_name)
        except ValueError:
            abort(404)
        fits_path = _resolve_cutout_path(band_name, filename, out_dir)
        png = _render_fits_to_png(fits_path, band, size=size)
        return send_file(io.BytesIO(png), mimetype="image/png",
                         max_age=3600)

    # ---------------- Star cutouts (valid in all 4 bands) ----------------
    # The viewer collection ``cutouts`` serves the stars themselves; this
    # counts them for the Cutouts tab.
    @app.route("/api/star-cutouts/totals")
    @requires_fasrc
    def api_star_cutouts_totals():
        size, ids = _valid_4band_stars(force=True)
        return jsonify({"count": len(ids), "size": size})
