"""cutouts routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

from euclid_polish.config import Config
from flask import abort
from flask import jsonify
from flask import redirect
from flask import render_template
from flask import request
from flask import send_file
from flask import url_for
import io
from euclid_polish.web.helpers.fits_render import _list_band_cutouts, _render_fits_to_png, _resolve_cutout_path
from euclid_polish.web.helpers.paths import _safe_relpath
from euclid_polish.web.helpers.status import _catalog_status, _cutout_layout_status, _ensure_local_star_cutout, _valid_4band_stars


def register(app):

    # ---------------- Cutouts page ----------------
    @app.route("/cutouts")
    def cutouts_page():
        size, ids = _valid_4band_stars(force=True)
        return render_template(
            "cutouts.html",
            catalog=_catalog_status(),
            bands=Config.BANDS,
            cutout_layout=_cutout_layout_status(),
            default_vis_pixels=Config.DEFAULT_CUTOUT_SIZE,
            n_valid=len(ids),
            cutout_size=size,
        )

    # ---------------- Cutout gallery + live FITS→PNG ----------------
    @app.route("/cutouts/<band_name>")
    def cutouts_gallery(band_name: str):
        """Per-band paginated thumbnail gallery."""
        try:
            band = Config.get_band(band_name)
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
        end   = start + per_page
        return render_template(
            "cutouts_gallery.html",
            band=band, files=files[start:end],
            total=total, page=page, n_pages=n_pages,
            per_page=per_page, output_dir=out_dir,
        )

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

    # ---------------- Star cutouts navigator (merged into /cutouts) ------
    #
    # Same navigator as /sky, but one real-Euclid star per index: only stars
    # valid in ALL 4 bands. Each band's cutout is pulled from FASRC ONCE to
    # the canonical local dir (data/euclid_stars/cutouts/<band>/) and SAVED
    # there persistently, so revisiting never re-pulls it. The UI lives at
    # the top of the /cutouts page; these endpoints back it.
    @app.route("/api/star-cutouts/totals")
    def api_star_cutouts_totals():
        size, ids = _valid_4band_stars(force=True)
        return jsonify({"count": len(ids), "size": size})

    def _resolve_star_cutout(band: str, i_arg: str):
        """``(BandConfig, local_path)`` for the i-th valid-in-all-4 star's
        ``band`` cutout, pulling+saving it locally on first request; abort
        otherwise. Shared by the image + inspect routes."""
        try:
            band_cfg = Config.get_band(band)
        except Exception:
            abort(404)
        try:
            idx = int(i_arg)
        except (TypeError, ValueError):
            idx = 0
        # Cached catalog (force=False) — the page load already pulled it.
        size, ids = _valid_4band_stars(force=False)
        if not ids or size is None:
            abort(404)
        idx = max(0, min(idx, len(ids) - 1))
        local_path = _ensure_local_star_cutout(band, ids[idx], size)
        if not local_path:
            abort(404)
        return band_cfg, local_path

    @app.route("/view/star-cutout")
    def view_star_cutout():
        band_cfg, local_path = _resolve_star_cutout(
            request.args.get("band", "VIS"), request.args.get("i", "0"))
        png = _render_fits_to_png(local_path, band_cfg)
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=0)

    @app.route("/star-cutout/inspect")
    def star_cutout_inspect():
        """Save the cutout locally (once) then jump into the universal FITS
        inspector — header + download — for the navigator's current star."""
        _band_cfg, local_path = _resolve_star_cutout(
            request.args.get("band", "VIS"), request.args.get("i", "0"))
        return redirect(url_for("inspect_fits_page",
                                fits=_safe_relpath(local_path)))
