"""catalog routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

from euclid_polish.config import Config
from euclid_polish.web import euclid_session
from flask import render_template
from euclid_polish.web.helpers.status import _catalog_status, _cutout_layout_status


def register(app):

    # ---------------- Catalog page ----------------
    @app.route("/catalog")
    def catalog_page():
        return render_template(
            "catalog.html",
            status=_catalog_status(),
            bands=Config.BANDS,
            authenticated=euclid_session.is_authenticated(),
            current_user=euclid_session.current_user(),
            cutout_layout=_cutout_layout_status(),
        )
