"""Route for the realism overview (Synthetic › Status): readiness of every prior."""
from __future__ import annotations

from flask import jsonify

from euclid_polish.web.helpers.realism_overview import overview_payload
from euclid_polish.web.routes.system import check_records_noise


def register(app):
    @app.get("/api/realism/overview")
    def api_realism_overview():
        """Read-only readiness checklist (``helpers/realism_overview.py``)."""
        return jsonify(overview_payload(check_records_noise=check_records_noise))
