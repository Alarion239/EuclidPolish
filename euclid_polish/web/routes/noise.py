"""Route for the Noise tab: the per-scene sky-noise distribution."""
from __future__ import annotations

from flask import jsonify

from euclid_polish.web.helpers.noise_levels import noise_levels_payload


def register(app):
    @app.route("/api/noise")
    def api_noise():
        return jsonify(noise_levels_payload())
