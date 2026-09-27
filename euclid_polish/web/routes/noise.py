"""Routes for the Noise tab: the per-scene sky-noise distribution."""
from __future__ import annotations

from flask import jsonify

from euclid_polish.web.helpers.noise_levels import noise_levels_payload, noise_position


def register(app):
    @app.route("/api/noise")
    def api_noise():
        return jsonify(noise_levels_payload())

    @app.get("/api/noise/positions/<tile>")
    def api_noise_position(tile: str):
        """One measured position with its 4×4 sub-tile levels (read-only)."""
        position = noise_position(tile)
        if position is None:
            return jsonify({"ok": False, "error": f"no measured noise position in tile {tile}"}), 404
        return jsonify(position)
