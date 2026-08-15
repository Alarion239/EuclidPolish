"""Universal job-config page + save endpoint (extracted into its own tab).

The ``/config`` tab edits the shared per-job knobs (see
:mod:`euclid_polish.web.job_config`) and persists them so they're consistent
across reloads/relaunches and get injected into the relevant job submissions.
Reachable offline (no FASRC connection needed) — it's local persistence.
"""
from __future__ import annotations

from flask import jsonify, render_template, request

from euclid_polish.web import job_config


def register(app):

    @app.route("/config")
    def config_page():
        return render_template("config.html", cfg=job_config.load().to_dict())

    @app.route("/api/config")
    def api_config_get():
        return jsonify({"ok": True, "config": job_config.load().to_dict()})

    @app.route("/api/config/save", methods=["POST"])
    def api_config_save():
        # Forward the whole form; ``update`` ignores unknown keys and blanks, so
        # every JobConfig field on the page persists without a hand-maintained
        # allowlist here.
        cfg = job_config.update(request.form.to_dict())
        note = None
        try:
            requested = int(request.form.get("vis_pixels", cfg.vis_pixels))
            if requested != cfg.vis_pixels:
                note = (f"VIS cutout must be odd — adjusted "
                        f"{requested} → {cfg.vis_pixels}.")
        except (TypeError, ValueError):
            pass
        return jsonify({"ok": True, "config": cfg.to_dict(), "note": note})
