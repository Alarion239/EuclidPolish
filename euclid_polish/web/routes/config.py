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
        before = job_config.load().vis_pixels
        cfg = job_config.update({
            "vis_pixels":    request.form.get("vis_pixels"),
            "stars_per_psf": request.form.get("stars_per_psf"),
            "n_train":       request.form.get("n_train"),
            "n_valid":       request.form.get("n_valid"),
            "hr_image_size": request.form.get("hr_image_size"),
            "asinh_scale":   request.form.get("asinh_scale"),
            "star_density_arcmin2": request.form.get("star_density_arcmin2"),
            "star_mag_slope":       request.form.get("star_mag_slope"),
            "star_mag_bright":      request.form.get("star_mag_bright"),
            "star_mag_faint":       request.form.get("star_mag_faint"),
        })
        note = None
        try:
            requested = int(request.form.get("vis_pixels", cfg.vis_pixels))
            if requested != cfg.vis_pixels:
                note = (f"VIS cutout must be odd — adjusted "
                        f"{requested} → {cfg.vis_pixels}.")
        except (TypeError, ValueError):
            pass
        return jsonify({"ok": True, "config": cfg.to_dict(), "note": note})
