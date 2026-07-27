"""Routes for the field-statistics and population-comparison workspace."""
from __future__ import annotations

from flask import jsonify, request

from euclid_polish.web import euclid_session, fasrc_fetcher
from euclid_polish.web.helpers.paths import _sky_records_remote_dir
from euclid_polish.web.helpers.population_comparison import (
    availability,
    build_comparison,
    query_euclid_population,
    read_comparison,
)
from euclid_polish.web.jobs import REGISTRY


def register(app):
    @app.route("/api/population-comparison")
    def api_population_comparison():
        return jsonify({
            "comparison": read_comparison(),
            "availability": availability(),
            "authenticated": euclid_session.is_authenticated(),
        })

    @app.route("/api/population-comparison/build", methods=["POST"])
    def api_population_comparison_build():
        job_id = REGISTRY.spawn(
            label="population comparison: local fields",
            target=lambda cap: build_comparison(
                progress=lambda done, total, label: cap.tick(done, total, label)
            ),
        )
        return jsonify({"ok": True, "job_id": job_id})

    @app.route("/api/population-comparison/query-euclid", methods=["POST"])
    def api_population_comparison_query_euclid():
        if euclid_session.catalog() is None:
            return jsonify({"ok": False, "error": (
                "Log in to the Euclid archive on the Catalog page first."
            )}), 400
        try:
            ra = float(request.form["ra"])
            dec = float(request.form["dec"])
            radius_arcmin = float(request.form["radius_arcmin"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"ok": False, "error": (
                "ra, dec, and radius_arcmin must be valid numbers"
            )}), 400
        if not 0 <= ra < 360:
            return jsonify({"ok": False, "error": "ra must be in [0, 360)"}), 400
        if not -90 <= dec <= 90:
            return jsonify({"ok": False, "error": "dec must be in [-90, 90]"}), 400
        if not 0 < radius_arcmin <= 30:
            return jsonify({"ok": False, "error": (
                "radius_arcmin must be greater than 0 and at most 30"
            )}), 400

        def run(cap):
            cap.tick(0, 1, "Euclid MER cone query")
            meta = query_euclid_population(ra, dec, radius_arcmin)
            cap.tick(1, 1, "Euclid MER cone query")
            cap.write(
                f"cached {meta['rows']} clean sources over "
                f"{meta['area_arcmin2']:.2f} arcmin²\n"
            )
            return meta

        job_id = REGISTRY.spawn(
            label="population comparison: Euclid cone",
            target=run,
        )
        return jsonify({"ok": True, "job_id": job_id})

    @app.route("/api/population-comparison/sync-training-catalog", methods=["POST"])
    def api_population_comparison_sync_training_catalog():
        remote = f"{_sky_records_remote_dir()}/sources_train.csv"

        def run(cap):
            cap.tick(0, 1, "training source catalog")
            result = fasrc_fetcher.fetch_one_file(
                remote, force=True, max_bytes=1024 * 1024 * 1024
            )
            if not result.ok:
                raise RuntimeError(result.error or "training source catalog sync failed")
            cap.tick(1, 1, "training source catalog")
            cap.write(
                f"synced sources_train.csv ({result.size_bytes or 0:,} bytes)\n"
            )
            return {"path": result.local_path, "size_bytes": result.size_bytes}

        job_id = REGISTRY.spawn(
            label="population comparison: training source catalog",
            target=run,
        )
        return jsonify({"ok": True, "job_id": job_id})
