"""Routes for the field-statistics and population-comparison workspace."""
from __future__ import annotations

from flask import jsonify, request

from euclid_polish.web import euclid_session, fasrc_fetcher
from euclid_polish.web.helpers.galaxy_distributions import build_galaxy_distributions
from euclid_polish.web.helpers.paths import _sky_records_remote_dir
from euclid_polish.web.helpers.population_comparison import (
    availability,
    build_comparison,
    read_comparison,
    read_previous_comparison,
    refresh_population_comparison,
)
from euclid_polish.web.jobs import REGISTRY
from euclid_polish.web.remote import ensure_ssh_connected


def register(app):
    @app.route("/api/population-comparison")
    def api_population_comparison():
        """The field statistics. ``comparison`` is the cache at the current
        schema, else null; ``previous`` is then the last cache built at an
        older schema (Synthetic › Fields shows it with a stale badge and a
        Measure button, never rebuilding it on a visit), else null."""
        include_training = request.args.get(
            "include_training", ""
        ).strip().lower() in {"1", "true", "yes", "on"}

        def variant(payload):
            if payload is None:
                return None
            payload = dict(payload)
            if include_training:
                payload["population"] = payload.get(
                    "population_with_training",
                    payload.get("population"),
                )
            payload.pop("population_with_training", None)
            return payload

        comparison = variant(read_comparison())
        previous = variant(read_previous_comparison()) if comparison is None else None
        return jsonify({
            "comparison": comparison,
            "previous": previous,
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

    @app.route("/api/population-comparison/sync-training-catalog", methods=["POST"])
    def api_population_comparison_sync_training_catalog():
        """Pull ``sources_train.csv``, refresh the census and (``rebuild=1``,
        the Realism header's one sync action) rebuild the galaxy plots so
        their training variant exists — one job instead of a client-side
        chain that breaks when the page is left."""
        remote = f"{_sky_records_remote_dir()}/sources_train.csv"
        rebuild = request.form.get("rebuild", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        total = 4 if rebuild else 3

        def run(cap):
            cap.tick(0, total, "connecting to FASRC")
            ensure_ssh_connected()
            cap.tick(1, total, "training source catalog")
            result = fasrc_fetcher.fetch_one_file(
                remote, force=True, max_bytes=1024 * 1024 * 1024
            )
            if not result.ok:
                raise RuntimeError(result.error or "training source catalog sync failed")
            cap.tick(2, total, "population histograms")
            refresh_population_comparison()
            cap.tick(3, total, "population histograms")
            cap.write(
                f"synced sources_train.csv ({result.size_bytes or 0:,} bytes)\n"
            )
            plots = None
            if rebuild:
                cap.tick(3, total, "rebuild galaxy plots")
                plots = build_galaxy_distributions()["version"]
                cap.tick(4, total, "galaxy plots rebuilt")
            return {"path": result.local_path, "size_bytes": result.size_bytes,
                    "galaxy_plots_version": plots}

        job_id = REGISTRY.spawn(
            label="population comparison: training source catalog",
            target=run,
        )
        return jsonify({"ok": True, "job_id": job_id})
