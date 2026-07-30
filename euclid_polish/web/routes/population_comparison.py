"""Routes for the field-statistics and population-comparison workspace."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from flask import jsonify, request

from euclid_polish.web import euclid_session, fasrc_fetcher, job_config
from euclid_polish.web.helpers.paths import _sky_records_remote_dir
from euclid_polish.web.helpers.population_comparison import (
    availability,
    build_comparison,
    query_euclid_population,
    query_euclid_population_multi,
    read_comparison,
    read_cosmos_euclid_fit,
    refresh_population_comparison,
)
from euclid_polish.web.jobs import REGISTRY
from euclid_polish.web.remote import ensure_ssh_connected


def _fit_and_evaluate_cached_cones(
    cap, *, progress_start: int = 0, progress_total: int = 2,
) -> dict:
    """Fit cached Euclid cones, apply their density, and refresh evaluations."""
    cap.tick(progress_start, progress_total, "fit COSMOS observation layer")
    project_root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [sys.executable, "scripts/fit_cosmos_euclid_counts.py"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    fit_payload = read_cosmos_euclid_fit()
    if fit_payload is None:
        raise RuntimeError("fit completed without a readable fit artifact")

    density_fit = fit_payload.get("generator_density_recommendation") or {}
    configured_density = None
    if density_fit.get("apply_to_config"):
        configured_density = float(density_fit["density_arcmin2"])
        job_config.update({
            "galaxy_density_arcmin2": f"{configured_density:.6g}",
        })
        cap.write(
            "updated generator galaxy density to "
            f"{configured_density:.2f} / arcmin²\n"
        )

    cap.tick(
        progress_start + 1, progress_total,
        "refresh field-statistics evaluations",
    )
    refreshed = refresh_population_comparison()
    cap.tick(progress_start + 2, progress_total, "fit and evaluations ready")

    selected_fit = (
        fit_payload.get("local_normalization_sensitivity_fit")
        if density_fit.get("apply_to_config")
        else fit_payload.get("fit")
    ) or {}
    if selected_fit:
        cap.write(
            "fit deviance / dof "
            f"{float(selected_fit.get('poisson_deviance', 0.0)):.2f} / "
            f"{int(selected_fit.get('dof', 0))}; "
            f"VIS 50% completeness "
            f"{float(selected_fit.get('completeness_m50', 0.0)):.2f}\n"
        )
    if refreshed is None:
        cap.write(
            "Fit saved; run Rebuild statistics once to create the field cache.\n"
        )
    else:
        cap.write("refreshed cached population evaluations\n")
    return {
        "fit": fit_payload,
        "configured_density_arcmin2": configured_density,
        "population_refreshed": refreshed is not None,
    }


def register(app):
    @app.route("/api/population-comparison")
    def api_population_comparison():
        comparison = read_comparison()
        include_training = request.args.get(
            "include_training", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if comparison is not None:
            comparison = dict(comparison)
            if include_training:
                comparison["population"] = comparison.get(
                    "population_with_training",
                    comparison.get("population"),
                )
            population = dict(comparison.get("population") or {})
            population["cosmos_euclid_fit"] = read_cosmos_euclid_fit()
            comparison["population"] = population
            comparison.pop("population_with_training", None)
        return jsonify({
            "comparison": comparison,
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
            cap.tick(0, 2, "Euclid MER cone query")
            meta = query_euclid_population(ra, dec, radius_arcmin)
            cap.tick(1, 2, "population histograms")
            refreshed = refresh_population_comparison()
            cap.tick(2, 2, "population histograms")
            cap.write(
                f"cached {meta['rows']} clean sources over "
                f"{meta['area_arcmin2']:.2f} arcmin²\n"
            )
            if refreshed is None:
                cap.write("Run Measure local fields to create the comparison cache.\n")
            else:
                cap.write("updated Euclid population statistics\n")
            return meta

        job_id = REGISTRY.spawn(
            label="population comparison: Euclid cone",
            target=run,
        )
        return jsonify({"ok": True, "job_id": job_id})

    @app.route(
        "/api/population-comparison/query-euclid-multi", methods=["POST"]
    )
    def api_population_comparison_query_euclid_multi():
        if euclid_session.catalog() is None:
            return jsonify({"ok": False, "error": (
                "Log in to the Euclid archive on the Catalog page first."
            )}), 400
        try:
            count = int(request.form.get("count", "6"))
            radius = float(request.form.get(
                "radius_arcmin", "3.404"
            ))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "invalid cone settings"}), 400
        if not 1 <= count <= 12 or not 0 < radius <= 30:
            return jsonify({"ok": False, "error": (
                "count must be 1–12 and radius_arcmin must be in (0, 30]"
            )}), 400

        def run(cap):
            meta = query_euclid_population_multi(
                count=count,
                radius_arcmin=radius,
                progress=lambda done, _total, label: cap.tick(
                    done, count + 2, label
                ),
            )
            _fit_and_evaluate_cached_cones(
                cap,
                progress_start=count,
                progress_total=count + 2,
            )
            cap.write(
                f"cached {meta['rows']} unique sources from "
                f"{meta['cone_count']} cones over "
                f"{meta['area_arcmin2']:.2f} arcmin²\n"
            )
            return meta

        job_id = REGISTRY.spawn(
            label=f"population comparison: {count} random Euclid cones",
            target=run,
        )
        return jsonify({"ok": True, "job_id": job_id})

    @app.route(
        "/api/population-comparison/fit-euclid", methods=["POST"]
    )
    def api_population_comparison_fit_euclid():
        if not availability().get("euclid_catalog", {}).get("cached"):
            return jsonify({
                "ok": False,
                "error": "Query and cache at least one Euclid cone first.",
            }), 400

        job_id = REGISTRY.spawn(
            label="population comparison: fit cached Euclid cones",
            target=_fit_and_evaluate_cached_cones,
        )
        return jsonify({"ok": True, "job_id": job_id})

    @app.route("/api/population-comparison/sync-training-catalog", methods=["POST"])
    def api_population_comparison_sync_training_catalog():
        remote = f"{_sky_records_remote_dir()}/sources_train.csv"

        def run(cap):
            cap.tick(0, 3, "connecting to FASRC")
            ensure_ssh_connected()
            cap.tick(1, 3, "training source catalog")
            result = fasrc_fetcher.fetch_one_file(
                remote, force=True, max_bytes=1024 * 1024 * 1024
            )
            if not result.ok:
                raise RuntimeError(result.error or "training source catalog sync failed")
            cap.tick(2, 3, "population histograms")
            refresh_population_comparison()
            cap.tick(3, 3, "population histograms")
            cap.write(
                f"synced sources_train.csv ({result.size_bytes or 0:,} bytes)\n"
            )
            return {"path": result.local_path, "size_bytes": result.size_bytes}

        job_id = REGISTRY.spawn(
            label="population comparison: training source catalog",
            target=run,
        )
        return jsonify({"ok": True, "job_id": job_id})
