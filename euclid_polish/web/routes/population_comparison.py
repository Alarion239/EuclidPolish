"""Routes for the field-statistics and population-comparison workspace."""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

from flask import abort, jsonify, request, send_file

from euclid_polish.web import euclid_session, fasrc_fetcher, job_config
from euclid_polish.web.helpers.paths import _sky_records_remote_dir
from euclid_polish.web.helpers.population_calibration import (
    activate_density_candidate,
    activate_galaxy_recommendation,
    activate_joint_galaxy_candidate,
    activate_photometric_transfer,
    activate_star_candidate,
    density_state,
    galaxy_recommendation_state,
    joint_galaxy_state,
    star_state,
    transfer_state,
)
from euclid_polish.web.helpers.population_comparison import (
    availability,
    build_comparison,
    query_euclid_population,
    query_euclid_population_multi,
    read_comparison,
    read_cosmos_euclid_fit,
    refresh_cached_euclid_population_multi,
    refresh_population_comparison,
)
from euclid_polish.web.helpers.publication_figures import (
    render_population_atlas,
    render_star_population_calibration,
)
from euclid_polish.web.helpers.q1_star_counts import query_q1_phz_star_counts
from euclid_polish.web.helpers.star_population import (
    fit_star_population,
    query_gaia_field_cones,
)
from euclid_polish.web.jobs import REGISTRY
from euclid_polish.web.remote import ensure_ssh_connected

MAX_POPULATION_CONES = 24


def _run_analysis_script(
    project_root: Path, script: str, *arguments: str,
) -> None:
    """Run one local analysis script and preserve its useful failure output."""
    try:
        subprocess.run(
            [sys.executable, script, *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if len(detail) > 4000:
            detail = detail[-4000:]
        message = f"{script} failed"
        if detail:
            message += f":\n{detail}"
        raise RuntimeError(message) from exc


def _fit_and_evaluate_cached_cones(
    cap, *, progress_start: int = 0, progress_total: int = 1,
) -> dict:
    """Fit the shared analytical population to cached local catalogues."""
    project_root = Path(__file__).resolve().parents[3]
    cap.tick(progress_start, progress_total, "fit joint COSMOS + Euclid population")
    _run_analysis_script(
        project_root, "scripts/fit_cosmos_euclid_counts.py"
    )
    fit_payload = read_cosmos_euclid_fit()
    if fit_payload is None:
        raise RuntimeError("fit completed without a readable fit artifact")
    quality = fit_payload.get("fit_quality") or {}
    response = ((fit_payload.get("model") or {}).get("euclid_response") or {})
    cosmos_deviance = quality.get(
        "cosmos_reduced_negative_binomial_deviance",
        quality.get("cosmos_reduced_poisson_deviance", 0.0),
    )
    cap.write(
        "COSMOS reduced deviance "
        f"{float(cosmos_deviance):.2f}; "
        "Euclid reduced deviance "
        f"{float(quality.get('euclid_reduced_poisson_deviance', 0.0)):.2f}; "
        "VIS m50 "
        f"{float(response.get('completeness_m50', 0.0)):.2f}\n"
    )
    cap.write("No TNG catalogue or image was read.\n")
    cap.tick(progress_start + 1, progress_total, "joint population fit ready")
    return {
        "fit": fit_payload,
        "tng_used": False,
    }


def register(app):
    def figure_response(payload: bytes, output_format: str, stem: str):
        mimetype = {
            "png": "image/png",
            "pdf": "application/pdf",
            "svg": "image/svg+xml",
        }[output_format]
        inline = request.args.get("inline", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        return send_file(
            io.BytesIO(payload),
            mimetype=mimetype,
            as_attachment=not inline,
            download_name=f"{stem}.{output_format}",
            max_age=0,
        )

    @app.route("/view/population-atlas")
    def view_population_atlas():
        """Download the reviewed joint population diagnostics as one figure."""
        fit = read_cosmos_euclid_fit()
        if fit is None:
            abort(404)
        output_format = (request.args.get("format") or "png").strip().lower()
        if output_format not in {"png", "pdf", "svg"}:
            abort(400)
        try:
            dpi = int(request.args.get("dpi", "300"))
            candidate = joint_galaxy_state().get("candidate") or {}
            payload = render_population_atlas(
                fit,
                magnitude_plot=candidate.get("magnitude_plot"),
                output_format=output_format,
                dpi=dpi,
            )
        except (TypeError, ValueError):
            abort(400)
        return figure_response(
            payload, output_format, "euclidpolish_population_atlas",
        )

    @app.route("/view/star-population-calibration")
    def view_star_population_calibration():
        """Render the reviewed Gaia × Euclid stellar-prior diagnostics."""
        state = star_state()
        calibration = state.get("active") or state.get("candidate")
        if not calibration:
            abort(404)
        output_format = (request.args.get("format") or "png").strip().lower()
        if output_format not in {"png", "pdf", "svg"}:
            abort(400)
        try:
            dpi = int(request.args.get("dpi", "300"))
            payload = render_star_population_calibration(
                calibration, output_format=output_format, dpi=dpi,
            )
        except (TypeError, ValueError):
            abort(400)
        return figure_response(
            payload, output_format,
            "euclidpolish_star_population_calibration",
        )

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
            if population.get("tng_prior"):
                tng_prior = dict(population["tng_prior"])
                tng_prior["configured_prior_arcmin2"] = float(
                    job_config.load().galaxy_density_arcmin2
                )
                visible = dict(tng_prior.get("visible") or {})
                if visible:
                    visible["detection_residual_arcmin2"] = float(
                        visible.get("synthetic_detected_density_arcmin2", 0.0)
                    ) - float(visible.get("real_detected_density_arcmin2", 0.0))
                    visible["actionable"] = False
                    visible["transfer_compatibility"] = {
                        "compatible": False,
                        "source_fingerprints": [],
                        "active_fingerprint": (
                            (transfer_state().get("active") or {}).get("fingerprint")
                        ),
                        "reason": "not actionable—brightness transfer changed",
                    }
                    tng_prior["visible"] = visible
                tng_prior["density_calibration"] = density_state()
                tng_prior["photometric_transfer"] = transfer_state()
                tng_prior.setdefault("historical_incompatible_points", [
                    {
                        "density_arcmin2": 320.0, "job_id": "36490243",
                        "offset_mag": 0.2863918, "magnitude_slope": 0.7454719,
                        "scatter_mag": 0.3924409,
                    },
                    {
                        "density_arcmin2": 281.0,
                        "job_id": "36501765/36503544",
                        "offset_mag": 0.7595169, "magnitude_slope": 0.6969390,
                        "scatter_mag": 0.7986188,
                    },
                ])
                tng_prior["recommendation"] = (
                    "A single regenerated sample reports only a detection "
                    "residual. Run the local joint fit to evaluate the actual "
                    "COSMOS/TNG draw prior through the fitted Euclid brightness "
                    "and completeness model before changing raw density."
                )
                population["tng_prior"] = tng_prior
            comparison["population"] = population
            comparison.pop("population_with_training", None)
        return jsonify({
            "comparison": comparison,
            "availability": availability(),
            "authenticated": euclid_session.is_authenticated(),
            "calibrations": {
                "brightness_transfer": transfer_state(),
                "galaxy_density": density_state(),
                "joint_galaxy": joint_galaxy_state(),
                "stars": star_state(),
                "galaxy_recommendation": galaxy_recommendation_state(),
            },
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
        catalog = euclid_session.catalog()
        if catalog is None:
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
            meta = query_euclid_population(
                ra, dec, radius_arcmin, relogin=catalog.relogin,
            )
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
        catalog = euclid_session.catalog()
        if catalog is None:
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
        if not 1 <= count <= MAX_POPULATION_CONES or not 0 < radius <= 30:
            return jsonify({"ok": False, "error": (
                f"count must be 1–{MAX_POPULATION_CONES} and "
                "radius_arcmin must be in (0, 30]"
            )}), 400

        def run(cap):
            meta = query_euclid_population_multi(
                count=count,
                radius_arcmin=radius,
                relogin=catalog.relogin,
                progress=lambda done, _total, label: cap.tick(
                    done, count + 3, label
                ),
            )
            _fit_and_evaluate_cached_cones(
                cap,
                progress_start=count,
                progress_total=count + 3,
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

    def activation_job(label, action):
        def run(cap):
            cap.tick(0, 1, label)
            result = action()
            cap.tick(1, 1, label)
            return result
        return jsonify({
            "ok": True,
            "job_id": REGISTRY.spawn(label=label, target=run),
        })

    @app.route(
        "/api/population-comparison/activate-transfer", methods=["POST"]
    )
    def api_population_comparison_activate_transfer():
        return activation_job(
            "activate fixed-normalization transfer",
            activate_photometric_transfer,
        )

    @app.route(
        "/api/population-comparison/activate-density", methods=["POST"]
    )
    def api_population_comparison_activate_density():
        return activation_job(
            "activate local density calibration",
            activate_density_candidate,
        )

    @app.route(
        "/api/population-comparison/activate-galaxy-recommendation",
        methods=["POST"],
    )
    def api_population_comparison_activate_galaxy_recommendation():
        return activation_job(
            "activate fitted galaxy generator parameters",
            activate_galaxy_recommendation,
        )

    @app.route(
        "/api/population-comparison/activate-joint-galaxy", methods=["POST"]
    )
    def api_population_comparison_activate_joint_galaxy():
        return activation_job(
            "activate joint analytical TNG population",
            activate_joint_galaxy_candidate,
        )

    @app.route(
        "/api/population-comparison/query-gaia-stars", methods=["POST"]
    )
    def api_population_comparison_query_gaia_stars():
        catalog = euclid_session.catalog()
        if catalog is None:
            return jsonify({"ok": False, "error": (
                "Log in to the Euclid archive on the Catalog page first."
            )}), 400
        if not availability().get("euclid_catalog", {}).get("cached"):
            return jsonify({"ok": False, "error": (
                "Query and cache the Euclid cones first."
            )}), 400

        def run(cap):
            q1 = query_q1_phz_star_counts(
                relogin=catalog.relogin,
                progress=lambda done, total, label: cap.tick(
                    done, total + 1, label
                ),
            )
            q1_bins = len(q1["bins"])
            meta = query_gaia_field_cones(
                progress=lambda done, total, label: cap.tick(
                    q1_bins + done, q1_bins + total + 1, label
                )
            )
            cap.tick(q1_bins + meta["cone_count"],
                     q1_bins + meta["cone_count"] + 1, "fit star prior")
            fit = fit_star_population()
            cap.tick(q1_bins + meta["cone_count"] + 1,
                     q1_bins + meta["cone_count"] + 1, "star prior ready")
            cap.write(
                f"Q1 PHZ expected stars {q1['expected_stars']:.1f}; "
                f"cached {meta['rows']} Gaia DR3 color sources; "
                f"matched {fit['euclid_mapping']['matched_stars']} Euclid stars\n"
            )
            return {"q1_phz_counts": q1, "gaia_colors": meta, "fit": fit}

        return jsonify({
            "ok": True,
            "job_id": REGISTRY.spawn(
                label="population comparison: Q1 PHZ + stellar colors",
                target=run,
            ),
        })

    @app.route(
        "/api/population-comparison/refresh-euclid-multi", methods=["POST"]
    )
    def api_population_comparison_refresh_euclid_multi():
        catalog = euclid_session.catalog()
        if catalog is None:
            return jsonify({"ok": False, "error": (
                "Log in to the Euclid archive on the Catalog page first."
            )}), 400

        def run(cap):
            meta = refresh_cached_euclid_population_multi(
                relogin=catalog.relogin,
                progress=lambda done, total, label: cap.tick(
                    done, total + 3, label
                ),
            )
            _fit_and_evaluate_cached_cones(
                cap,
                progress_start=int(meta["cone_count"]),
                progress_total=int(meta["cone_count"]) + 3,
            )
            cap.write(
                f"refreshed {meta['rows']} unique sources from the same "
                f"{meta['cone_count']} saved cones\n"
            )
            return meta

        return jsonify({
            "ok": True,
            "job_id": REGISTRY.spawn(
                label="population comparison: refresh same Euclid cones",
                target=run,
            ),
        })

    @app.route(
        "/api/population-comparison/activate-star-prior", methods=["POST"]
    )
    def api_population_comparison_activate_star_prior():
        return activation_job("activate stellar population", activate_star_candidate)

    @app.route(
        "/api/population-comparison/run-local-galaxy-calibration",
        methods=["POST"],
    )
    def api_population_comparison_run_local_galaxy_calibration():
        """Fit the shared analytical population from cached local catalogues."""
        def run(cap):
            project_root = Path(__file__).resolve().parents[3]
            cap.tick(0, 1, "fit smooth COSMOS + Euclid population")
            _run_analysis_script(
                project_root, "scripts/fit_cosmos_euclid_counts.py"
            )
            result = read_cosmos_euclid_fit()
            if result is None:
                raise RuntimeError("joint fit completed without an artifact")
            cap.write("Saved all parameter and survey-comparison plots locally.\n")
            cap.write("No TNG catalogue or image was read.\n")
            cap.tick(1, 1, "joint analytical population ready")
            return {"fit": result, "tng_used": False}

        return jsonify({
            "ok": True,
            "job_id": REGISTRY.spawn(
                label="local galaxy population calibration", target=run,
            ),
        })

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
