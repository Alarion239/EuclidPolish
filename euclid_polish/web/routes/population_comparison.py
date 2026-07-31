"""Routes for the field-statistics and population-comparison workspace."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from flask import jsonify, request

from euclid_polish.web import euclid_session, fasrc_config, fasrc_fetcher, job_config
from euclid_polish.web.helpers.paths import _sky_records_remote_dir
from euclid_polish.web.helpers.population_calibration import (
    activate_density_candidate,
    activate_photometric_transfer,
    activate_star_candidate,
    density_calibration_path,
    density_state,
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
    refresh_population_comparison,
)
from euclid_polish.web.helpers.star_population import (
    fit_star_population,
    query_gaia_same_cones,
)
from euclid_polish.web.jobs import REGISTRY
from euclid_polish.web.remote import ensure_ssh_connected


def _run_analysis_script(project_root: Path, script: str) -> None:
    """Run one local analysis script and preserve its useful failure output."""
    try:
        subprocess.run(
            [sys.executable, script],
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
    cap, *, progress_start: int = 0, progress_total: int = 3,
) -> dict:
    """Rebuild current truth, fit cached cones, and refresh evaluations."""
    project_root = Path(__file__).resolve().parents[3]
    cap.tick(progress_start, progress_total, "rebuild current synthetic truth")
    _run_analysis_script(
        project_root, "scripts/fit_tng_vis_counts.py"
    )
    cap.tick(progress_start + 1, progress_total, "fit COSMOS observation layer")
    _run_analysis_script(
        project_root, "scripts/fit_cosmos_euclid_counts.py"
    )
    fit_payload = read_cosmos_euclid_fit()
    if fit_payload is None:
        raise RuntimeError("fit completed without a readable fit artifact")

    latent_estimate = (
        fit_payload.get("euclid_latent_density_estimate")
        or fit_payload.get("generator_density_recommendation")
        or {}
    )
    use_local_normalization = bool(
        latent_estimate.get(
            "use_local_normalization",
            latent_estimate.get("apply_to_config", False),
        )
    )
    latent_density = None
    if use_local_normalization:
        latent_density = float(latent_estimate["density_arcmin2"])
        cap.write(
            "Euclid-inferred latent density "
            f"{latent_density:.2f} / arcmin² "
            "(completeness-model estimate, not a generator setting)\n"
        )

    cap.tick(
        progress_start + 2, progress_total,
        "refresh field-statistics evaluations",
    )
    refreshed = refresh_population_comparison()
    cap.tick(progress_start + 3, progress_total, "fit and evaluations ready")

    selected_fit = (
        fit_payload.get("local_normalization_sensitivity_fit")
        if use_local_normalization
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
        "euclid_latent_density_arcmin2": latent_density,
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
                tng_prior["pilot_grid_arcmin2"] = [240, 280, 320, 360, 400]
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
                    "residual. Run the matched-seed sweep with one active "
                    "fixed-normalization brightness transfer before changing "
                    "the raw draw density."
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
                "stars": star_state(),
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
            "activate matched density calibration",
            activate_density_candidate,
        )

    @app.route(
        "/api/population-comparison/query-gaia-stars", methods=["POST"]
    )
    def api_population_comparison_query_gaia_stars():
        if not availability().get("euclid_catalog", {}).get("cached"):
            return jsonify({"ok": False, "error": (
                "Query and cache the Euclid cones first."
            )}), 400

        def run(cap):
            meta = query_gaia_same_cones(
                progress=lambda done, total, label: cap.tick(
                    done, total + 1, label
                )
            )
            cap.tick(meta["cone_count"], meta["cone_count"] + 1, "fit star prior")
            fit = fit_star_population()
            cap.tick(meta["cone_count"] + 1, meta["cone_count"] + 1, "star prior ready")
            cap.write(
                f"cached {meta['rows']} Gaia DR3 sources; "
                f"matched {fit['euclid_mapping']['matched_stars']} Euclid stars\n"
            )
            return {"gaia": meta, "fit": fit}

        return jsonify({
            "ok": True,
            "job_id": REGISTRY.spawn(
                label="population comparison: Gaia + stellar prior",
                target=run,
            ),
        })

    @app.route(
        "/api/population-comparison/activate-star-prior", methods=["POST"]
    )
    def api_population_comparison_activate_star_prior():
        return activation_job("activate stellar population", activate_star_candidate)

    @app.route(
        "/api/population-comparison/sync-density-calibration", methods=["POST"]
    )
    def api_population_comparison_sync_density_calibration():
        remote = (
            f"{fasrc_config.load().data_dir}/population_comparison/"
            "calibrations/tng_density_calibration.json"
        )

        def run(cap):
            cap.tick(0, 2, "connect to FASRC")
            ensure_ssh_connected()
            cap.tick(1, 2, "matched density calibration")
            result = fasrc_fetcher.fetch_one_file(remote, force=True)
            if not result.ok or not result.local_path:
                raise RuntimeError(result.error or "density calibration sync failed")
            target = density_calibration_path()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result.local_path, target)
            state = density_state()
            cap.tick(2, 2, "matched density calibration")
            return state

        return jsonify({
            "ok": True,
            "job_id": REGISTRY.spawn(
                label="sync matched density calibration", target=run,
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
