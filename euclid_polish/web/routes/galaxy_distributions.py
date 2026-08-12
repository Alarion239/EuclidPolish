"""Routes for the dedicated galaxy-distribution workspace."""

import io

from flask import abort, jsonify, request, send_file

from euclid_polish.web import euclid_session
from euclid_polish.web.helpers.galaxy_distributions import (
    build_galaxy_distributions,
    read_galaxy_distributions,
)
from euclid_polish.web.helpers.population_calibration import (
    activate_joint_galaxy_candidate,
    fit_euclid_joint_galaxy_candidate,
    joint_galaxy_state,
)
from euclid_polish.web.helpers.publication_figures import render_population_atlas
from euclid_polish.web.helpers.q1_galaxy_counts import (
    fit_q1_galaxy_aperture_counts,
    query_q1_galaxy_aperture_counts,
    read_q1_galaxy_aperture_counts,
)
from euclid_polish.web.helpers.q1_galaxy_radius_statistics import (
    query_q1_galaxy_radius_statistics,
    read_q1_galaxy_radius_statistics,
)
from euclid_polish.web.helpers.q1_star_counts import (
    query_q1_phz_star_counts,
    read_q1_phz_star_counts,
)
from euclid_polish.web.helpers.q1_stellar_colors import (
    q1_stellar_color_query_count,
    q1_stellar_color_sample_state,
    query_q1_stellar_color_sample,
)
from euclid_polish.web.jobs import REGISTRY


def _q1_counts_state():
    try:
        return read_q1_galaxy_aperture_counts()
    except ValueError:
        return None


def _q1_radius_state():
    try:
        return read_q1_galaxy_radius_statistics()
    except ValueError:
        return None


def _q1_star_state():
    try:
        return read_q1_phz_star_counts()
    except ValueError:
        return None


def register(app):
    @app.route("/view/population-atlas")
    def view_population_atlas():
        """Download the reviewed Euclid brightness-radius fit."""
        output_format = (request.args.get("format") or "png").strip().lower()
        if output_format not in {"png", "pdf", "svg"}:
            abort(400)
        candidate = joint_galaxy_state().get("candidate")
        if not candidate:
            abort(404)
        try:
            dpi = int(request.args.get("dpi", "300"))
            payload = render_population_atlas(
                candidate, output_format=output_format, dpi=dpi,
            )
        except (TypeError, ValueError):
            abort(400)
        mimetype = {
            "png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml",
        }[output_format]
        inline = request.args.get("inline", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        return send_file(
            io.BytesIO(payload), mimetype=mimetype, as_attachment=not inline,
            download_name=f"euclidpolish_population_atlas.{output_format}",
            max_age=0,
        )

    @app.get("/api/galaxy-distributions")
    def api_galaxy_distributions():
        return jsonify({
            **read_galaxy_distributions(),
            "authenticated": euclid_session.is_authenticated(),
            "q1_counts": _q1_counts_state(),
            "q1_radius": _q1_radius_state(),
            "q1_stars": _q1_star_state(),
            "stellar_colors": q1_stellar_color_sample_state(),
            "calibration": joint_galaxy_state(),
        })

    @app.post("/api/galaxy-distributions/query-q1-counts")
    def api_query_q1_galaxy_counts():
        catalog = euclid_session.catalog()
        if catalog is None:
            return jsonify({
                "ok": False,
                "error": "Log in to the Euclid archive on the Catalog page first.",
            }), 400

        def run(cap):
            # One acquisition action owns every population query.  Each
            # helper still checkpoints its own versioned aggregate artifact.
            aperture_total = 560
            radius_total = 170
            star_total = 260
            stellar_color_total = q1_stellar_color_query_count()
            fit_steps = 3
            grand_total = (
                aperture_total + radius_total + star_total
                + stellar_color_total + fit_steps
            )

            def stage(offset, size):
                def report(done, total, label):
                    fraction = 0.0 if total <= 0 else min(max(done / total, 0.0), 1.0)
                    cap.tick(offset + int(round(size * fraction)), grand_total, label)
                return report

            result = query_q1_galaxy_aperture_counts(
                relogin=catalog.relogin,
                progress=stage(0, aperture_total),
            )
            radii = query_q1_galaxy_radius_statistics(
                relogin=catalog.relogin,
                progress=stage(aperture_total, radius_total),
            )
            stars = query_q1_phz_star_counts(
                relogin=catalog.relogin,
                progress=stage(aperture_total + radius_total, star_total),
            )
            color_offset = aperture_total + radius_total + star_total
            stellar_colors = query_q1_stellar_color_sample(
                relogin=catalog.relogin,
                progress=stage(color_offset, stellar_color_total),
            )
            fit_offset = color_offset + stellar_color_total
            cap.tick(fit_offset, grand_total, "fit Q1 VIS 2FWHM straight line")
            brightness_fit = fit_q1_galaxy_aperture_counts()
            cap.tick(fit_offset + 1, grand_total, "fit aggregate Sersic R_e relation")
            joint_fit = fit_euclid_joint_galaxy_candidate()
            cap.tick(fit_offset + 2, grand_total, "rebuild galaxy-distribution plots")
            plots = build_galaxy_distributions()
            cap.tick(grand_total, grand_total, "MER + PHZ populations ready")
            cap.write(
                f"Q1 MER + PHZ galaxy brightness ready from "
                f"VIS {result['bright']:.1f} to {result['faint']:.1f}: "
                f"{result['completed_queries']}/"
                f"{result['total_queries']} aperture-bin checkpoints over "
                f"{result['footprint_area_deg2']:.1f} deg²\n"
            )
            cap.write(
                f"Q1 aggregate Sersic R_e statistics: "
                f"{radii['completed_queries']}/{radii['total_queries']} "
                "brackets; no object rows or random field sampling\n"
            )
            cap.write(
                f"Q1 stellar counts: {stars['expected_stars']:.1f} expected "
                f"PHZ stars over {stars['footprint_area_deg2']:.1f} deg²\n"
            )
            cap.write(
                "Fixed-Q1 Gaia-Euclid colour sample: "
                f"{stellar_colors['euclid']['rows']} Euclid rows and "
                f"{stellar_colors['gaia']['rows']} Gaia rows; "
                "density still comes only from Q1 magnitude brackets\n"
            )
            cap.write(
                f"Joint galaxy candidate {joint_fit['fingerprint'][:12]}… ready; "
                "review the plots before activation\n"
            )
            return {
                "galaxy_counts": result,
                "galaxy_radius": radii,
                "star_counts": stars,
                "stellar_colors": stellar_colors,
                "brightness_fit": brightness_fit,
                "joint_galaxy_fit": joint_fit,
                "plots": {"version": plots["version"]},
            }

        return jsonify({
            "ok": True,
            "job_id": REGISTRY.spawn(
                label=(
                    "population distributions: all MER + PHZ queries "
                    "and galaxy fits"
                ),
                target=run,
            ),
        })

    @app.post("/api/galaxy-distributions/fit-q1-counts")
    def api_fit_q1_galaxy_counts():
        counts = _q1_counts_state()
        if counts is None:
            return jsonify({
                "ok": False,
                "error": "Query and cache Q1 MER + PHZ aperture counts first.",
            }), 400
        fit_ready = bool(counts.get("fit_ready")) or any(
            int(aperture.get("queried_bins") or len(aperture.get("bins", [])))
            >= 4
            for aperture in counts.get("apertures", {}).values()
        )
        if not fit_ready:
            return jsonify({
                "ok": False,
                "error": (
                    "Cache at least four Q1 magnitude bins per aperture "
                    "before fitting. Zero-count bins are allowed."
                ),
            }), 400

        def run(cap):
            cap.tick(0, 1, "fit cached Q1 aperture-count curves")
            result = fit_q1_galaxy_aperture_counts()
            cap.tick(1, 1, "Q1 aperture-count curves fitted")
            cap.write(
                f"fitted {len(result['apertures'])} Q1 aperture curves; "
                "no object catalogue was used\n"
            )
            return result

        return jsonify({
            "ok": True,
            "job_id": REGISTRY.spawn(
                label="galaxy distributions: fit cached Q1 aperture counts",
                target=run,
            ),
        })

    @app.post("/api/galaxy-distributions/build")
    def api_build_galaxy_distributions():
        def run(cap):
            return build_galaxy_distributions(lambda done, total, label: cap.tick(done, total, label))

        return jsonify(
            {
                "ok": True,
                "job_id": REGISTRY.spawn(
                    label="galaxy distributions: build plot data",
                    target=run,
                ),
            }
        )

    @app.post("/api/galaxy-distributions/activate")
    def api_activate_joint_galaxy():
        def run(cap):
            cap.tick(0, 1, "activate Euclid brightness-radius population")
            result = activate_joint_galaxy_candidate()
            cap.tick(1, 1, "Euclid brightness-radius population active")
            return result

        return jsonify({
            "ok": True,
            "job_id": REGISTRY.spawn(
                label="galaxy distributions: activate joint galaxy model",
                target=run,
            ),
        })
