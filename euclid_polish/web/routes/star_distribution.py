"""Routes for the Q1 PHZ × Gaia × Euclid stellar workspace."""
from __future__ import annotations

import io

from flask import abort, jsonify, request, send_file

from euclid_polish.web import euclid_session
from euclid_polish.web.helpers.population_calibration import (
    activate_star_candidate,
    star_state,
)
from euclid_polish.web.helpers.publication_figures import (
    render_star_population_calibration,
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
from euclid_polish.web.helpers.star_population import (
    fit_star_population,
    star_distribution_payload,
)
from euclid_polish.web.jobs import REGISTRY


def _q1_counts_state():
    try:
        return read_q1_phz_star_counts()
    except ValueError:
        return None


def register(app):
    @app.route("/view/star-population-calibration")
    def view_star_population_calibration():
        """Render the reviewed Gaia-Euclid stellar-prior diagnostics."""
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
        mimetype = {
            "png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml",
        }[output_format]
        inline = request.args.get("inline", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        return send_file(
            io.BytesIO(payload), mimetype=mimetype, as_attachment=not inline,
            download_name=(
                f"euclidpolish_star_population_calibration.{output_format}"
            ),
            max_age=0,
        )

    @app.route("/api/star-distribution")
    def api_star_distribution():
        return jsonify({
            "authenticated": euclid_session.is_authenticated(),
            "color_sample": q1_stellar_color_sample_state(),
            "calibration": star_state(),
            "distribution": star_distribution_payload(),
            "q1_counts": _q1_counts_state(),
        })

    @app.post("/api/star-distribution/query")
    def api_star_distribution_query():
        catalog = euclid_session.catalog()
        if catalog is None:
            return jsonify({
                "ok": False,
                "error": "Log in to the Euclid archive on the Catalog page first.",
            }), 400

        def run(cap):
            count_total = 260
            color_total = q1_stellar_color_query_count()
            grand_total = count_total + color_total

            def stage(offset, size):
                def report(done, total, label):
                    fraction = (
                        0.0 if total <= 0
                        else min(max(done / total, 0.0), 1.0)
                    )
                    cap.tick(
                        offset + int(round(size * fraction)),
                        grand_total,
                        label,
                    )
                return report

            counts = query_q1_phz_star_counts(
                relogin=catalog.relogin,
                progress=stage(0, count_total),
            )
            colors = query_q1_stellar_color_sample(
                relogin=catalog.relogin,
                progress=stage(count_total, color_total),
            )
            cap.tick(grand_total, grand_total, "stellar query caches ready")
            cap.write(
                f"Q1 stellar counts: {counts['expected_stars']:.1f} expected "
                f"PHZ stars over {counts['footprint_area_deg2']:.1f} deg²\n"
            )
            cap.write(
                "Fixed-Q1 Gaia-Euclid colour sample: "
                f"{colors['euclid']['rows']} Euclid rows and "
                f"{colors['gaia']['rows']} Gaia rows; "
                "density still comes only from Q1 magnitude brackets\n"
            )
            return {"q1_counts": counts, "color_sample": colors}

        return jsonify({
            "ok": True,
            "job_id": REGISTRY.spawn(
                label="star distribution: MER + PHZ and Gaia queries",
                target=run,
            ),
        })

    @app.route("/api/star-distribution/fit", methods=["POST"])
    def api_star_distribution_fit():
        if not q1_stellar_color_sample_state().get("cached"):
            return jsonify({
                "ok": False,
                "error": (
                    "No current fixed-Q1 Euclid-Gaia colour sample is cached. "
                    "Query the stellar MER + PHZ and Gaia data on this page first."
                ),
            }), 400
        try:
            q1 = read_q1_phz_star_counts()
        except ValueError:
            return jsonify({
                "ok": False,
                "error": "Query the direct Q1 PHZ stellar counts first.",
            }), 400

        def run(cap):
            cap.tick(0, 1, "fit stellar counts and colours from cached data")
            fit = fit_star_population()
            cap.tick(1, 1, "stellar distribution ready")
            cap.write(
                f"Q1 PHZ expected stars {q1['expected_stars']:.1f} over "
                f"{q1['footprint_area_deg2']:.1f} deg²; "
                f"matched {fit['euclid_mapping']['matched_stars']} cached "
                "Euclid-Gaia stars for colours only\n"
            )
            return {"q1_phz_counts": q1, "fit": fit}

        return jsonify({
            "ok": True,
            "job_id": REGISTRY.spawn(
                label="star distribution: fit cached stellar prior",
                target=run,
            ),
        })

    @app.route("/api/star-distribution/activate", methods=["POST"])
    def api_star_distribution_activate():
        def run(cap):
            cap.tick(0, 1, "activate stellar population")
            result = activate_star_candidate()
            cap.tick(1, 1, "stellar population active")
            return result

        return jsonify({
            "ok": True,
            "job_id": REGISTRY.spawn(
                label="star distribution: activate stellar population",
                target=run,
            ),
        })
