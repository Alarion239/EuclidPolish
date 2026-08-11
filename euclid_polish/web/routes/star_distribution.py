"""Routes for the Q1 PHZ × Gaia × Euclid stellar workspace."""
from __future__ import annotations

from flask import jsonify

from euclid_polish.web import euclid_session
from euclid_polish.web.helpers.population_calibration import (
    activate_star_candidate,
    star_state,
)
from euclid_polish.web.helpers.population_comparison import availability
from euclid_polish.web.helpers.q1_star_counts import (
    query_q1_phz_star_counts,
    read_q1_phz_star_counts,
)
from euclid_polish.web.helpers.star_population import (
    fit_star_population,
    query_gaia_field_cones,
    star_distribution_payload,
)
from euclid_polish.web.jobs import REGISTRY


def _q1_counts_state():
    try:
        return read_q1_phz_star_counts()
    except ValueError:
        return None


def register(app):
    @app.route("/api/star-distribution")
    def api_star_distribution():
        return jsonify({
            "authenticated": euclid_session.is_authenticated(),
            "availability": availability(),
            "calibration": star_state(),
            "distribution": star_distribution_payload(),
            "q1_counts": _q1_counts_state(),
        })

    @app.route("/api/star-distribution/query-q1-counts", methods=["POST"])
    def api_star_distribution_query_q1_counts():
        catalog = euclid_session.catalog()
        if catalog is None:
            return jsonify({
                "ok": False,
                "error": "Log in to the Euclid archive on the Catalog page first.",
            }), 400

        def run(cap):
            q1 = query_q1_phz_star_counts(
                relogin=catalog.relogin,
                progress=lambda done, total, label: cap.tick(
                    done, total, label
                ),
            )
            cap.write(
                f"Q1 expected point sources "
                f"{q1['expected_point_sources']:.1f}; PHZ expected stars "
                f"{q1['expected_stars']:.1f} over "
                f"{q1['footprint_area_deg2']:.1f} deg² in "
                f"{len(q1['bins'])} VIS bins\n"
            )
            return q1

        return jsonify({
            "ok": True,
            "job_id": REGISTRY.spawn(
                label="star distribution: direct Q1 point-source + PHZ counts",
                target=run,
            ),
        })

    @app.route("/api/star-distribution/query", methods=["POST"])
    def api_star_distribution_query():
        if not availability().get("euclid_catalog", {}).get("cached"):
            return jsonify({
                "ok": False,
                "error": "Query and cache at least one Euclid cone first.",
            }), 400
        try:
            q1 = read_q1_phz_star_counts()
        except ValueError:
            return jsonify({
                "ok": False,
                "error": "Query the direct Q1 PHZ stellar counts first.",
            }), 400

        def run(cap):
            meta = query_gaia_field_cones(
                progress=lambda done, total, label: cap.tick(
                    done, total + 1, label
                )
            )
            cap.tick(meta["cone_count"], meta["cone_count"] + 1,
                     "fit stellar distribution")
            fit = fit_star_population()
            cap.tick(meta["cone_count"] + 1, meta["cone_count"] + 1,
                     "stellar distribution ready")
            cap.write(
                f"Q1 PHZ expected stars {q1['expected_stars']:.1f} over "
                f"{q1['footprint_area_deg2']:.1f} deg²; "
                f"cached {meta['rows']} Gaia DR3 sources for colours; "
                f"matched {fit['euclid_mapping']['matched_stars']} Euclid stars\n"
            )
            return {"q1_phz_counts": q1, "gaia_colors": meta, "fit": fit}

        return jsonify({
            "ok": True,
            "job_id": REGISTRY.spawn(
                label="star distribution: Gaia field-cone color fit",
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
