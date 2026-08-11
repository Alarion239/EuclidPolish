"""Routes for the dedicated galaxy-distribution workspace."""

from flask import jsonify

from euclid_polish.web import euclid_session
from euclid_polish.web.helpers.galaxy_distributions import (
    build_galaxy_distributions,
    read_galaxy_distributions,
)
from euclid_polish.web.helpers.q1_galaxy_counts import (
    fit_q1_galaxy_aperture_counts,
    query_q1_galaxy_aperture_counts,
    read_q1_galaxy_aperture_counts,
)
from euclid_polish.web.jobs import REGISTRY


def _q1_counts_state():
    try:
        return read_q1_galaxy_aperture_counts()
    except ValueError:
        return None


def register(app):
    @app.get("/api/galaxy-distributions")
    def api_galaxy_distributions():
        return jsonify({
            **read_galaxy_distributions(),
            "authenticated": euclid_session.is_authenticated(),
            "q1_counts": _q1_counts_state(),
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
            result = query_q1_galaxy_aperture_counts(
                relogin=catalog.relogin,
                progress=lambda done, total, label: cap.tick(
                    done, total, label,
                ),
            )
            cap.write(
                f"Q1 MER + PHZ galaxy counts ready from "
                f"VIS {result['bright']:.1f} to {result['faint']:.1f}: "
                f"{result['completed_queries']}/"
                f"{result['total_queries']} aperture-bin checkpoints over "
                f"{result['footprint_area_deg2']:.1f} deg²\n"
            )
            return result

        return jsonify({
            "ok": True,
            "job_id": REGISTRY.spawn(
                label=(
                    "galaxy distributions: progressive Q1 MER + PHZ "
                    "aperture counts"
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
                "no cone catalogue was used\n"
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
