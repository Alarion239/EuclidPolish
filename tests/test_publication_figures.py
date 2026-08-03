"""Focused checks for the presentation-quality population atlas."""
from __future__ import annotations

from euclid_polish.web.helpers.publication_figures import (
    render_population_atlas,
    render_star_population_calibration,
)


def _density(x, observed=None, model=None):
    payload = {"x": x}
    if observed is not None:
        payload["observed"] = observed
    if model is not None:
        payload["model"] = model
    if observed is None and model is None:
        payload["density"] = [1.0, 2.0, 1.0]
    return payload


def _fit():
    observed = [0.5, None, 1.5]  # missing stays missing; it must not become zero
    model = [0.6, 1.1, 1.4]
    return {
        "diagnostics": {
            "magnitude_counts": {
                "cosmos": _density([20.0, 21.0, 22.0], observed, model),
                "euclid": _density([20.0, 21.0, 22.0], observed, model),
            },
            "redshift": _density([0.25, 0.75, 1.25], observed, model),
            "angular_radius": {
                "cosmos": _density([-1.5, -1.0, -0.5], observed, model),
                "euclid": _density([-1.5, -1.0, -0.5], observed, model),
            },
            "tng_draw": {
                "full": {
                    "magnitude": _density([20.0, 21.0, 22.0]),
                    "redshift": _density([0.25, 0.75, 1.25]),
                    "angular_radius": _density([-1.5, -1.0, -0.5]),
                },
                "comparison_window": {
                    "magnitude": _density([20.0, 21.0, 22.0]),
                    "redshift": _density([0.25, 0.75, 1.25]),
                    "angular_radius": _density([-1.5, -1.0, -0.5]),
                },
            },
        }
    }


def test_population_atlas_exports_raster_and_vector_formats():
    fit = _fit()
    png = render_population_atlas(fit, output_format="png", dpi=120)
    pdf = render_population_atlas(fit, output_format="pdf", dpi=120)
    svg = render_population_atlas(fit, output_format="svg", dpi=120)

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf.startswith(b"%PDF")
    assert b"<svg" in svg[:1000]
    assert b"COSMOS constrains" not in svg
    assert b"Angular radii use" not in svg
    assert b"Missing bins are" not in svg
    assert b"Galaxy population calibration" not in svg
    assert b"COSMOS data" in svg
    assert b"Euclid data" in svg
    assert b"TNG target" in svg
    assert b"TNG truth" not in svg
    assert b"20&lt;VIS&lt;28" not in svg


def _star_calibration():
    color = {
        "x": [-0.2, 0.0, 0.2],
        "fitted": [0.3, 1.0, 0.4],
        "observed": [0.2, 0.9, 0.5],
        "posterior_predictive": [0.25, 0.85, 0.45],
        "dirty_observed": [0.15, 0.8, 0.55],
    }
    return {
        "coverage": {"high_quality_matched_rows": 120},
        "fit": {"selection": {"gaia_bright_g_max": 20.5}},
        "population": {"density_arcmin2": 4.5},
        "diagnostics": {
            "star_density_per_cone": {
                "x": [1, 2, 3],
                "observed": [4.0, 5.0, 4.5],
                "fitted": [4.5, 4.5, 4.5],
                "unit": "point sources / arcmin²",
            },
            "parameters": {
                "vis_y": color,
                "y_j": color,
                "j_h": color,
            },
        },
    }


def test_star_population_calibration_exports_raster_and_vector_formats():
    calibration = _star_calibration()
    png = render_star_population_calibration(calibration, dpi=120)
    pdf = render_star_population_calibration(
        calibration, output_format="pdf", dpi=120,
    )
    svg = render_star_population_calibration(
        calibration, output_format="svg", dpi=120,
    )

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf.startswith(b"%PDF")
    assert b"<svg" in svg[:1000]
    assert b"Gaia anchors" not in svg
    assert b"constrain faint counts" not in svg
    assert b"negative-flux rows" not in svg
    assert b"Fitted true-colour population" in svg
    assert b"Estimated true colours of observed stars" in svg
    assert b"Estimated colours with simulated Euclid noise" in svg
    assert b"Raw Euclid catalogue colours" in svg
