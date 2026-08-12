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


def _galaxy_calibration():
    return {
        "magnitude_plot": {
            "observed": {
                "x": [14.5, 20.5, 27.5], "density": [0.1, 2.0, 50.0],
            },
            "law": {
                "x": [14.0, 21.5, 29.0], "density": [0.08, 3.0, 100.0],
            },
            "fit_interval": [19.5, 25.0],
            "sampling_interval": [14.0, 29.0],
            "extrapolated_interval": [28.0, 29.0],
        },
        "plots": {
            "radius": {
                "x": [-1.5, -1.0, -0.5],
                "observed_density": [0.5, None, 1.5],
                "density": [0.6, 1.1, 1.4],
            },
            "conditional_radius": {
                "magnitude": [16.0, 21.0, 26.0],
                "observed_mean_log10_arcsec": [-0.1, None, -0.5],
                "model_mean_log10_arcsec": [-0.12, -0.3, -0.48],
                "model_low_log10_arcsec": [-0.3, -0.48, -0.66],
                "model_high_log10_arcsec": [0.06, -0.12, -0.3],
            },
        }
    }


def test_population_atlas_exports_raster_and_vector_formats():
    calibration = _galaxy_calibration()
    png = render_population_atlas(
        calibration, output_format="png", dpi=120,
    )
    pdf = render_population_atlas(
        calibration, output_format="pdf", dpi=120,
    )
    svg = render_population_atlas(
        calibration, output_format="svg", dpi=120,
    )

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf.startswith(b"%PDF")
    assert b"<svg" in svg[:1000]
    assert b"COSMOS constrains" not in svg
    assert b"Angular radii use" not in svg
    assert b"Missing bins are" not in svg
    assert b"Galaxy population calibration" not in svg
    assert b"Q1 MER + PHZ 2FWHM counts" in svg
    assert b"Q1 2FWHM straight law" in svg
    assert b"Euclid PHZ/MER measured" in svg
    assert b"joint-fit" in svg
    assert b"joint conditional mean" in svg
    assert b"COSMOS" not in svg
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
            "stellar_density_by_magnitude": {
                "x": [12.0, 18.5, 25.0],
                "gaia_x": [12.25, 18.75, 24.75],
                "observed": [0.02, 0.2, 2.0],
                "fitted": [0.01, 0.1, 1.0],
                "gaia_observed": [0.03, 0.3, 3.0],
                "gaia_fitted": [0.02, 0.2, 2.0],
                "fit_ranges": {"q1": [18.0, 23.0], "gaia": [12.0, 18.0]},
                "unit": "stars / arcmin² / mag",
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
    assert b"Q1-normalized straight law" in svg
    assert b"native Gaia G" in svg
