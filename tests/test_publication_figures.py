"""Focused checks for the presentation-quality population atlas."""
from __future__ import annotations

from euclid_polish.web.helpers.publication_figures import (
    render_galaxy_distribution_plate,
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
            "generation_law": {
                "x": [14.0, 26.3, 29.0], "density": [0.08, 100.0, 100.0],
            },
            "fit_interval": [19.5, 25.0],
            "sampling_interval": [14.0, 29.0],
            "generation_interval": [14.0, 29.0],
            "break_magnitude": 26.3,
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
                "model_core_low_log10_arcsec": [-0.3, -0.48, -0.66],
                "model_core_high_log10_arcsec": [0.06, -0.12, -0.3],
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
    assert b"Q1 2FWHM fitted main law" in svg
    assert b"Euclid PHZ/MER cleaned circularized" in svg
    assert b"joint-fit" in svg
    assert b"straight conditional mean" in svg
    assert b"generation law: continuous bright bridge" in svg
    assert b"faint tail = 100" in svg
    assert b"COSMOS" not in svg
    assert b"TNG truth" not in svg
    assert b"20&lt;VIS&lt;28" not in svg



def test_galaxy_distribution_plate_uses_current_generated_measurements():
    magnitude_edges = [20.0, 21.0, 22.0, 23.0]
    log_radius_edges = [-1.5, -1.0, -0.5, 0.0]
    density = [
        [2.0, 4.0, 1.0],
        [3.0, 8.0, 2.0],
        [1.0, 5.0, 3.0],
    ]
    curve_x = [20.5, 21.5, 22.5]
    radius_x = [-1.25, -0.75, -0.25]
    payload = {
        "parameters": {
            "magnitude": {"photometry_series": {
                key: {"x": curve_x, "density": values}
                for key, values in {
                    "q1_vis_f2": [2.0, 5.0, 9.0],
                    "synthetic_vis_2fwhm": [1.8, 4.8, 8.4],
                    "generator_vis_f2": [2.1, 5.2, 8.8],
                }.items()
            }},
            "radius": {"radius_series": {
                key: {"x": radius_x, "density": values}
                for key, values in {
                    "euclid_sersic_re": [3.0, 8.0, 2.0],
                    "synthetic_requested_re": [4.0, 7.0, 1.5],
                    "synthetic_clean_half_light": [2.5, 6.0, 2.0],
                    "fit_re": [3.2, 7.7, 1.8],
                    "euclid_sersic_re_shape": [0.4, 1.1, 0.3],
                    "fit_re_q1_weighted_shape": [0.5, 1.0, 0.3],
                    "fit_re_full_generation_shape": [0.7, 0.9, 0.2],
                }.items()
            }},
        },
        "joint_maps": {
            "available": True,
            "magnitude_edges": magnitude_edges,
            "log_radius_edges": log_radius_edges,
            "maps": [
                {
                    "key": key,
                    "label": label,
                    "color": color,
                    "density": density,
                }
                for key, label, color in (
                    ("q1", "Q1 MER + PHZ", "#1267d6"),
                    (
                        "synthetic", "Current generated galaxies",
                        "#0072b2",
                    ),
                    ("model", "Active generation law", "#d55e00"),
                )
            ],
        },
    }

    svg = render_galaxy_distribution_plate(
        payload, output_format="svg", dpi=120,
    )

    assert b"<svg" in svg[:1000]
    assert b"VIS 2FWHM magnitude density" in svg
    assert b"Half-light-radius surface density" in svg
    assert b"Normalized half-light shape" in svg
    assert b"Q1 density + generated/model contours" in svg
    assert b"current generated VIS 2FWHM" in svg
    assert b"generated requested" in svg
    assert b"10/25/50/80/95/99/99.5%" in svg
    assert b"#0072b2" in svg
    assert b"#d55e00" in svg
    assert b"Q1 aggregate" not in svg

    payload["training_included"] = True
    payload["joint_maps"]["maps"][1]["label"] = (
        "Current generated galaxies - exact 2FWHM subset"
    )
    all_split_svg = render_galaxy_distribution_plate(
        payload, output_format="svg", dpi=120,
    )
    assert b"test + validation VIS 2FWHM" in all_split_svg
    assert b"all-catalogue requested" in all_split_svg
    assert b"test + validation clean-image" in all_split_svg


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
