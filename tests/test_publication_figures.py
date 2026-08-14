"""Focused checks for the presentation-quality population atlas."""
from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.population.euclid_galaxy_prior import (
    ConditionalRadiusLaw,
    joint_density_grid,
)
from euclid_polish.population.magnitude_law import (
    FaintCappedMagnitudeLaw,
    StraightMagnitudeLaw,
)
from euclid_polish.web.helpers.publication_figures import (
    render_population_atlas,
    render_population_fit_comparison,
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
    assert b"Q1 2FWHM fitted middle law" in svg
    assert b"Euclid PHZ/MER measured" in svg
    assert b"joint-fit" in svg
    assert b"broken conditional mean" in svg
    assert b"generation law: straight then flat" in svg
    assert b"faint tail = 100" in svg
    assert b"COSMOS" not in svg
    assert b"TNG truth" not in svg
    assert b"20&lt;VIS&lt;28" not in svg


def _comparison_calibration(
    *, fingerprint: str, tail_fraction: float, radius_intercept: float,
    magnitude_intercept: float,
    kind: str = "euclid_vis2fwhm_sersic_re_joint",
):
    straight = StraightMagnitudeLaw(
        slope=0.2,
        intercept=magnitude_intercept,
        mag_bright=14.0,
        mag_faint=29.0,
        fit_bright=19.5,
        fit_faint=25.0,
        covariance=((1e-4, 0.0), (0.0, 1e-3)),
        r_squared=0.99,
        rms_log10_density=0.03,
        source="fixture",
    )
    magnitude = FaintCappedMagnitudeLaw(
        straight_law=straight,
        density_cap_arcmin2_mag=100.0,
    )
    radius = ConditionalRadiusLaw(
        version=3,
        pivot_mag=23.0,
        intercept_log10_arcsec=radius_intercept,
        slope_log10_arcsec_per_mag=-0.07,
        scatter_dex=0.18,
        log_radius_min=float(np.log10(0.03)),
        log_radius_max=float(np.log10(10.0)),
        fitted_rows=1000,
        clipped_rows=0,
        weighted_rows=800.0,
        residual_rms_dex=0.08,
        r_squared=0.9,
        covariance=((1e-4, 0.0), (0.0, 1e-5)),
        selection="fixture",
        bright_intercept_log10_arcsec=-0.9,
        break_magnitude=18.0,
        tail_fraction=tail_fraction,
        tail_distribution="uniform_log_radius",
        fit_min_selected_per_magnitude_bin=20,
        fit_effective_weight_cap=1000.0,
        fit_faint_magnitude=25.5,
        tail_taper_start_magnitude=25.5,
        tail_taper_end_magnitude=27.0,
    )
    observed_x = np.asarray((15.0, 19.0, 23.0, 27.0))
    return {
        "kind": kind,
        "fingerprint": fingerprint,
        "magnitude_law": magnitude.to_payload(),
        "radius_law": radius.to_payload(),
        "magnitude_plot": {
            "observed": {
                "x": observed_x.tolist(),
                "density": magnitude.density(observed_x).tolist(),
            },
        },
    }


def _radius_aggregate(calibration):
    magnitude_edges = np.linspace(14.0, 28.0, 15)
    radius_edges = np.geomspace(0.03, 10.0, 9)
    magnitude = FaintCappedMagnitudeLaw.from_payload(
        calibration["magnitude_law"],
    )
    radius = ConditionalRadiusLaw.from_payload(calibration["radius_law"])
    grid = joint_density_grid(
        magnitude,
        radius,
        magnitude_edges=magnitude_edges,
        log_radius_edges=np.log10(radius_edges),
    )
    area_arcmin2 = 100.0
    expected = grid["density"] * area_arcmin2
    return {
        "magnitude_edges": magnitude_edges.tolist(),
        "radius_edges_arcsec": radius_edges.tolist(),
        "footprint_area_arcmin2": area_arcmin2,
        "joint_bins": [
            {
                "magnitude_bin": magnitude_index,
                "radius_bin": radius_index,
                "expected_radii": float(expected[magnitude_index, radius_index]),
            }
            for magnitude_index in range(expected.shape[0])
            for radius_index in range(expected.shape[1])
        ],
    }


def test_population_fit_comparison_exports_four_panel_raster_and_vector_plate():
    previous = _comparison_calibration(
        fingerprint="a" * 64,
        tail_fraction=0.15,
        radius_intercept=-0.30,
        magnitude_intercept=-3.0,
    )
    candidate = _comparison_calibration(
        fingerprint="b" * 64,
        tail_fraction=0.02,
        radius_intercept=-0.48,
        magnitude_intercept=-3.1,
        kind="euclid_vis2fwhm_circularized_sersic_re_joint",
    )
    aggregate = _radius_aggregate(candidate)

    png = render_population_fit_comparison(
        previous, candidate, aggregate, output_format="png", dpi=120,
    )
    pdf = render_population_fit_comparison(
        previous, candidate, aggregate, output_format="pdf", dpi=120,
    )
    svg = render_population_fit_comparison(
        previous, candidate, aggregate, output_format="svg", dpi=120,
    )

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf.startswith(b"%PDF")
    assert b"<svg" in svg[:1000]
    assert b"Galaxy population fit" in svg
    assert b"VIS 2FWHM magnitude density" in svg
    assert b"Normalized half-light-radius marginal shape" in svg
    assert b"Conditional S\xc3\xa9rsic size" in svg
    assert b"Q1 circularized density" in svg
    assert b"Previous generation law" in svg
    assert b"Candidate generation law" in svg
    assert b"Previous major-axis model shape" in svg
    assert b"Candidate circularized model shape" in svg
    assert b"Q1 circularized shape" in svg
    assert b"Q1 morphology subset is incomplete" in svg
    assert b"no field regeneration" in svg


def test_population_fit_comparison_rejects_malformed_aggregate():
    previous = _comparison_calibration(
        fingerprint="a" * 64,
        tail_fraction=0.15,
        radius_intercept=-0.30,
        magnitude_intercept=-3.0,
    )
    candidate = _comparison_calibration(
        fingerprint="b" * 64,
        tail_fraction=0.02,
        radius_intercept=-0.48,
        magnitude_intercept=-3.1,
    )

    with pytest.raises(ValueError, match="Q1 radius aggregate"):
        render_population_fit_comparison(previous, candidate, {})


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
