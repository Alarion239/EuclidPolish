from __future__ import annotations

import csv

import numpy as np

from scripts import fit_cosmos_euclid_counts as fit_mod


def test_euclid_counts_use_fractional_point_like_membership(tmp_path):
    path = tmp_path / "euclid.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "mag_vis", "spurious_prob", "point_like_prob", "type",
        ))
        writer.writeheader()
        writer.writerows([
            {"mag_vis": 20.2, "spurious_prob": 0.0,
             "point_like_prob": 0.25, "type": "star"},
            {"mag_vis": 20.3, "spurious_prob": 0.0,
             "point_like_prob": 0.80, "type": "galaxy"},
            {"mag_vis": 20.4, "spurious_prob": 0.9,
             "point_like_prob": 0.0, "type": "galaxy"},
            {"mag_vis": 21.2, "spurious_prob": 0.0,
             "point_like_prob": "", "type": "unknown"},
        ])

    counts, diagnostics = fit_mod.read_euclid_weighted_counts(path)

    assert counts[0] == 0.95
    assert counts.sum() == 0.95
    assert diagnostics["selected_rows"] == 2
    assert diagnostics["missing_probability_rows"] == 1
    assert diagnostics["classification_weighting"] == (
        "galaxy_weight=1-POINT_LIKE_PROB"
    )


def test_observation_model_preserves_latent_counts_without_selection():
    intrinsic = np.exp((fit_mod.MODEL_GRID - 24.0) * 0.5)
    direct = fit_mod._integrate_grid(intrinsic)
    latent, detected, completeness = fit_mod.observation_model(
        intrinsic,
        population_scale=1.0,
        magnitude_offset=0.0,
        magnitude_slope=1.0,
        scatter_mag=0.02,
        completeness_m50=28.0,
        completeness_width_mag=0.04,
    )

    np.testing.assert_allclose(latent, direct, rtol=0.02)
    np.testing.assert_allclose(detected[:-1], latent[:-1], rtol=0.02)
    assert completeness[0] > 0.99
    assert completeness[-1] < 0.01


def test_fit_recovers_euclid_detection_turnover():
    intrinsic = np.exp((fit_mod.MODEL_GRID - 24.0) * 0.55)
    true = {
        "population_scale": 0.9,
        "magnitude_offset": -0.15,
        "magnitude_slope": 0.92,
        "scatter_mag": 0.15,
        "completeness_m50": 25.2,
        "completeness_width_mag": 0.32,
    }
    _latent, detected, _completeness = fit_mod.observation_model(
        intrinsic,
        **true,
    )
    area = 500.0
    counts = np.rint(detected * area).astype(np.int64)

    fitted, _latent, prediction, _completeness = fit_mod.fit_observation_layer(
        intrinsic,
        counts,
        area,
        fit_population_scale=True,
    )

    np.testing.assert_allclose(prediction, counts / area, rtol=0.05, atol=0.002)
    assert abs(fitted.completeness_m50 - true["completeness_m50"]) < 0.15
    assert abs(
        fitted.completeness_width_mag - true["completeness_width_mag"]
    ) < 0.15
    assert abs(fitted.magnitude_slope - true["magnitude_slope"]) < 0.15
