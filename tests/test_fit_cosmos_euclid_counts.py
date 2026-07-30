from __future__ import annotations

import numpy as np

from scripts import fit_cosmos_euclid_counts as fit_mod


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
