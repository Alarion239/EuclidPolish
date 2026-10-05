"""EmpiricalStellarPrior.from_payload keeps its two covariances apart.

``residual_covariance`` is the 4×4 Euclid-band residual of the Gaia→Euclid
mapping; the colour model's ``intrinsic_color_covariance`` is the 3×3 latent
locus scatter. Each must land in its own field.
"""

from __future__ import annotations

import numpy as np

from euclid_polish.population.magnitude_law import StraightMagnitudeLaw
from euclid_polish.sky.generation.stellar_sed import EmpiricalStellarPrior

_RESIDUAL = np.diag([0.01, 0.02, 0.03, 0.04])
_INTRINSIC = np.eye(3) * 0.05


def _payload() -> dict:
    slope, bright, faint, density = float(np.log10(3.0)), 20.0, 22.0, 1.0
    beta = slope * np.log(10.0)
    integral = (np.exp(beta * faint) - np.exp(beta * bright)) / beta
    law = StraightMagnitudeLaw(
        slope=slope, intercept=float(np.log10(density / integral)),
        mag_bright=bright, mag_faint=faint, fit_bright=bright, fit_faint=faint,
        covariance=((1.0e-4, 0.0), (0.0, 1.0e-3)), r_squared=1.0,
        rms_log10_density=0.0, source="fixture",
    )
    return {
        "gaia": {"bp_rp_quantiles": [0.5, 1.0],
                 "temperature_quantiles_k": [4000.0, 6000.0]},
        "euclid_mapping": {
            "g_to_band_offset_coefficients": {
                key: [0.0, 0.0, 0.0]
                for key in ("mag_vis", "mag_y_e", "mag_j_e", "mag_h_e")
            },
            "residual_covariance": _RESIDUAL.tolist(),
        },
        "population": {"magnitude_distribution": law.to_payload()},
        "color_model": {
            "kind": "gaia_euclid_latent_locus_v1",
            "bp_rp_edges": [0.0, 0.75, 1.5],
            "bp_rp_nodes": [0.4, 1.1],
            "temperature_nodes_k": [6500.0, 4500.0],
            "locus_colors": [[0.2, 0.1, 0.05], [0.8, 0.3, 0.1]],
            "intrinsic_color_covariance": _INTRINSIC.tolist(),
            "magnitude_edges": [20.0, 21.0, 22.0],
            "magnitude_node_weights": [[0.5, 0.5], [0.5, 0.5]],
        },
    }


def test_residual_covariance_is_the_4x4_mapping_residual():
    prior = EmpiricalStellarPrior.from_payload(_payload())
    assert prior.residual_covariance.shape == (4, 4)
    np.testing.assert_allclose(prior.residual_covariance, _RESIDUAL)


def test_intrinsic_colour_covariance_stays_in_the_colour_model():
    prior = EmpiricalStellarPrior.from_payload(_payload())
    assert prior.color_model is not None
    intrinsic = prior.color_model["intrinsic_color_covariance"]
    assert intrinsic.shape == (3, 3)
    np.testing.assert_allclose(intrinsic, _INTRINSIC)
