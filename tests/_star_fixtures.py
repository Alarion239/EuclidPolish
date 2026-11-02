"""A small valid Euclid-only stellar colour model for tests."""

from __future__ import annotations

import numpy as np

from euclid_polish.sky.generation.stellar_sed import build_star_color_model


def fixture_star_color_model(
    *, seed: int = 0, rows: int = 240, mag_bright: float = 12.0,
    mag_faint: float = 25.0, noise_mag: float = 0.02,
) -> dict:
    """``color_model`` section: ``rows`` point sources over VIS 18–23 on a
    straight locus (Y−J and J−H follow VIS−Y), with ``noise_mag`` errors."""
    rng = np.random.default_rng(seed)
    magnitude = rng.uniform(18.0, 23.0, rows)
    vis_y = rng.uniform(0.3, 1.5, rows)
    true = np.column_stack((vis_y, 0.05 + 0.1 * vis_y, 0.02 * vis_y))
    mag_sigma = np.full((rows, 4), noise_mag)
    color = true + rng.normal(0.0, noise_mag * np.sqrt(2.0), (rows, 3))
    return build_star_color_model(
        magnitude, color, mag_sigma, np.full(rows, 0.9),
        mag_bright=mag_bright, mag_faint=mag_faint,
    )
