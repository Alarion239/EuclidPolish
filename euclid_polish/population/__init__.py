"""Analytical stellar and galaxy population models."""

from .joint_galaxy import (
    COSMOS_FIT_MAG_MAX,
    COSMOS_FIT_MAG_MIN,
    COSMOS_FIT_Z_MAX,
    COSMOS_FIT_Z_MIN,
    EuclidResponseFit,
    SchechterEvolutionFit,
    SizeEvolutionFit,
    fit_euclid_response,
    fit_schechter_evolution,
    fit_size_evolution,
    latent_population_cube,
    predict_euclid_histogram,
    read_cosmos_population,
    read_euclid_population,
    tng_draw_population_cube,
)

__all__ = [
    "COSMOS_FIT_MAG_MAX",
    "COSMOS_FIT_MAG_MIN",
    "COSMOS_FIT_Z_MAX",
    "COSMOS_FIT_Z_MIN",
    "EuclidResponseFit",
    "SchechterEvolutionFit",
    "SizeEvolutionFit",
    "fit_euclid_response",
    "fit_schechter_evolution",
    "fit_size_evolution",
    "latent_population_cube",
    "predict_euclid_histogram",
    "read_cosmos_population",
    "read_euclid_population",
    "tng_draw_population_cube",
]
