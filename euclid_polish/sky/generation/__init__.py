"""
Generation sub-package: scene synthesis.

Re-exports the generation-side public API from the flat ``euclid_polish.sky``
modules. All ``from euclid_polish.sky.<module> import X`` paths continue to
work unchanged; this namespace just adds ``euclid_polish.sky.generation.X``
as an alternative.
"""

from euclid_polish.sky.sky_simulator import SkySimulator, SkySimulatorConfig
from euclid_polish.sky.cosmos2025 import (
    CosmosCatalog, Cosmos2025Catalog, GalaxyParams,
    open_cosmos2025, ensure_prefiltered_catalog,
)
from euclid_polish.sky.lens_population import (
    LensPopulation, LensParams,
    einstein_radius_sis, render_lens_to_multiband_canvas, sample_lens_geometry,
)
from euclid_polish.sky.profiles import (
    add_sersic_to_bands, compute_sersic_stamp, draw_bulge_disk,
    draw_sersic, evaluate_sersic_at_coords, sersic_b_n, sersic_amp_from_flux,
)
from euclid_polish.sky.tng_galaxy import (
    list_tng_galaxies, sample_tng_stamp, composite_stamp,
    tng_stamp_at_redshift,
)
from euclid_polish.sky.redshift_model import (
    angular_diameter_distance, sample_galaxy_redshift, sample_target_logmass,
    rebin_factor_for_redshift, tolman_dimming_factor, band_drift_factors,
    compactness_factor, physical_pc_to_arcsec, sigma_v_from_stellar_mass,
    load_tng_properties, predicted_vis_mag, TNG_NATIVE_PC_PER_PIXEL,
)
from euclid_polish.sky.source_catalog import SourceCatalogWriter, read_sources
from euclid_polish.sky.gen_provenance import GenerationContext, ShardStampPlan

__all__ = [
    "SkySimulator", "SkySimulatorConfig",
    "CosmosCatalog", "Cosmos2025Catalog", "GalaxyParams",
    "open_cosmos2025", "ensure_prefiltered_catalog",
    "LensPopulation", "LensParams",
    "einstein_radius_sis", "render_lens_to_multiband_canvas", "sample_lens_geometry",
    "add_sersic_to_bands", "compute_sersic_stamp", "draw_bulge_disk",
    "draw_sersic", "evaluate_sersic_at_coords", "sersic_b_n", "sersic_amp_from_flux",
    "list_tng_galaxies", "sample_tng_stamp", "composite_stamp", "tng_stamp_at_redshift",
    "angular_diameter_distance", "sample_galaxy_redshift", "sample_target_logmass",
    "rebin_factor_for_redshift", "tolman_dimming_factor", "band_drift_factors",
    "compactness_factor", "physical_pc_to_arcsec", "sigma_v_from_stellar_mass",
    "load_tng_properties", "predicted_vis_mag", "TNG_NATIVE_PC_PER_PIXEL",
    "SourceCatalogWriter", "read_sources",
    "GenerationContext", "ShardStampPlan",
]
