"""
Generation sub-package: scene synthesis.
"""

from euclid_polish.sky.generation.cosmos2025 import (
    Cosmos2025Catalog,
    CosmosCatalog,
    GalaxyParams,
    ensure_prefiltered_catalog,
    open_cosmos2025,
)
from euclid_polish.sky.generation.gen_provenance import GenerationContext, ShardStampPlan
from euclid_polish.sky.generation.lens_population import (
    LensParams,
    LensPopulation,
    einstein_radius_sis,
    render_lens_to_multiband_canvas,
    sample_lens_geometry,
)
from euclid_polish.sky.generation.profiles import (
    add_sersic_to_bands,
    compute_sersic_stamp,
    draw_bulge_disk,
    draw_sersic,
    evaluate_sersic_at_coords,
    sersic_amp_from_flux,
    sersic_b_n,
)
from euclid_polish.sky.generation.redshift_model import (
    TNG_NATIVE_PC_PER_PIXEL,
    angular_diameter_distance,
    band_drift_factors,
    compactness_factor,
    load_tng_properties,
    physical_pc_to_arcsec,
    predicted_vis_mag,
    rebin_factor_for_redshift,
    sample_galaxy_redshift,
    sample_target_logmass,
    sigma_v_from_stellar_mass,
    tolman_dimming_factor,
)
from euclid_polish.sky.generation.sky_simulator import SkySimulator, SkySimulatorConfig
from euclid_polish.sky.generation.source_catalog import SourceCatalogWriter, read_sources
from euclid_polish.sky.generation.tng_galaxy import (
    composite_stamp,
    list_tng_galaxies,
    sample_tng_stamp,
    tng_stamp_at_redshift,
)

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
