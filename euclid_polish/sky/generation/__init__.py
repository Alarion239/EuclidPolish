"""Scene-generation APIs, resolved lazily to keep imports lightweight."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Cosmos2025Catalog": ("cosmos2025", "Cosmos2025Catalog"),
    "CosmosCatalog": ("cosmos2025", "CosmosCatalog"),
    "GalaxyParams": ("cosmos2025", "GalaxyParams"),
    "ensure_prefiltered_catalog": ("cosmos2025", "ensure_prefiltered_catalog"),
    "open_cosmos2025": ("cosmos2025", "open_cosmos2025"),
    "GenerationContext": ("gen_provenance", "GenerationContext"),
    "ShardStampPlan": ("gen_provenance", "ShardStampPlan"),
    "LensParams": ("lens_population", "LensParams"),
    "LensPopulation": ("lens_population", "LensPopulation"),
    "einstein_radius_sis": ("lens_population", "einstein_radius_sis"),
    "render_lens_to_multiband_canvas": ("lens_population", "render_lens_to_multiband_canvas"),
    "sample_lens_geometry": ("lens_population", "sample_lens_geometry"),
    "PhzGalaxyPopulationPrior": (
        "phz_galaxy_prior", "PhzGalaxyPopulationPrior",
    ),
    "build_phz_galaxy_population_payload": (
        "phz_galaxy_prior", "build_phz_galaxy_population_payload",
    ),
    "add_sersic_to_bands": ("profiles", "add_sersic_to_bands"),
    "compute_sersic_stamp": ("profiles", "compute_sersic_stamp"),
    "draw_bulge_disk": ("profiles", "draw_bulge_disk"),
    "draw_sersic": ("profiles", "draw_sersic"),
    "evaluate_sersic_at_coords": ("profiles", "evaluate_sersic_at_coords"),
    "sersic_amp_from_flux": ("profiles", "sersic_amp_from_flux"),
    "sersic_b_n": ("profiles", "sersic_b_n"),
    "TNG_NATIVE_PC_PER_PIXEL": ("redshift_model", "TNG_NATIVE_PC_PER_PIXEL"),
    "angular_diameter_distance": ("redshift_model", "angular_diameter_distance"),
    "band_drift_factors": ("redshift_model", "band_drift_factors"),
    "compactness_factor": ("redshift_model", "compactness_factor"),
    "load_tng_properties": ("redshift_model", "load_tng_properties"),
    "physical_pc_to_arcsec": ("redshift_model", "physical_pc_to_arcsec"),
    "predicted_vis_mag": ("redshift_model", "predicted_vis_mag"),
    "rebin_factor_for_redshift": ("redshift_model", "rebin_factor_for_redshift"),
    "sample_galaxy_redshift": ("redshift_model", "sample_galaxy_redshift"),
    "sample_target_logmass": ("redshift_model", "sample_target_logmass"),
    "sigma_v_from_stellar_mass": ("redshift_model", "sigma_v_from_stellar_mass"),
    "tolman_dimming_factor": ("redshift_model", "tolman_dimming_factor"),
    "SkySimulator": ("sky_simulator", "SkySimulator"),
    "SkySimulatorConfig": ("sky_simulator", "SkySimulatorConfig"),
    "SourceCatalogWriter": ("source_catalog", "SourceCatalogWriter"),
    "read_sources": ("source_catalog", "read_sources"),
    "composite_stamp": ("tng_galaxy", "composite_stamp"),
    "list_tng_galaxies": ("tng_galaxy", "list_tng_galaxies"),
    "sample_tng_stamp": ("tng_galaxy", "sample_tng_stamp"),
    "tng_stamp_at_redshift": ("tng_galaxy", "tng_stamp_at_redshift"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
