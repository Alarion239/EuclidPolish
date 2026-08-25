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
    "SkySimulator": ("sky_simulator", "SkySimulator"),
    "SkySimulatorConfig": ("sky_simulator", "SkySimulatorConfig"),
    "SourceCatalogWriter": ("source_catalog", "SourceCatalogWriter"),
    "read_sources": ("source_catalog", "read_sources"),
}

__all__ = list(_EXPORTS)  # pyright: ignore[reportUnsupportedDunderAll]


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
