"""Multi-band sky generation and observation APIs, resolved lazily."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Image": ("euclid_polish.image", "Image"),
    "Cosmos2025Catalog": ("euclid_polish.sky.generation.cosmos2025", "Cosmos2025Catalog"),
    "CosmosCatalog": ("euclid_polish.sky.generation.cosmos2025", "CosmosCatalog"),
    "GalaxyParams": ("euclid_polish.sky.generation.cosmos2025", "GalaxyParams"),
    "open_cosmos2025": ("euclid_polish.sky.generation.cosmos2025", "open_cosmos2025"),
    "LensParams": ("euclid_polish.sky.generation.lens_population", "LensParams"),
    "LensPopulation": ("euclid_polish.sky.generation.lens_population", "LensPopulation"),
    "einstein_radius_sis": ("euclid_polish.sky.generation.lens_population", "einstein_radius_sis"),
    "render_lens_to_canvas": (
        "euclid_polish.sky.generation.lens_population",
        "render_lens_to_canvas",
    ),
    "render_lens_to_multiband_canvas": (
        "euclid_polish.sky.generation.lens_population",
        "render_lens_to_multiband_canvas",
    ),
    "add_sersic_to_bands": ("euclid_polish.sky.generation.profiles", "add_sersic_to_bands"),
    "compute_sersic_stamp": ("euclid_polish.sky.generation.profiles", "compute_sersic_stamp"),
    "draw_bulge_disk": ("euclid_polish.sky.generation.profiles", "draw_bulge_disk"),
    "draw_sersic": ("euclid_polish.sky.generation.profiles", "draw_sersic"),
    "evaluate_sersic_at_coords": (
        "euclid_polish.sky.generation.profiles",
        "evaluate_sersic_at_coords",
    ),
    "sersic_amp_from_flux": ("euclid_polish.sky.generation.profiles", "sersic_amp_from_flux"),
    "sersic_b_n": ("euclid_polish.sky.generation.profiles", "sersic_b_n"),
    "SkySimulator": ("euclid_polish.sky.generation.sky_simulator", "SkySimulator"),
    "SkySimulatorConfig": ("euclid_polish.sky.generation.sky_simulator", "SkySimulatorConfig"),
    "ObservationSimulator": (
        "euclid_polish.sky.observation.observation_simulator",
        "ObservationSimulator",
    ),
    "ObservationSimulatorConfig": (
        "euclid_polish.sky.observation.observation_simulator",
        "ObservationSimulatorConfig",
    ),
    "upsample": ("euclid_polish.sky.observation.resample", "upsample"),
    "lanczos3_upsample": ("euclid_polish.sky.observation.resample", "lanczos3_upsample"),
    "cubic_upsample": ("euclid_polish.sky.observation.resample", "cubic_upsample"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
