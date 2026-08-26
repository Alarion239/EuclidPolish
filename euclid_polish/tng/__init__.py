"""The IllustrisTNG atlas, donor, and rendering domain.

The package root exposes the small class API used by scene generation while
resolving implementations lazily, so importing :mod:`euclid_polish.tng` does
not eagerly load FITS, OpenCV, SciPy, or plotting dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from euclid_polish.tng.atlas import TNGAtlas, TNGGalaxy
    from euclid_polish.tng.catalog import TNGGalaxyProperties, TNGPropertyCatalog
    from euclid_polish.tng.image import TNGSurfaceBrightnessImage
    from euclid_polish.tng.radius_manifest import TNGRadiusManifest
    from euclid_polish.tng.renderer import TNGRenderer
    from euclid_polish.tng.types import RenderedTNG, TNGRenderTrace, TNGView

_EXPORTS = {
    "RenderedTNG": ("types", "RenderedTNG"),
    "TNGAtlas": ("atlas", "TNGAtlas"),
    "TNGGalaxy": ("atlas", "TNGGalaxy"),
    "TNGGalaxyProperties": ("catalog", "TNGGalaxyProperties"),
    "TNGPropertyCatalog": ("catalog", "TNGPropertyCatalog"),
    "TNGRadiusManifest": ("radius_manifest", "TNGRadiusManifest"),
    "TNGRenderer": ("renderer", "TNGRenderer"),
    "TNGRenderTrace": ("types", "TNGRenderTrace"),
    "TNGSurfaceBrightnessImage": ("image", "TNGSurfaceBrightnessImage"),
    "TNGView": ("types", "TNGView"),
}

__all__ = list(_EXPORTS)  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
