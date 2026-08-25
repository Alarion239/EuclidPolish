"""The image data layer: lightweight cubes plus persistent project images.

The cube vocabulary is imported eagerly and has no TensorFlow dependency.
Persistent ``Image``/``ImageSet`` types are resolved lazily so lightweight
scientific utilities can use :class:`ImageCube` without importing the training
stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from euclid_polish.image.cube import (
    AngularGrid,
    CubeLike,
    ImageCube,
    PhysicalGrid,
    PixelUnit,
)

if TYPE_CHECKING:
    from euclid_polish.image.collection import ImageSet
    from euclid_polish.image.core import FitsWCS, Image, Role

_EXPORTS = {
    "FitsWCS": ("euclid_polish.image.core", "FitsWCS"),
    "Image": ("euclid_polish.image.core", "Image"),
    "ImageSet": ("euclid_polish.image.collection", "ImageSet"),
    "Role": ("euclid_polish.image.core", "Role"),
}

__all__ = [
    "AngularGrid",
    "CubeLike",
    "FitsWCS",
    "Image",
    "ImageCube",
    "ImageSet",
    "PhysicalGrid",
    "PixelUnit",
    "Role",
]


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
