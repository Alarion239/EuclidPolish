"""Back-compat shim: the central sky-image type now lives in
:mod:`euclid_polish.image`.

Historically this module defined ``MultiBandSkyImage``. That class has been
renamed to :class:`~euclid_polish.image.core.Image` and moved into the leaf
``euclid_polish.image`` package (self-contained data atom: serialization,
plotting, crop/rebin, metrics — no operator dependencies). The orchestrated
physics that used to hang off it (PSF convolution, band noise) now lives on the
operator classes.

This module re-exports the class under its old name so the many existing
``from euclid_polish.sky.types import MultiBandSkyImage`` call sites keep working
unchanged.
"""

from __future__ import annotations

from euclid_polish.image.core import Image, Role  # noqa: F401

# Back-compat alias — ``MultiBandSkyImage`` is exactly ``Image``.
MultiBandSkyImage = Image

__all__ = ["MultiBandSkyImage", "Image", "Role"]
