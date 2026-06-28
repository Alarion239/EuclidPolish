"""The image data layer: the ``Image`` atom and the ``ImageSet`` collection.

These sit at the bottom of the import graph — self-contained serialization,
plotting, crop/rebin and metrics only. Transforms that need an operator live on
the operator (simulator, forward model, trained model, archive).
"""

from euclid_polish.image.collection import ImageSet
from euclid_polish.image.core import FitsWCS, Image, Role

__all__ = ["FitsWCS", "Image", "ImageSet", "Role"]
