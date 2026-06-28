"""
Euclid catalog package.

The :class:`EuclidCatalog` client owns all live archive operations (auth,
queries, cutout downloads); :class:`CatalogObject` is the queried-object record
it returns and persists.
"""

from euclid_polish.catalog.catalog_object import CatalogObject
from euclid_polish.catalog.client import EuclidAuthError, EuclidCatalog
from euclid_polish.catalog.downloader import DownloadConfig
from euclid_polish.catalog.validator import FitsValidator, angular_separation_arcsec

__all__ = [
    "EuclidCatalog",
    "EuclidAuthError",
    "CatalogObject",
    "FitsValidator",
    "DownloadConfig",
    "angular_separation_arcsec",
]
