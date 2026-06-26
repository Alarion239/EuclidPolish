"""
Euclid Operations module.

Classes for working with Euclid telescope data: catalog querying and cutout
downloading.
"""

from euclid_polish.catalog.star_catalog import StarCatalog
from euclid_polish.catalog.validator import (
    FitsValidator,
    angular_separation_arcsec,
    validate_file_exists,
    validate_directory_exists,
    validate_range,
    validate_positive,
)
from euclid_polish.catalog.downloader import EuclidCutoutDownloader, DownloadConfig
from euclid_polish.catalog import auth

__all__ = [
    "StarCatalog",
    "FitsValidator",
    "EuclidCutoutDownloader",
    "DownloadConfig",
    "angular_separation_arcsec",
    "validate_file_exists",
    "validate_directory_exists",
    "validate_range",
    "validate_positive",
    "auth",
]
