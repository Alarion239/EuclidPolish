"""
Euclid Operations module.

This module provides classes for working with Euclid telescope data,
including catalog management, cutout downloading, and PSF extraction.
"""

from euclid_polish.euclid.types import PSF, estimate_fwhm
from euclid_polish.euclid.catalog import StarCatalog
from euclid_polish.euclid.validator import (
    FitsValidator,
    angular_separation_arcsec,
    validate_file_exists,
    validate_directory_exists,
    validate_range,
    validate_positive,
)
from euclid_polish.euclid.psf_extractor import PSFExtractor, PSFExtractionConfig
from euclid_polish.euclid.downloader import EuclidCutoutDownloader, DownloadConfig

__all__ = [
    "PSF",
    "estimate_fwhm",
    "StarCatalog",
    "FitsValidator",
    "PSFExtractor",
    "PSFExtractionConfig",
    "EuclidCutoutDownloader",
    "DownloadConfig",
    "angular_separation_arcsec",
    "validate_file_exists",
    "validate_directory_exists",
    "validate_range",
    "validate_positive",
]
