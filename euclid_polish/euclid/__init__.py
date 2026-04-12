"""
Euclid Operations module.

This module provides classes for working with Euclid telescope data,
including catalog management, cutout downloading, and PSF extraction.
"""

from euclid_polish.euclid.catalog import StarCatalog
from euclid_polish.euclid.validator import FitsValidator, validate_file_exists, validate_directory_exists, validate_positive
from euclid_polish.euclid.psf_extractor import PSFExtractor, PSFExtractionConfig
from euclid_polish.euclid.downloader import EuclidCutoutDownloader, DownloadConfig

__all__ = [
    "StarCatalog",
    "FitsValidator",
    "PSFExtractor",
    "PSFExtractionConfig",
    "EuclidCutoutDownloader",
    "DownloadConfig",
    "validate_file_exists",
    "validate_directory_exists",
    "validate_positive",
]
