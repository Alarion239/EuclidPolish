"""
Central configuration for EuclidPolish.

This module provides a single source of truth for all configuration values,
eliminating magic strings and numbers scattered throughout the codebase.
"""


class Config:
    """Configuration constants for EuclidPolish."""

    # Directory and file constants
    DATA_DIR = "./data"
    DEFAULT_OUTPUT_DIR = "./data/euclid_stars"
    CLEAN_DATA_DIR = "./data/clean_data"
    DIRTY_DATA_DIR = "./data/dirty_data"
    EUCLID_PSF_DIR = "./data/euclid_psf"
    CATALOG_FILE = "stars.json"
    CUTOUTS_SUBDIR = "cutouts"

    # Default values for command-line arguments
    DEFAULT_CUTOUT_SIZE = 256
    DEFAULT_CLIP_PERCENTILE = 99.5
    DEFAULT_MAGNITUDE_LIMIT = 20.0
    DEFAULT_RADIUS = 0.5
    DEFAULT_NUM_STARS = 5

    # Coordinate ranges
    RA_MIN = 265.0
    RA_MAX = 275.0
    DEC_MIN = 62.0
    DEC_MAX = 70.0

    # Visual output constants
    SUCCESS_PREFIX = "✓"
    ERROR_PREFIX = "✗"
    PENDING_PREFIX = "⏳"
    CORRUPTED_PREFIX = "🔴"
    FAILED_PREFIX = "❌"
    INFO_PREFIX = "📊"

    # Header formatting
    HEADER_WIDTH = 60

    # PSF extraction defaults
    DEFAULT_PSF_SIZE = 255
    DEFAULT_PSF_FWHM = 3.0
    DEFAULT_PSF_THRESHOLD = 50.0
    DEFAULT_PSF_MAX_ITERS = 10
    DEFAULT_PSF_ACCURACY = 0.001
