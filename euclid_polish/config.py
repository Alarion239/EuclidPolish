"""
Central configuration for EuclidPolish.

This module provides a single source of truth for all configuration values,
eliminating magic strings and numbers scattered throughout the codebase.
"""


class Config:
    """Configuration constants for EuclidPolish."""

    # Directory and file constants
    DATA_DIR = "./data"
    DEFAULT_COSMOS_CATALOG_DIR = "./data/COSMOS"
    DEFAULT_OUTPUT_DIR = "./data/euclid_stars"
    CLEAN_DATA_DIR = "./data/clean_data"
    DIRTY_DATA_DIR = "./data/dirty_data"
    EUCLID_PSF_DIR = "./data/euclid_psf"
    CATALOG_FILE = "stars.json"
    CUTOUTS_SUBDIR = "cutouts"

    # Visualization output
    VIS_DIR              = "./data/vis"
    VIS_CUTOUTS_DIR      = "./data/vis/cutouts"
    VIS_PSF_DIR          = "./data/vis/psf"
    VIS_CLEAN_DIR        = "./data/vis/clean"
    VIS_DIRTY_DIR        = "./data/vis/dirty"

    # TFRecord storage
    RECORDS_DIR          = "./data/images/records"

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

    # Sky generation defaults
    DEFAULT_IMAGE_SIZE           = 256
    DEFAULT_PIXEL_SCALE          = 0.05     # arcsec / pixel
    DEFAULT_GAL_DENSITY_ARCMIN2  = 40.0
    DEFAULT_STAR_DENSITY_ARCMIN2 = 2.0
    DEFAULT_NIMAGES              = 100

    # VIS instrument
    DEFAULT_VIS_ZEROPOINT        = 26.2     # mag → flux conversion
    VIS_PIXEL_SCALE_ARCSEC       = 0.10     # native Euclid VIS pixel scale (arcsec/pixel)

    # PSF convolution defaults
    DEFAULT_REBIN_FACTOR         = 2
    DEFAULT_ADD_NOISE            = True
    DEFAULT_NOISE_STD            = 5.0
    DEFAULT_NORMALIZE            = True
    DEFAULT_NBIT                 = 16

    # Star magnitude distribution (probability thresholds and ranges)
    STAR_MAG_PROB_FAINT          = 0.70     # below → faint bin
    STAR_MAG_PROB_MID            = 0.95     # below → mid bin
    STAR_MAG_FAINT_BASE          = 22.0
    STAR_MAG_FAINT_RANGE         = 3.0
    STAR_MAG_MID_BASE            = 18.0
    STAR_MAG_MID_RANGE           = 4.0
    STAR_MAG_BRIGHT_BASE         = 16.0
    STAR_MAG_BRIGHT_RANGE        = 2.0

    # GalSim numerical parameters
    GALSIM_MAX_FFT_SIZE          = 16384
    GALSIM_FOLDING_THRESHOLD     = 1e-4
    GALSIM_MAXK_THRESHOLD        = 1e-2

    # Training defaults
    DEFAULT_TRAIN_STEPS          = 100_000
    DEFAULT_BATCH_SIZE           = 16
    DEFAULT_EVALUATE_EVERY       = 1000
    DEFAULT_NUM_RES_BLOCKS       = 32
    DEFAULT_CHECKPOINT_DIR       = "./ckpt/wdsr"
    DEFAULT_HR_CROP_SIZE         = 96

    # Reconstruction output
    VIS_RECONSTRUCTION_DIR       = "./data/vis/reconstruction"

    # PSF extraction defaults
    DEFAULT_PSF_SIZE = 255
    DEFAULT_PSF_FWHM = 3.0
    DEFAULT_PSF_THRESHOLD = 50.0
    DEFAULT_PSF_MAX_ITERS = 10
    DEFAULT_PSF_ACCURACY = 0.001
    DEFAULT_PSF_FITS_FILENAME = "euclid_psf.fits"
