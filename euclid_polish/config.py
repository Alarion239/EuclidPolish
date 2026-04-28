"""
Central configuration for EuclidPolish.

This module provides a single source of truth for all configuration values,
eliminating magic strings and numbers scattered throughout the codebase.
"""

import math


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
    VIS_STAR_POSITIONS   = "./data/vis/star_positions.png"

    # TFRecord storage
    RECORDS_DIR          = "./data/images/records"

    # Default values for command-line arguments
    DEFAULT_CUTOUT_SIZE = 512
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
    DEFAULT_GAL_DENSITY_ARCMIN2  = 4.0e5 / 3600.0   # ≈ 111.11 (4×10⁵ galaxies / deg²)
    DEFAULT_STAR_DENSITY_ARCMIN2 = 5.0e3 / 3600.0   # ≈ 1.389 (5×10³ stars / deg²)
    DEFAULT_NIMAGES              = 100

    # VIS instrument
    # Catalog zeropoint: interprets MER catalog flux_vis_1fwhm_aper → AB mag
    # (used by euclid/catalog.py — do not change without re-validating against
    # real Euclid Q1 catalog data).
    DEFAULT_VIS_ZEROPOINT        = 26.2
    VIS_PIXEL_SCALE_ARCSEC       = 0.10     # native Euclid VIS pixel scale (arcsec/pixel)

    # Euclid VIS detector parameters (MSSL VIS-PP, Cropper+ 2014, Euclid Q1 docs)
    EXPOSURE_TIME_S              = 565.0    # single VIS frame duration (s)
    N_EXPOSURES                  = 4        # Wide Survey dithers per stack
    T_TOTAL_S                    = N_EXPOSURES * EXPOSURE_TIME_S  # 2260.0 s
    READ_NOISE_E                 = 4.5      # RMS read noise per exposure (e⁻)
    GAIN_E_PER_ADU               = 3.1      # documentation only; pipeline stays in e⁻
    DARK_E_PER_S_PER_PIX         = 0.001    # dark current (e⁻/pix/s)
    SKY_MAG_AB_ARCSEC2           = 22.35    # typical Wide Survey sky brightness
    VIS_AB_ZP_E_PER_S            = 25.50    # m_AB of source giving 1 e⁻/s

    # Simulator zeropoint: m_AB of a source contributing 1 e⁻ over the full
    # stacked integration. Used by clean_generator.py to convert magnitude →
    # expected electrons-per-pixel for a synthetic point source.
    SIM_VIS_ZEROPOINT_E          = VIS_AB_ZP_E_PER_S + 2.5 * math.log10(T_TOTAL_S)

    # Sky surface brightness in e⁻/s/arcsec², derived from VIS_AB_ZP_E_PER_S.
    SKY_E_PER_S_PER_ARCSEC2      = 10 ** (-0.4 * (SKY_MAG_AB_ARCSEC2 - VIS_AB_ZP_E_PER_S))

    # PSF convolution / model normalization defaults
    DEFAULT_REBIN_FACTOR         = 2
    DEFAULT_ADD_NOISE            = True
    # Stretch scale used to compress the dynamic range of pixel values before
    # the network sees them: stretched = asinh(x / STRETCH_SCALE_E). Linear for
    # |x| ≪ scale, log-like for |x| ≫ scale; signed; smooth inverse (sinh).
    STRETCH_SCALE_E              = 1000.0   # e⁻ — knee between linear and log regimes

    # Star magnitude distribution (probability thresholds and ranges)
    STAR_MAG_PROB_FAINT          = 0.70     # below → faint bin
    STAR_MAG_PROB_MID            = 0.95     # below → mid bin
    STAR_MAG_FAINT_BASE          = 22.0
    STAR_MAG_FAINT_RANGE         = 3.0
    STAR_MAG_MID_BASE            = 18.0
    STAR_MAG_MID_RANGE           = 4.0
    STAR_MAG_BRIGHT_BASE         = 16.0
    STAR_MAG_BRIGHT_RANGE        = 2.0

    # Donut-galaxy (toy gravitational-lens ring) defaults.
    # Sized so the central hole is blurred by the Euclid VIS PSF (FWHM ≈ 0.14"):
    # at the small end the hole vanishes into a fuzzy blob; at the large end
    # the ring/arc structure survives but the hole is partially filled in.
    DEFAULT_DONUT_DENSITY_ARCMIN2 = 60.0   # ≈2.7 donuts per 256² HR field at 0.05"/pix
    DONUT_RADIUS_ARCSEC_MIN       = 0.12   # ring radius lower bound (arcsec)
    DONUT_RADIUS_ARCSEC_MAX       = 0.36   # ring radius upper bound (arcsec)
    DONUT_THICKNESS_FRAC          = 0.15   # σ_thickness / radius (Gaussian thickness)
    DONUT_MAG_MIN                 = 21.0   # bright end of donut magnitude range
    DONUT_MAG_MAX                 = 24.0   # faint end
    DONUT_ELLIPTICITY_MAX         = 0.40   # |g_total| upper bound for random shear
    DONUT_STAMP_PIX               = 64     # numpy stamp side at HR pixel scale (3.2")

    # GalSim numerical parameters
    GALSIM_MAX_FFT_SIZE          = 16384
    GALSIM_FOLDING_THRESHOLD     = 1e-4
    GALSIM_MAXK_THRESHOLD        = 1e-2

    # Training defaults
    DEFAULT_TRAIN_STEPS          = 100_000
    DEFAULT_BATCH_SIZE           = 16
    DEFAULT_EVALUATE_EVERY       = 1000
    DEFAULT_VALIDATE_IMAGES      = 30
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

    # Euclid archive authentication
    DEFAULT_CREDENTIALS_FILE = "~/.euclid_credentials"   # two lines: username, password
    DEFAULT_BRIGHTEST_N       = 1000
