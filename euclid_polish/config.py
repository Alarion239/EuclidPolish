"""
Central configuration for EuclidPolish.

This module provides a single source of truth for all configuration values,
eliminating magic strings and numbers scattered throughout the codebase.
"""

import math
import os
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Band configuration
# ---------------------------------------------------------------------------
#
# A ``BandConfig`` collects every per-band quantity needed by the multi-band
# forward model: PSF, photometric zeropoint, sky background, exposure budget,
# read/dark noise, and the asinh stretch knee used by the data loader.
#
# The single source of truth for these values is the four ``Config.BAND_*``
# instances below; everything downstream (forward model, data loader,
# visualization) MUST read from there rather than re-hardcoding constants.

@dataclass(frozen=True)
class BandConfig:
    """All per-band parameters for one Euclid imaging band.

    ``zeropoint_ab_e_per_s`` is the AB magnitude of a source giving 1 e⁻/s
    after the full optical chain (instrument response × QE × mirror). It is
    the per-second zeropoint; multiply by the stack integration via
    :attr:`sim_zeropoint_e` to get the magnitude-of-1-electron-over-the-stack.

    ``pixel_scale_lr_arcsec`` is the **native** detector pixel scale (0.10"
    for VIS, 0.30" for the NISP HAWAII-2RGs). After the forward model the
    NISP channels are resampled to match VIS LR scale; the resampled grid
    is what reaches the network, but the noise floor is set by the native
    pixel.
    """

    name: str
    pixel_scale_lr_arcsec: float
    psf_fwhm_arcsec: float
    psf_fits_filename: str
    zeropoint_ab_e_per_s: float
    sky_mag_ab_arcsec2: float
    exposure_time_s: float
    n_exposures: int
    read_noise_e: float
    dark_e_per_s_per_pix: float
    asinh_stretch_scale_e: float
    # Euclid archive identifiers used when downloading cutouts. VIS is
    # served as instrument='VIS' with no filter qualifier; NISP filters
    # are served as instrument='NISP' with filter_name='NIR_Y' etc.
    archive_instrument: str = "VIS"
    archive_filter: str = ""
    # ePSF oversampling factor: the photutils EPSFBuilder is given this
    # as ``oversampling=N`` so the resulting ePSF lives on a grid with
    # pixel scale ``pixel_scale_lr_arcsec / N``. Picked per-band so every
    # band's saved ePSF sits at 0.05"/pix (the HR grid the forward model
    # convolves on): VIS → 2, NISP → 6.
    epsf_oversampling: int = 2
    # Native detector pixel pitch (microns). Used for cosmic-ray rate
    # scaling (CRs/cm²/s → CRs/pixel). VIS CCD pixels are 12 µm; the
    # NISP HAWAII-2RG H2RG pixels are 18 µm.
    detector_pixel_um: float = 12.0
    # Post-rejection CR efficiency. The raw L2 GCR rate (5 hits/cm²/s)
    # is heavily suppressed in VIS by across-dither image differencing
    # (~95–98% of hits are removed → factor ~0.02). NISP's up-the-ramp
    # slope fitting is less aggressive and most hits survive.
    cr_rate_factor: float = 1.0

    @property
    def epsf_pixel_scale_arcsec(self) -> float:
        """Pixel scale of the saved oversampled ePSF (= native / oversampling)."""
        return self.pixel_scale_lr_arcsec / self.epsf_oversampling

    def cutout_size_for_arcsec(self, arcsec_side: float) -> int:
        """Return the native pixel count covering ``arcsec_side`` arcsec.

        Rounded to the nearest integer ≥1 (each cutout side ends up
        ``arcsec_side`` arcsec ± half-a-native-pixel). Used when the
        user requests a fixed *angular* field size and the downloader
        needs to convert that into a per-band pixel count.
        """
        return max(1, int(round(float(arcsec_side) / self.pixel_scale_lr_arcsec)))

    @property
    def t_total_s(self) -> float:
        """Total integration time across all dithers (s)."""
        return self.exposure_time_s * self.n_exposures

    @property
    def sim_zeropoint_e(self) -> float:
        """AB magnitude of a source contributing 1 electron over the stack."""
        return self.zeropoint_ab_e_per_s + 2.5 * math.log10(self.t_total_s)

    @property
    def sky_e_per_s_per_arcsec2(self) -> float:
        """Sky surface brightness in e⁻/s/arcsec²."""
        return 10.0 ** (-0.4 * (self.sky_mag_ab_arcsec2 - self.zeropoint_ab_e_per_s))


class Config:
    """Configuration constants for EuclidPolish."""

    # Single data root: all subdirectories (cutouts, PSFs, TFRecords,
    # visualizations) live under this prefix. Override with the
    # ``EUCLID_POLISH_DATA_DIR`` env var so SLURM jobs can point at
    # netscratch without touching code. Default keeps the historical
    # ``./data`` behavior for local checkouts.
    DATA_DIR = os.environ.get("EUCLID_POLISH_DATA_DIR", "./data")

    DEFAULT_COSMOS_CATALOG_DIR = os.path.join(DATA_DIR, "COSMOS")           # historical alias kept for path-rewriting tools
    COSMOS2025_DIR             = os.path.join(DATA_DIR, "COSMOS2025")
    COSMOS2025_CATALOG_PATH    = os.path.join(DATA_DIR, "COSMOS2025/cosmos2025.fits")
    COSMOS2025_HDU_PHOTOMETRY = 1       # PHOTOMETRY HOTCOLD AND SE++
    COSMOS2025_HDU_LEPHARE    = 2       # LEPHARE photo-z + physical params
    COSMOS2025_HDU_BD         = 6       # B+D bulge+disk decomposition
    DEFAULT_OUTPUT_DIR        = os.path.join(DATA_DIR, "euclid_stars")
    CLEAN_DATA_DIR            = os.path.join(DATA_DIR, "clean_data")
    DIRTY_DATA_DIR            = os.path.join(DATA_DIR, "dirty_data")
    EUCLID_PSF_DIR            = os.path.join(DATA_DIR, "euclid_psf")
    EUCLID_NISP_CUTOUTS_DIR   = os.path.join(DATA_DIR, "euclid_nisp_stars")   # NISP stamps for ePSF
    CATALOG_FILE = "stars.csv"
    CUTOUTS_SUBDIR = "cutouts"

    # Visualization output
    VIS_DIR              = os.path.join(DATA_DIR, "vis")
    VIS_CUTOUTS_DIR      = os.path.join(DATA_DIR, "vis/cutouts")
    VIS_PSF_DIR          = os.path.join(DATA_DIR, "vis/psf")
    VIS_CLEAN_DIR        = os.path.join(DATA_DIR, "vis/clean")
    VIS_DIRTY_DIR        = os.path.join(DATA_DIR, "vis/dirty")
    VIS_STAR_POSITIONS   = os.path.join(DATA_DIR, "vis/star_positions.png")

    # TFRecord storage
    RECORDS_DIR          = os.path.join(DATA_DIR, "images/records")

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

    # ---------------------------------------------------------------------
    # Per-band photometric configuration (multi-band forward model)
    # ---------------------------------------------------------------------
    #
    # VIS values match the scalar VIS constants above (single source of
    # truth for the VIS forward model). NISP Y_E/J_E/H_E values are taken
    # from Schirmer+ 2022 (NISP photometric system, arXiv:2203.01650) and
    # Euclid III. NISP Instrument (Schirmer+ 2025); read/dark are
    # HAWAII-2RG typical values.

    BAND_VIS = BandConfig(
        name                    = "VIS",
        pixel_scale_lr_arcsec   = 0.10,
        psf_fwhm_arcsec         = 0.16,
        psf_fits_filename       = "euclid_psf_VIS.fits",
        zeropoint_ab_e_per_s    = 25.50,
        sky_mag_ab_arcsec2      = 22.35,
        exposure_time_s         = 565.0,
        n_exposures             = 4,
        read_noise_e            = 4.5,
        dark_e_per_s_per_pix    = 0.001,
        asinh_stretch_scale_e   = 1000.0,
        # VIS CR rejection via across-dither image differencing kills ~98%
        # of hits before they reach the science image (0.02 ≈ 1/50).
        cr_rate_factor          = 0.02,
    )

    # Euclid archive delivers every band — VIS and NISP alike — resampled to
    # 0.10″/pixel mosaics. So the LR grid is uniform across all four bands;
    # ePSF oversampling = 2 puts every saved PSF on the same 0.05″/pix HR
    # grid the forward model convolves on.
    # NISP constants from Schirmer+ 2022 (A&A 662, A92, NISP photometric
    # system; arXiv:2203.01650) and Euclid Coll. III (Schirmer+ 2025,
    # A&A 697, A3 NISP Instrument); ROS exposure time from Scaramella+ 2022
    # (A&A 662, A112). Read noise = effective per-ramp value after MACC
    # up-the-ramp slope fitting (Kubik+ 2021, arXiv:2104.12752); single
    # CDS noise is ~13 e⁻ but the ramp fit suppresses it to ~7.5.
    BAND_Y_E = BandConfig(
        name                    = "Y_E",
        pixel_scale_lr_arcsec   = 0.10,
        psf_fwhm_arcsec         = 0.40,
        psf_fits_filename       = "euclid_psf_Y.fits",
        zeropoint_ab_e_per_s    = 25.04,
        sky_mag_ab_arcsec2      = 22.3,
        exposure_time_s         = 112.0,
        n_exposures             = 4,
        read_noise_e            = 7.5,
        dark_e_per_s_per_pix    = 0.01,
        asinh_stretch_scale_e   = 1000.0,
        archive_instrument      = "NISP",
        archive_filter          = "NIR_Y",
        detector_pixel_um       = 18.0,
    )

    BAND_J_E = BandConfig(
        name                    = "J_E",
        pixel_scale_lr_arcsec   = 0.10,
        psf_fwhm_arcsec         = 0.45,
        psf_fits_filename       = "euclid_psf_J.fits",
        zeropoint_ab_e_per_s    = 25.26,
        sky_mag_ab_arcsec2      = 22.1,
        exposure_time_s         = 112.0,
        n_exposures             = 4,
        read_noise_e            = 7.5,
        dark_e_per_s_per_pix    = 0.01,
        asinh_stretch_scale_e   = 1000.0,
        archive_instrument      = "NISP",
        archive_filter          = "NIR_J",
        detector_pixel_um       = 18.0,
    )

    BAND_H_E = BandConfig(
        name                    = "H_E",
        pixel_scale_lr_arcsec   = 0.10,
        psf_fwhm_arcsec         = 0.48,
        psf_fits_filename       = "euclid_psf_H.fits",
        zeropoint_ab_e_per_s    = 25.21,
        sky_mag_ab_arcsec2      = 22.4,
        exposure_time_s         = 112.0,
        n_exposures             = 4,
        read_noise_e            = 7.5,
        dark_e_per_s_per_pix    = 0.01,
        asinh_stretch_scale_e   = 1000.0,
        archive_instrument      = "NISP",
        archive_filter          = "NIR_H",
        detector_pixel_um       = 18.0,
    )

    # Canonical band ordering for the 4-channel network input
    # (HR target uses only BAND_VIS; LR input uses all four in this order).
    BANDS = (BAND_VIS, BAND_Y_E, BAND_J_E, BAND_H_E)
    LR_INPUT_BAND_NAMES = ("VIS", "Y_E", "J_E", "H_E")
    HR_TARGET_BAND_NAME = "VIS"
    NUM_LR_CHANNELS = 4
    NUM_HR_CHANNELS = 1

    # Per-channel cross-band tuning knobs for COSMOS2025-derived bandpasses
    # used as Euclid proxies (UVISTA Y/J/H ≈ NISP Y_E/J_E/H_E, HST F814W ≈ VIS).
    COSMOS2025_BAND_TO_CATALOG_COLUMN = {
        "VIS": "hst-f814w",
        "Y_E": "uvista-y",
        "J_E": "uvista-j",
        "H_E": "uvista-h",
    }

    # NISP → VIS-LR resampling kernel for the forward model.
    # Lanczos-3 matches the SWarp/MER pipeline; spline-cubic is a faster
    # near-equivalent if profiling shows Lanczos to be a bottleneck.
    NISP_RESAMPLE_KERNEL    = "lanczos3"        # one of {"lanczos3", "cubic"}
    NISP_LR_TO_VIS_LR_RATIO = 3                 # 0.30" → 0.10"

    # Fixed stellar SED for the simulator (G-type, V-J ≈ 0.7 mag).
    # Per-band magnitude offsets relative to VIS (m_band - m_VIS).
    STAR_BAND_OFFSETS_MAG = {
        "VIS": 0.00,
        "Y_E": 0.40,
        "J_E": 0.70,
        "H_E": 0.85,
    }

    # PSF convolution / model normalization defaults
    DEFAULT_REBIN_FACTOR         = 2
    DEFAULT_ADD_NOISE            = True
    # Stretch scale used to compress the dynamic range of pixel values before
    # the network sees them: stretched = asinh(x / STRETCH_SCALE_E). Linear for
    # |x| ≪ scale, log-like for |x| ≫ scale; signed; smooth inverse (sinh).
    STRETCH_SCALE_E              = 1000.0   # e⁻ — knee between linear and log regimes

    # PSNR peak references — set to a mag-17 star's electron count over the
    # stack (a "very bright" plausible source). The asinh-space peak is the
    # asinh of the raw peak under the same STRETCH_SCALE_E. Both are derived
    # quantities; change PSNR_PEAK_MAG to retune.
    PSNR_PEAK_MAG                = 17.0
    PSNR_PEAK_E                  = 10 ** (-0.4 * (PSNR_PEAK_MAG - SIM_VIS_ZEROPOINT_E))
    PSNR_PEAK_STRETCHED          = math.asinh(PSNR_PEAK_E / STRETCH_SCALE_E)

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
    DONUT_RADIUS_ARCSEC_MIN       = 0.06   # ring radius lower bound (arcsec)
    DONUT_RADIUS_ARCSEC_MAX       = 0.20   # ring radius upper bound (arcsec)
    DONUT_THICKNESS_FRAC          = 0.15   # σ_thickness / radius (Gaussian thickness)
    DONUT_MAG_MIN                 = 22.5   # bright end of donut magnitude range
    DONUT_MAG_MAX                 = 25.5   # faint end (~4× fainter than 21–24 default)
    DONUT_ELLIPTICITY_MAX         = 0.40   # |g_total| upper bound for random shear
    DONUT_STAMP_PIX               = 64     # numpy stamp side at HR pixel scale (3.2")

    # GalSim numerical parameters
    GALSIM_MAX_FFT_SIZE          = 16384
    GALSIM_FOLDING_THRESHOLD     = 1e-4
    GALSIM_MAXK_THRESHOLD        = 1e-2

    # ---------------------------------------------------------------------
    # Sersic-rasterisation knobs (used by ``euclid_polish.sky.profiles``)
    # ---------------------------------------------------------------------
    #
    # The renderer integrates each pixel by sub-sampling at ``csub × csub``
    # points and averaging. The sub-pixel factor depends on (n, r_e_pix)
    # because the Sersic central peak is sharp for high n / compact r_e.
    #
    # Two-tier "core+wings" sampling: ``csub=1`` over the full source
    # stamp, with the high csub applied only inside a small core stamp of
    # radius ``SERSIC_CORE_RADIUS_R_E × r_e_pix``. This is exact in the
    # core (where the profile changes rapidly per pixel) and good-enough
    # in the wings (where pixel-centre sampling is already <1% accurate).

    # Core stamp radius in units of R_e (in pixels). Larger → more
    # accurate, slower. 3.0 captures ~95% of n=4 flux; smaller values
    # speed up at the cost of leaving slightly-undersampled pixels at the
    # core boundary.
    SERSIC_CORE_RADIUS_R_E  = 3.0

    # Below this csub the two-tier scheme is skipped (the single-pass
    # cost is already small enough). Must be odd.
    SERSIC_CSUB_FAST_PATH_THRESHOLD = 3

    # Per-(n, r_e_pix) sub-pixel sampling heuristic. Each entry is
    # ``(n_max, base_csub)`` evaluated in order: the first match wins.
    # All values must be odd positive integers.
    SERSIC_CSUB_BASE_BY_N = (
        (1.2,  1),   # disks
        (2.0,  3),
        (3.5,  9),
        (5.0, 15),
        (1e9, 21),   # n > 5 — extreme outliers
    )
    # Compactness bumps applied AFTER base lookup (only when n exceeds the
    # threshold listed). Each entry: ``(r_e_pix_max, n_min, csub_increment)``.
    SERSIC_CSUB_COMPACTNESS_BUMPS = (
        (2.0, 1.2, "x2_plus_1"),  # very compact + non-disk → double base + 1
        (6.0, 3.0, 6),             # compact-ish bulge → +6
        (10.0, 2.0, 2),            # moderately compact → +2
    )
    # Upper bound on csub after all adjustments (cap to avoid pathological
    # memory blowups for extreme inputs).
    SERSIC_CSUB_MAX_CAP = 41

    # Stamp truncation: how many R_e to render before the Sersic is
    # treated as negligible. Larger → more flux captured at the wings,
    # slower. The three bins are picked by n.
    SERSIC_STAMP_RADIUS_R_E_LOW_N    = 8.0    # n ≤ 1.5
    SERSIC_STAMP_RADIUS_R_E_MID_N    = 12.0   # 1.5 < n ≤ 3
    SERSIC_STAMP_RADIUS_R_E_HIGH_N   = 18.0   # n > 3
    SERSIC_STAMP_RADIUS_MIN_PIX      = 8.0    # never crop tighter than this

    # Gradient clipping by global L2 norm. Bounds the worst-case update so a
    # bad-batch spike can't corrupt weights. Set to math.inf to disable.
    GRAD_CLIP_NORM               = 5.0


    # Training defaults
    DEFAULT_TRAIN_STEPS          = 100_000
    DEFAULT_BATCH_SIZE           = 16
    DEFAULT_EVALUATE_EVERY       = 1000
    DEFAULT_VALIDATE_IMAGES      = 100
    DEFAULT_NUM_RES_BLOCKS       = 32
    DEFAULT_CHECKPOINT_DIR       = os.environ.get(
        "EUCLID_POLISH_CKPT_DIR", "./ckpt/wdsr",
    )
    DEFAULT_HR_CROP_SIZE         = 96

    # Reconstruction output
    VIS_RECONSTRUCTION_DIR       = os.path.join(DATA_DIR, "vis/reconstruction")

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

    # ---------------------------------------------------------------------
    # Strong lens population (Collett 2015, arXiv:1507.02657)
    # ---------------------------------------------------------------------
    #
    # The simulator places synthetic galaxy-galaxy strong lenses into clean
    # HR fields, with priors drawn from Collett 2015. The values below are
    # the truncation ranges used at sampling time (rejection-sampled where
    # the analytic distribution is unbounded). Lensing geometry uses a flat
    # ΛCDM cosmology with the same H0/Ωm/ΩΛ defaults as Collett.

    LENS_COSMOLOGY_H0       = 70.0      # km/s/Mpc
    LENS_COSMOLOGY_OMEGA_M  = 0.3
    LENS_COSMOLOGY_OMEGA_L  = 0.7

    # 512² HR field at 0.05"/pix = 0.182 arcmin² → 16.5/arcmin² yields ~3
    # lenses per field on average. This is well above the Collett 2015
    # observed-sky density (~1 per 100 arcmin² at Euclid VIS depth) — the
    # over-representation is deliberate so the network sees enough lensed
    # examples per epoch. Drop to 1e-2 for realistic-sky generation.
    LENS_DENSITY_ARCMIN2    = 16.5
    LENS_Z_LENS_MIN         = 0.20
    LENS_Z_LENS_MAX         = 1.20
    LENS_Z_SOURCE_OFFSET    = 0.30      # minimum z_s - z_l
    LENS_Z_SOURCE_MAX       = 3.50
    LENS_SIGMA_V_MIN_KMS    = 150.0     # velocity-dispersion truncation
    LENS_SIGMA_V_MAX_KMS    = 350.0
    LENS_AXIS_RATIO_MIN     = 0.50      # lens-galaxy axis ratio q
    LENS_AXIS_RATIO_MAX     = 0.95
    LENS_EXT_SHEAR_SIGMA    = 0.05      # 1-D Gaussian on each γ component
    LENS_SOURCE_OFFSET_FRAC = 0.7       # source impact parameter in units of θ_E

    # ---------------------------------------------------------------------
    # Detector artifacts (cosmic rays + hot pixels)
    # ---------------------------------------------------------------------
    # CR rate at Euclid's L2 orbit, integrated across the full GCR
    # spectrum (Holmes+ 1989 / 2012 SREM calibration; Euclid mission noise
    # budget): ~5 hits/cm²/s. Median deposited charge per hit ~ 1500 e⁻
    # for a normal-incidence MIP traversing a 100 µm depleted layer.
    CR_RATE_PER_S_PER_CM2  = 5.0
    CR_CHARGE_MEDIAN_E     = 1500.0     # exponential-distribution scale
    # Hot-pixel fraction: ~0.1% of the detector pixels exhibit anomalous
    # dark current that saturates over the integration. Modelled as a
    # large positive offset (effective additional well filling).
    HOT_PIXEL_FRACTION     = 1.0e-3
    HOT_PIXEL_CHARGE_MEAN_E = 10000.0

    # ---------------------------------------------------------------------
    # Multi-band TFRecord storage
    # ---------------------------------------------------------------------
    #
    # ``(H, W, C)`` tensors with explicit channel metadata. Schema is
    # versioned so a reader can check it got what it expected.

    RECORDS_DIR_V2          = os.path.join(DATA_DIR, "images/records_v2")
    TFRECORD_SCHEMA_VERSION = 2

    # ---------------------------------------------------------------------
    # NISP cutout/ePSF extraction
    # ---------------------------------------------------------------------
    #
    # NISP stamps for ePSF construction. One subdirectory per band; same
    # geometry as ``EUCLID_PSF_DIR`` / ``DEFAULT_OUTPUT_DIR`` for VIS.

    NISP_DEFAULT_PSF_SIZE = 127         # native NISP grid (0.30" / 2 oversample)
    # Cutout size for NISP star stamps. NISP PSF FWHM is ~3× VIS FWHM but on
    # ~3× larger pixels — the *pixel* count needed is similar to VIS to cover
    # the wings. Round to odd so the stamp has a true centre pixel.
    NISP_DEFAULT_CUTOUT_SIZE = 255

    # ---------------------------------------------------------------------
    # Per-band PSF kernel sizing
    # ---------------------------------------------------------------------
    #
    # Every PSF is resampled (or generated) onto the HR grid at
    # ``DEFAULT_PIXEL_SCALE`` so the forward model can convolve them with
    # the HR clean field. The *side length* of each kernel is band-specific
    # because broader-FWHM PSFs need more support to capture their wings —
    # there is no good reason to crop H_E to the same number of pixels as
    # VIS when H_E's FWHM is 4× larger.
    #
    # The half-side of every kernel (in pixels at the target scale) is
    #     half = ceil(PSF_HALF_SUPPORT_FWHM_FACTOR × fwhm_arcsec / pixel_scale)
    # and the full side is ``2*half + 1`` (forced odd so the centre pixel
    # sits at the geometric centroid). Setting ``factor = 10`` captures
    # >99% of a Gaussian PSF's flux and is enough for the empirical VIS
    # diffraction spikes (which extend to ~6× FWHM).
    PSF_HALF_SUPPORT_FWHM_FACTOR = 10.0
    PSF_MIN_HALF_PIXELS          = 16    # don't crop tighter than this

    # ---------------------------------------------------------------------
    # Per-band cutout layout
    # ---------------------------------------------------------------------
    #
    # All band cutouts live under one root, in band-named subdirectories::
    #
    #     data/euclid_stars/cutouts/
    #         VIS/  star_XXXX_SSS.fits
    #         Y_E/  star_XXXX_SSS.fits
    #         J_E/  star_XXXX_SSS.fits
    #         H_E/  star_XXXX_SSS.fits
    #
    # The shared ``stars.csv`` catalog tracks per-(band, size) validity so
    # the same star id refers to the same sky position across bands.

    STAR_CUTOUTS_ROOT = os.path.join(DATA_DIR, "euclid_stars/cutouts")

    @classmethod
    def get_band(cls, name: str) -> "BandConfig":
        """Return the ``BandConfig`` instance for a band name (e.g. 'VIS')."""
        for b in cls.BANDS:
            if b.name == name:
                return b
        raise ValueError(f"Unknown band {name!r}. Known: {[b.name for b in cls.BANDS]}")

    @classmethod
    def cutout_dir_for_band(cls, band_name: str,
                            root: Optional[str] = None) -> str:
        """Return ``<root>/<band_name>`` — the subdir where ``band_name``'s
        star cutouts live. Defaults to :attr:`STAR_CUTOUTS_ROOT`."""
        root = root or cls.STAR_CUTOUTS_ROOT
        return os.path.join(root, band_name)
