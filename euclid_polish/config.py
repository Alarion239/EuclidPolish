"""
Central configuration for EuclidPolish.

This module provides a single source of truth for all configuration values,
eliminating magic strings and numbers scattered throughout the codebase.
"""

import math
import os
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Tuple


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

    ``pixel_scale_lr_arcsec`` is the **archive mosaic** pixel scale that
    reaches the network — 0.10"/pix for *every* band, because the Euclid
    MER pipeline resamples VIS and NISP alike onto a common 0.10" grid. It
    drives archive cutout sizing, the downloader, and ePSF oversampling.

    ``native_detector_scale_arcsec`` is the **native detector** pixel pitch
    (0.10" for the VIS CCDs, 0.30" for the NISP HAWAII-2RGs). It is used only
    for the cosmic-ray areal density on the H2RG/CCD (CRs/cm²/s → CRs/pixel);
    it is NOT used by the forward rebin, which works on the uniform 0.10"
    archive grid that reaches the network. For VIS the two scales coincide.
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
    # Native detector pixel pitch in arcsec — used for the cosmic-ray areal
    # density on the H2RG/CCD (CRs/cm²/s → CRs/pixel). NOT used by the forward
    # rebin, which works on the uniform 0.10" archive grid. VIS CCDs sample at
    # 0.10"; NISP HAWAII-2RGs at 0.30". Distinct from ``pixel_scale_lr_arcsec``
    # (the 0.10" archive/network grid). See the class docstring.
    native_detector_scale_arcsec: float = 0.10
    # ePSF oversampling factor: the photutils EPSFBuilder is given this
    # as ``oversampling=N`` so the resulting ePSF lives on a grid with
    # pixel scale ``pixel_scale_lr_arcsec / N``. Every band's archive grid
    # is 0.10", so N=2 puts the saved ePSF at 0.05"/pix (the HR grid the
    # forward model convolves on) for all four bands.
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
    # Stars are deliberately OVER-represented vs the real Wide-Survey sky
    # (~1.389/arcmin²): point sources are the only thing that teaches the
    # network to invert the PSF, and at the real density a 192px HR train
    # crop (0.0256 arcmin²) lands a star only ~3.5% of the time — the net
    # never sees one and learns to blur, not deconvolve. At 100/arcmin² a
    # crop averages ~2.6 stars (~92% contain ≥1), so deconvolution gets a
    # constant gradient. This is a TRAINING-SIGNAL knob, not sky realism;
    # the magnitudes stay physical (real 13–25 VIS range, per-band colours,
    # flux via each band's sim_zeropoint_e), only the COUNT is inflated.
    DEFAULT_STAR_DENSITY_ARCMIN2 = 3.6e5 / 3600.0   # = 100 stars / arcmin²
    DEFAULT_NIMAGES              = 100

    # VIS instrument
    # AB zeropoint for MER fluxes quoted in microJansky (µJy): the catalogue's
    # flux columns (flux_vis_psf, flux_vis_*fwhm_aper, …) are in µJy, and for
    # an AB flux ``mag_AB = AB_ZP_UJY − 2.5·log10(flux_µJy)`` with
    # ``AB_ZP_UJY = 8.90 + 2.5·log10(1e6) = 23.90`` (3631 Jy = AB 0). Star-anchor
    # photometry uses this physical scale (the absolute electron scale is then
    # confirmed against the cutouts by scripts/verify_star_photometry.py).
    AB_ZP_UJY                    = 23.90
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

    # VIS fields reference the scalar VIS constants above so each physical
    # value (zeropoint, exposure budget, sky, noise) is defined exactly ONCE.
    # This is what keeps the per-band ``sim_zeropoint_e`` in lock-step with the
    # scalar ``SIM_VIS_ZEROPOINT_E`` / ``SKY_E_PER_S_PER_ARCSEC2`` — editing the
    # scalar updates both, so the synthetic and real-cutout electron scales can
    # never silently diverge.
    BAND_VIS = BandConfig(
        name                    = "VIS",
        pixel_scale_lr_arcsec   = VIS_PIXEL_SCALE_ARCSEC,
        psf_fwhm_arcsec         = 0.16,
        psf_fits_filename       = "euclid_psf_VIS.fits",
        zeropoint_ab_e_per_s    = VIS_AB_ZP_E_PER_S,
        sky_mag_ab_arcsec2      = SKY_MAG_AB_ARCSEC2,
        exposure_time_s         = EXPOSURE_TIME_S,
        n_exposures             = N_EXPOSURES,
        read_noise_e            = READ_NOISE_E,
        dark_e_per_s_per_pix    = DARK_E_PER_S_PER_PIX,
        asinh_stretch_scale_e   = 1000.0,
        # VIS CR rejection via across-dither image differencing kills ~98%
        # of hits before they reach the science image (0.02 ≈ 1/50).
        cr_rate_factor          = 0.02,
    )

    # Euclid archive delivers every band — VIS and NISP alike — resampled to
    # 0.10″/pixel mosaics, so the network input grid (the LR grid) is uniform
    # across all four bands and ePSF oversampling = 2 puts every saved PSF on
    # the same 0.05″/pix HR grid the forward model convolves on.
    # NISP constants from Schirmer+ 2022 (A&A 662, A92, NISP photometric
    # system; arXiv:2203.01650) and Euclid Coll. III (Schirmer+ 2025,
    # A&A 697, A3 NISP Instrument); ROS exposure time from Scaramella+ 2022
    # (A&A 662, A112). Read noise = effective per-ramp value after MACC
    # up-the-ramp slope fitting (Kubik+ 2021, arXiv:2104.12752); single
    # CDS noise is ~13 e⁻ but the ramp fit suppresses it to ~7.5.
    BAND_Y_E = BandConfig(
        name                    = "Y_E",
        pixel_scale_lr_arcsec   = 0.10,
        native_detector_scale_arcsec = 0.30,
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
        native_detector_scale_arcsec = 0.30,
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
        native_detector_scale_arcsec = 0.30,
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
    # Clip on the asinh-space value before the inverse sinh, so a runaway SR
    # output can't overflow when un-stretched: sinh(20)·STRETCH_SCALE_E ≈ 2.4e8 e⁻.
    SINH_STRETCH_CLIP            = 20.0

    # PSNR peak references — set to a mag-17 star's electron count over the
    # stack (a "very bright" plausible source). The asinh-space peak is the
    # asinh of the raw peak under the same STRETCH_SCALE_E. Both are derived
    # quantities; change PSNR_PEAK_MAG to retune.
    PSNR_PEAK_MAG                = 17.0
    PSNR_PEAK_E                  = 10 ** (-0.4 * (PSNR_PEAK_MAG - SIM_VIS_ZEROPOINT_E))
    PSNR_PEAK_STRETCHED          = math.asinh(PSNR_PEAK_E / STRETCH_SCALE_E)

    # Star magnitude distribution (probability thresholds and ranges).
    # The bright bin is widened toward brighter (mag 13) and given a higher
    # weight so the network sees BIG stars — brighter point sources spread
    # more flux above the noise through the PSF, so their wings (and the
    # diffraction structure the model must reconstruct) extend much further.
    # Without these, the brightest synthetic star was only mag 16.
    STAR_MAG_PROB_FAINT          = 0.65     # below → faint bin
    STAR_MAG_PROB_MID            = 0.88     # below → mid bin (bright bin = 12%)
    STAR_MAG_FAINT_BASE          = 22.0
    STAR_MAG_FAINT_RANGE         = 3.0
    STAR_MAG_MID_BASE            = 18.0
    STAR_MAG_MID_RANGE           = 4.0
    STAR_MAG_BRIGHT_BASE         = 13.0     # was 16.0 — add genuinely big stars
    STAR_MAG_BRIGHT_RANGE        = 3.0      # 13–16

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

    # Non-negativity regularisation on SR (the model's deconvolved-sky
    # output). Surface brightness is physically >= 0, but the WDSR head is
    # unconstrained, so SR can go slightly negative. A soft penalty
    # ``λ · mean(relu(-SR))`` (asinh space) added to every step's loss
    # pushes the model to output >= 0 naturally across all three lanes
    # (they all branch from the same SR). 0 disables; tune by watching the
    # validation negative-pixel fraction. A penalty makes negatives
    # rare/small, not impossible — clamp the delivered product for a hard
    # guarantee.
    NONNEG_SR_WEIGHT             = 1.0


    # Training defaults
    DEFAULT_TRAIN_STEPS          = 100_000
    # 4 (was 16): the HR crop was enlarged 96 -> 192 (see DEFAULT_HR_CROP_SIZE),
    # a 4x activation-area increase. Dropping the batch 4x keeps per-step memory
    # roughly constant (16*48^2 == 4*96^2 LR pixels) and matches the original
    # POLISH default (batch=4).
    DEFAULT_BATCH_SIZE           = 4
    DEFAULT_EVALUATE_EVERY       = 1000
    DEFAULT_VALIDATE_IMAGES      = 100
    DEFAULT_NUM_RES_BLOCKS       = 32
    DEFAULT_CHECKPOINT_DIR       = os.environ.get(
        "EUCLID_POLISH_CKPT_DIR", "./ckpt/wdsr",
    )
    # 192 HR / 96 LR (was 96/48). The WDSR-B receptive field is ~69 LR px;
    # a 48-px LR crop clipped it, so the model could never see context beyond
    # ~46 px and large objects (bright stars, extended galaxies) never fit in
    # one crop. 96 LR px now exceeds the RF and admits big sources. Paired with
    # the 4x smaller default batch so training memory stays ~constant.
    DEFAULT_HR_CROP_SIZE         = 192

    # Reconstruction output
    VIS_RECONSTRUCTION_DIR       = os.path.join(DATA_DIR, "vis/reconstruction")
    # Persistent storage for cutouts downloaded by the real-Euclid
    # inference flow (``_job_reconstruct_euclid_cutout``). Each
    # (ra, dec, size) gets its own sub-directory containing the four
    # band FITS files + the super-resolved VIS FITS. Files persist
    # across runs so the user can revisit / re-render a position
    # without re-downloading.
    EUCLID_INFERENCE_DIR         = os.path.join(DATA_DIR, "euclid_inference")

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

    # 512² HR field at 0.05"/pix = 0.182 arcmin² → 1.65/arcmin² yields ~0.30
    # lenses per field on average (10× fewer than the previous 16.5). Still
    # well above the Collett 2015 observed-sky density (~1 per 100 arcmin² at
    # Euclid VIS depth), so the network still sees lensed examples while the
    # scenes look far less lens-crowded. Drop to 1e-2 for fully realistic sky.
    LENS_DENSITY_ARCMIN2    = 1.65
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
    # Detector artifacts (cosmic rays + hot pixels + masked-trail streaks)
    # ---------------------------------------------------------------------
    # CR rate at Euclid's L2 orbit, integrated across the full GCR
    # spectrum (Holmes+ 1989 / 2012 SREM calibration; Euclid mission noise
    # budget): ~5 hits/cm²/s. The Q1 MER pipeline paper (Romelli+ 2025,
    # arXiv:2503.15305) reports that ~1.6% of pixels in a single VIS frame
    # are flagged as CR-affected; cross-dither median rejection across the
    # 4 ROS dithers cuts the *coincident* hits down to ~0.1% — the figure
    # we want to reproduce in the simulated stack. Median deposited charge
    # per surviving hit ~ 1500 e⁻ for a normal-incidence MIP traversing
    # the 100 µm depleted layer.
    CR_RATE_PER_S_PER_CM2  = 5.0
    CR_CHARGE_MEDIAN_E     = 1500.0     # exponential-distribution scale
    # Hot-pixel fraction: ~0.1% of the detector pixels exhibit anomalous
    # dark current that saturates over the integration. Modelled as a
    # large positive offset (effective additional well filling).
    HOT_PIXEL_FRACTION     = 1.0e-3
    HOT_PIXEL_CHARGE_MEAN_E = 10000.0

    # ---------------------------------------------------------------------
    # Long faint "interpolation-residual" streaks
    # ---------------------------------------------------------------------
    # Real Euclid VIS cutouts very often (~30-50% in the Q1 release) show
    # a long, narrow, very faint oblique feature spanning much of the
    # frame. They do not look like bright CR trails — instead they look
    # smooth and noise-suppressed, consistent with MER masking a
    # cosmic-ray trail, satellite/asteroid trail, or persistence streak
    # and interpolating across the masked pixels. The interpolated values
    # have ~zero residual brightness but the *local noise variance* is
    # reduced, leaving a smooth ridge visible against the otherwise noisy
    # background.
    #
    # We model these as a small Poisson number of additive features per
    # frame, with sub-sigma amplitude so they are invisible at normal
    # stretches and only emerge under tight clipping — matching what we
    # actually see in the cutouts.
    STREAK_RATE_PER_KPIX2     = 4.0     # streaks per (1000×1000) LR-pixel area
    STREAK_LENGTH_MEAN_PIX    = 250.0   # exponentially distributed
    STREAK_LENGTH_MIN_PIX     = 40
    STREAK_LENGTH_MAX_PIX     = 2000
    STREAK_WIDTH_PIX_MIN      = 1
    STREAK_WIDTH_PIX_MAX      = 3
    # Per-streak amplitude in units of local noise σ. Drawn uniform from
    # [min, max]; sign random (some interpolations over-shoot, some
    # under-shoot). 0.3-0.8 σ keeps the streak well below 1σ — invisible
    # in normal stretches, just discernible under tight clipping.
    STREAK_AMP_SIGMA_MIN      = 0.3
    STREAK_AMP_SIGMA_MAX      = 0.8

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

    # ---------------------------------------------------------------------
    # HST F814W → Euclid VIS photometric chain
    # ---------------------------------------------------------------------
    # The HST→Euclid TFRecord generator preserves HST's native photometry
    # instead of normalising and re-scaling against a single catalog row
    # — a 25″ × 25″ HLSP cutout typically contains many sources, and the
    # old "unit-flux × catalog_flux" path silently allocated a single
    # galaxy's electron budget across all of them, leaving every source
    # 10–100× dimmer than its real Euclid magnitude. We multiply the
    # HST cutout by ``HST→VIS_rate_ratio × t_total_VIS × area_correction``
    # instead; every source ends up at its physical brightness automatically.
    #
    # The two constants below feed that chain:
    #   - HLSP COSMOS F814W mosaics are calibrated such that pixel values
    #     are electrons/second; the AB zeropoint is ~25.94 (standard
    #     HAP/HLSP ACS/WFC F814W ZP, e⁻/s units). See
    #     https://archive.stsci.edu/hlsp/cosmos for the data products spec.
    HST_ACS_F814W_AB_ZP_E_PER_S   = 25.94
    #   - HLSP delivers F814W mosaics drizzled to 0.03″/pixel. Used in
    #     the resample step to compute the area correction
    #     (HR_pixel_area / HST_pixel_area) = (0.05/0.03)² ≈ 2.78 that
    #     compensates for interpolation-based zoom (preserves surface
    #     brightness, not per-pixel integrated flux).
    HST_HLSP_PIXEL_SCALE_ARCSEC   = 0.03

    # =====================================================================
    # Consolidated module constants (nested topical namespaces)
    # =====================================================================
    # Knobs that used to live as module-level constants in individual
    # pipeline modules are gathered here so there's a single place to tune
    # them. Each group is a frozen dataclass used purely as a namespace
    # (access via e.g. ``Config.HST.PSF_HALF_SIDE_PIX``). Derived full paths
    # that join these names onto ``DATA_DIR`` follow as flat attributes,
    # matching the existing path section.

    @dataclass(frozen=True)
    class HST:
        """HST F814W HLSP → Euclid VIS pipeline (download → ePSF → TFRecords).

        Migrated from scripts/fasrc_download_hst_hlsp.py,
        fasrc_extract_hst_psf.py, fasrc_generate_hst_tfrecords.py and
        fasrc_compute_differential_kernel.py.
        """
        # --- HLSP mosaic download (MAST) ---
        TARGET_NAME: str         = "COSMOS"
        FILTER: str              = "F814W"
        OBS_COLLECTION: str      = "HLSP"
        MAST_SCRATCH_SUBDIR: str = "mastDownload"  # MAST's nested write subtree
        # --- directory / file names under DATA_DIR ---
        HLSP_DIR_NAME: str    = "hst_hlsp"
        PSF_DIR_NAME: str     = "hst_psf"
        PSF_FILE_NAME: str    = "F814W.fits"
        STARS_DIR_NAME: str   = "hst_stars"   # per-star ePSF stamp library (UI)
        DIFF_KERNEL_FILE: str = "diff_kernel_VIS.fits"
        RECORDS_SUBDIR: str   = "images/records_v2_hst"
        # --- ePSF extraction ---
        FALLBACK_PIX_SCALE_ARCSEC: float = 0.05  # only if WCS lacks CDELT/CD
        EPSF_OVERSAMPLING: int     = 1           # keep PSF at native tile scale
        PSF_HALF_SIDE_PIX: int     = 255         # ePSF side = 2*half+1 = 511
        MAX_STARS_PER_TILE: int    = 100         # cap stars taken per HLSP tile
        MAX_UNCOVERED_FRAC: float  = 0.005       # >0.5% zero/NaN → seam/hole
        MIN_STAMP_PEAK_SNR: float  = 30.0        # peak ≥ this × MAD σ, else noise
        MAX_PEAK_OFFCENTER_PX: int = 5           # peak within this of stamp centre
        # --- TFRecord generation ---
        MIN_TILE_COVERAGE: float = 0.95          # min finite/non-zero fraction
        MIN_SOURCE_PIXELS: int   = 3             # min above-threshold source pixels

    @dataclass(frozen=True)
    class Matching:
        """Position/size tolerances. Catalog dedup and archive-cutout reuse
        use deliberately different position tolerances (0.05″ vs 0.5″) — they
        were two separate ``POSITION_TOLERANCE_ARCSEC`` constants before."""
        CATALOG_POSITION_TOL_ARCSEC: float  = 0.05  # euclid/catalog.py dedup
        DOWNLOAD_POSITION_TOL_ARCSEC: float = 0.5   # euclid/downloader.py reuse
        DOWNLOAD_SIZE_TOL_PIXELS: int       = 10

    @dataclass(frozen=True)
    class WebFetch:
        """FASRC artifact fetch cache + job-progress smoothing (euclid_polish/web)."""
        CACHE_SUBDIR: str          = "_fasrc_cache"
        MAX_PULL_BYTES: int        = 50 * 1024 * 1024        # per-file pull cap
        # ePSF FITS are now multi-extension (mean + K cluster PSFs), so a band
        # file is tens-to-hundreds of MB — well over the generic cap. They are
        # bounded (K kernels), so the PSF fetch path uses this larger cap.
        MAX_PSF_PULL_BYTES: int    = 512 * 1024 * 1024       # ePSF stack pull cap
        CACHE_TTL_SECONDS: int     = 300                     # re-pull after this
        MAX_CACHE_BYTES: int       = 2 * 1024 * 1024 * 1024  # 2 GB LRU budget
        STEP_RATE_EMA_ALPHA: float = 0.1                     # job step-rate EMA

    @dataclass(frozen=True)
    class EuclidSky:
        """Real-Euclid sky cutout download + round-trip TFRecord layout."""
        SKY_SUBDIR: str               = "euclid_sky"
        CUTOUTS_SUBDIR: str           = "cutouts"
        SKY_CATALOG_FILENAME: str     = "sky_positions.csv"
        ROUNDTRIP_RECORDS_SUBDIR: str = "images/records_v2_euclid_roundtrip"

    @dataclass(frozen=True)
    class Color:
        """Visualization color constants (euclid_polish/visualization/color.py).

        Solar absolute mags (Willmer 2018, ApJS 236:47; HST F814W for VIS,
        2MASS Y/J/H for NISP) and default RGB band picks ``(R, G, B)``."""
        SOLAR_AB_MAG: ClassVar[Dict[str, float]] = {
            "VIS": 4.52,   # HST F814W proxy
            "Y_E": 4.92,   # 2MASS Y proxy
            "J_E": 5.10,   # 2MASS J
            "H_E": 5.12,   # 2MASS H
        }
        RGB_SCHEMES: ClassVar[Dict[str, Tuple[str, str, str]]] = {
            "vis_nisp":  ("H_E", "J_E", "VIS"),   # spans full Euclid range
            "nisp_only": ("H_E", "J_E", "Y_E"),   # NIR-only, VIS-free
            "h_y_vis":   ("H_E", "Y_E", "VIS"),   # wider green-channel spacing
        }

    # --- derived full paths (join the nested names onto DATA_DIR) ---
    HLSP_DIR               = os.path.join(DATA_DIR, HST.HLSP_DIR_NAME)
    HST_PSF_DIR            = os.path.join(DATA_DIR, HST.PSF_DIR_NAME)
    HST_PSF_PATH           = os.path.join(HST_PSF_DIR, HST.PSF_FILE_NAME)
    HST_DIFF_KERNEL_PATH   = os.path.join(HST_PSF_DIR, HST.DIFF_KERNEL_FILE)
    HST_RECORDS_DIR        = os.path.join(DATA_DIR, HST.RECORDS_SUBDIR)
    EUCLID_SKY_DIR         = os.path.join(DATA_DIR, EuclidSky.SKY_SUBDIR)
    EUCLID_SKY_CUTOUTS_DIR = os.path.join(EUCLID_SKY_DIR, EuclidSky.CUTOUTS_SUBDIR)
    ROUNDTRIP_RECORDS_DIR  = os.path.join(DATA_DIR, EuclidSky.ROUNDTRIP_RECORDS_SUBDIR)
    FASRC_CACHE_DIR        = os.path.join(DATA_DIR, WebFetch.CACHE_SUBDIR)

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
