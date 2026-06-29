"""
Central configuration for EuclidPolish.

This module provides a single source of truth for all configuration values,
eliminating magic strings and numbers scattered throughout the codebase.
"""

import math
import os
from dataclasses import dataclass
from typing import ClassVar

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

    COSMOS2025_CATALOG_PATH    = os.path.join(DATA_DIR, "COSMOS2025/cosmos2025.fits")
    COSMOS2025_HDU_PHOTOMETRY = 1       # PHOTOMETRY HOTCOLD AND SE++
    COSMOS2025_HDU_LEPHARE    = 2       # LEPHARE photo-z + physical params
    COSMOS2025_HDU_BD         = 6       # B+D bulge+disk decomposition
    DEFAULT_OUTPUT_DIR        = os.path.join(DATA_DIR, "euclid_stars")
    EUCLID_PSF_DIR            = os.path.join(DATA_DIR, "euclid_psf")
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

    # Coordinate ranges
    RA_MIN = 265.0
    RA_MAX = 275.0
    DEC_MIN = 62.0
    DEC_MAX = 70.0

    # Visual output constants
    SUCCESS_PREFIX = "✓"
    ERROR_PREFIX = "✗"

    # Header formatting
    HEADER_WIDTH = 60

    # Sky generation defaults
    DEFAULT_IMAGE_SIZE           = 256
    DEFAULT_PIXEL_SCALE          = 0.05     # arcsec / pixel
    DEFAULT_GAL_DENSITY_ARCMIN2  = 4.0e5 / 3600.0   # ≈ 111.11 (4×10⁵ galaxies / deg²)
    # Reverted to the 5dece6f ("worked") density: ~1.389/arcmin² (the real
    # Wide-Survey stellar density). The 10/arcmin² inflation was paired with
    # the 192px crop to guarantee point sources per crop; with the crop back
    # at 96 and densities matched to the baseline, this is a clean apples-to-
    # apples comparison against the run that climbed 55→65 dB.
    DEFAULT_STAR_DENSITY_ARCMIN2 = 5.0e3 / 3600.0   # ≈ 1.389/arcmin²
    DEFAULT_NIMAGES              = 100

    # Empirical-ePSF background cleaning (psf/core.PSF.background_cleaned,
    # applied on load). The EPSFBuilder kernels carry a noise floor across
    # the whole square stamp; convolving a bright star imprints it as a
    # visible square well above the Euclid noise. Two cuts, both ≤ 0 to
    # disable: pixels at or below the FLOOR_PERCENTILE-th percentile of the
    # stamp are zeroed (the bulk of a 511² stamp is noise — real PSF pixels
    # sit far above it), and a radial cosine taper takes the kernel
    # smoothly to zero between INNER_FRAC and OUTER_FRAC of the half-side
    # (0.82→0.98 of half-256 ≈ 210→250 px on a 511² stamp), so no square
    # edge survives. Kernels are re-normalised to sum=1 afterwards.
    PSF_FLOOR_PERCENTILE = 90.0
    PSF_TAPER_INNER_FRAC = 0.82
    PSF_TAPER_OUTER_FRAC = 0.98

    # VIS instrument
    # AB zeropoint for MER fluxes quoted in microJansky (µJy): the catalogue's
    # flux columns (flux_vis_psf, flux_vis_*fwhm_aper, …) are in µJy, and for
    # an AB flux ``mag_AB = AB_ZP_UJY − 2.5·log10(flux_µJy)`` with
    # ``AB_ZP_UJY = 8.90 + 2.5·log10(1e6) = 23.90`` (3631 Jy = AB 0). Star-anchor
    # photometry uses this physical scale (the absolute electron scale is then
    # confirmed against the cutouts by scripts/verify_star_photometry.py).
    AB_ZP_UJY                    = 23.90
    VIS_PIXEL_SCALE_ARCSEC       = 0.10     # native Euclid VIS pixel scale (arcsec/pixel)

    # Euclid VIS detector parameters, in-flight values (Euclid II. VIS,
    # Cropper+ 2024, arXiv:2405.13492; Q1 VIS PF, McCracken+ 2025,
    # arXiv:2503.15303). A nominal science exposure is commanded at 566 s;
    # shutter motion leaves an effective integration of 560.52 s (the
    # EXPTIME keyword in calibrated frames). The ROS also takes two 89.52 s
    # short exposures (dithers 1–2, total 2422 s/pixel in the VIS/MER
    # stacks); we model the four nominal frames only — a 0.04 mag S/N
    # difference in the background-limited regime.
    EXPOSURE_TIME_S              = 560.52   # effective integration per nominal frame (s)
    N_EXPOSURES                  = 4        # Wide Survey dithers per stack
    T_TOTAL_S                    = N_EXPOSURES * EXPOSURE_TIME_S  # 2242.08 s
    READ_NOISE_E                 = 3.6      # RMS read noise per exposure (e⁻);
    #                                         flight chains measure 2.17–4.06 e⁻
    #                                         (Cropper+ 2024); 4.5 e⁻ was the
    #                                         requirement ceiling, not typical.
    DARK_E_PER_S_PER_PIX         = 0.001    # dark current (e⁻/pix/s)
    SKY_MAG_AB_ARCSEC2           = 22.35    # typical Wide Survey sky brightness
    # m_AB of a source giving 1 e⁻/s. McCracken+ 2025 (Q1 VIS PF) deliver VIS
    # frames in ADU with a zero-point of 24.57 s⁻¹ (i.e. 1 ADU/s = m_AB 24.57)
    # and a gain of 3.48 e⁻/ADU, so the electron-rate zero-point is
    # 24.57 + 2.5·log10(3.48) = 25.92. Keeping this in lock-step with the gain
    # used for saturation puts synthetic flux, the real-cutout ADU/s→e⁻
    # conversion, and the saturation well all on the one Q1 calibration. (Was
    # 25.50 — ~0.42 mag fainter, implying an inconsistent ~2.36 e⁻/ADU gain.)
    VIS_AB_ZP_E_PER_S            = 25.92

    # Simulator zeropoint: m_AB of a source contributing 1 e⁻ over the full
    # stacked integration. Used by sky/sky_simulator.py to convert
    # magnitude → expected electrons over the stack for each source class.
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
        asinh_stretch_scale_e   = 100.0,
        # VIS CR rejection via across-dither image differencing kills ~98%
        # of hits before they reach the science image (0.02 ≈ 1/50).
        cr_rate_factor          = 0.02,
    )

    # Euclid archive delivers every band — VIS and NISP alike — resampled to
    # 0.10″/pixel mosaics, so the network input grid (the LR grid) is uniform
    # across all four bands and ePSF oversampling = 2 puts every saved PSF on
    # the same 0.05″/pix HR grid the forward model convolves on.
    # NISP zeropoints from Schirmer+ 2022 (A&A 662, A92, NISP photometric
    # system; arXiv:2203.01650). Detector/exposure values from Euclid III.
    # The NISP Instrument (Jahnke+ 2024, arXiv:2405.13493):
    #   - exposure_time_s = 87.2 s is the MACC(4,16,4) integration time
    #     t_int ("effective time accumulating light; the time relevant for
    #     science", Table 4). The 112.0 s sometimes quoted is t_exp, the
    #     planning duration including the reset frame — NOT the
    #     photon-collecting time.
    #   - read noise = flight median photo-mode value after MACC slope
    #     fitting, 5.6–6.7 e⁻ across the 16 detectors (Table 3).
    #   - dark current ~0.02 e⁻/s/pix (Sect. 4.1.2).
    BAND_Y_E = BandConfig(
        name                    = "Y_E",
        pixel_scale_lr_arcsec   = 0.10,
        native_detector_scale_arcsec = 0.30,
        psf_fwhm_arcsec         = 0.40,
        psf_fits_filename       = "euclid_psf_Y.fits",
        zeropoint_ab_e_per_s    = 25.04,
        sky_mag_ab_arcsec2      = 22.3,
        exposure_time_s         = 87.2,
        n_exposures             = 4,
        read_noise_e            = 6.1,
        dark_e_per_s_per_pix    = 0.02,
        asinh_stretch_scale_e   = 100.0,
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
        exposure_time_s         = 87.2,
        n_exposures             = 4,
        read_noise_e            = 6.1,
        dark_e_per_s_per_pix    = 0.02,
        asinh_stretch_scale_e   = 100.0,
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
        exposure_time_s         = 87.2,
        n_exposures             = 4,
        read_noise_e            = 6.1,
        dark_e_per_s_per_pix    = 0.02,
        asinh_stretch_scale_e   = 100.0,
        archive_instrument      = "NISP",
        archive_filter          = "NIR_H",
        detector_pixel_um       = 18.0,
    )

    # Canonical band ordering for the 4-channel network input AND the
    # 4-channel HR target (same bands, same order: the model maps the
    # VIS+NISP LR stack to the deconvolved VIS+NISP sky, band k → band k).
    BANDS = (BAND_VIS, BAND_Y_E, BAND_J_E, BAND_H_E)
    LR_INPUT_BAND_NAMES = ("VIS", "Y_E", "J_E", "H_E")
    HR_TARGET_BAND_NAMES = ("VIS", "Y_E", "J_E", "H_E")
    # The single "primary" HR band: channel 0 (VIS). Still the band the
    # star-anchor delta-targets, the HST lane, and the VIS-only model use.
    HR_TARGET_BAND_NAME = "VIS"
    NUM_LR_CHANNELS = 4
    NUM_HR_CHANNELS = 4

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

    # --- Detector saturation masking (synthetic LR dirty image) ----------------
    # Any pixel at/above the band well depth saturates; the forward model masks a
    # blocky rectangular patch around it to ~0 (the Euclid/MER behaviour), for
    # bright stars AND bright galaxy nuclei alike (see saturation.py).
    #
    # Per-band well depth (electrons), referred to the CO-ADDED LR STACK our
    # forward model produces (VIS = 4 exposures), NOT a single readout.
    #
    #  • VIS — McCracken et al. (2025), Euclid Q1 VIS processing (OU-VIS):
    #    blooming/bleeding sets in at 40–61 kADU (mean 51 kADU) PER READOUT, and
    #    the 16-bit ADC hard-saturates at 65 535 ADU, at a Q1 gain of
    #    3.48 e⁻/ADU. McCracken delivers ADU; we work in electrons, so we apply
    #    the gain: per-readout blooming = 51 000 × 3.48 ≈ 177 480 e⁻. But each of
    #    our 4 exposures saturates independently at that level, and the masked
    #    stack therefore saturates at n_exposures × it ≈ 710 ke⁻. This
    #    stack-referred well reproduces the known VIS saturation magnitude
    #    (m_AB ≈ 17.8, Euclid Q1); a per-readout 177 ke⁻ compared to our stacked
    #    electrons would saturate ~1.5 mag too faint.
    #  • NISP (Y_E/J_E/H_E) — effective stack-referred clip levels for the
    #    4×87.2 s MACC integration (physical H2RG well is larger), from the
    #    STAR_SATURATION_CALIB_MAG calibration (scripts/measure_star_saturation.py).
    VIS_SATURATION_GAIN_E_PER_ADU = 3.48          # McCracken+25 Q1 VIS gain
    VIS_SATURATION_BLOOMING_ADU   = 51_000.0      # per-readout blooming (40–61 kADU)
    STAR_SATURATION_WELL_E = {
        # Per-readout blooming × gain × n_exposures → stack-referred (≈709 920 e⁻).
        "VIS": (BAND_VIS.n_exposures * VIS_SATURATION_BLOOMING_ADU
                * VIS_SATURATION_GAIN_E_PER_ADU),
        "Y_E": 7798.0, "J_E": 4571.0, "H_E": 9550.0,
    }
    # Calibration inputs retained for scripts/measure_star_saturation.py (which
    # re-derives the NISP wells above) and for star rendering (STAR_BAND_OFFSETS_MAG).
    STAR_SATURATION_FWHM_ARCSEC  = 2.0
    STAR_SATURATION_CALIB_MAG    = {"VIS": 14.0, "Y_E": 17.0,
                                    "J_E": 17.5, "H_E": 16.5}
    STAR_SATURATION_JITTER_DEX   = 0.15
    STAR_SATURATION_RECT_MIN_PX  = 3
    STAR_SATURATION_RECT_MAX_PX  = 6
    STAR_SATURATION_MAX_RECTS    = 3        # 1..3 overlapping rectangles

    # PSF convolution / model normalization defaults
    DEFAULT_REBIN_FACTOR         = 2
    # Stretch scale used to compress the dynamic range of pixel values before
    # the network sees them: stretched = asinh(x / STRETCH_SCALE_E). Linear for
    # |x| ≪ scale, log-like for |x| ≫ scale; signed; smooth inverse (sinh).
    # Elbow lowered 1000 → 100: at 1000 the asinh-MAE hid the source flux by
    # ~23000× (the "predict 0" collapse cost only ~6e-4), and the loss mass sat
    # in the rare bright cores. At 100 the knee is a few× the noise floor, the
    # collapse costs ~6× more, and the loss weight shifts onto the faint
    # extended galaxy structure we actually want to reconstruct.
    STRETCH_SCALE_E              = 100.0   # e⁻ — knee between linear and log regimes
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

    # Star magnitude distribution: the standard differential stellar number-count
    # law  dN/dm ∝ 10^(STAR_MAG_SLOPE · m)  over [STAR_MAG_BRIGHT, STAR_MAG_FAINT],
    # sampled smoothly by inverse-CDF (replaces the old 3-bin prior). The slope is
    # the high-Galactic-latitude star-count slope d log N/dm — Euclid observes far
    # from the plane, where it is shallow (~0.14–0.35 in the optical/NIR; e.g.
    # ~0.14 in 18<I<21.5, arXiv:0704.1182). 0.20 ≈ the effective slope the old
    # bins encoded. STAR_MAG_BRIGHT is the brightest star drawn: it was capped at
    # 16 to avoid ill-posed saturated deltas, but real fields contain a few bright
    # stars, so it now extends to 12 (the power law keeps them rare — a few per
    # many stamps). Faint limit ≈ the VIS noise floor. All three are tunable per
    # run (web /config; SkySimulatorConfig.star_mag_*).
    STAR_MAG_SLOPE               = 0.20     # d log N / dm (high Galactic latitude)
    STAR_MAG_BRIGHT              = 12.0     # brightest synthetic star (was a 16 cap)
    STAR_MAG_FAINT               = 25.0     # faint limit (≈ VIS 5σ point-source)


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
    # Spike guard: SKIP the optimiser step entirely when the PRE-clip global
    # grad norm is non-finite or exceeds this. Steady-state norm is ~0.5 and
    # clipping caps the applied step at GRAD_CLIP_NORM, yet a single
    # pathological batch (e.g. a very bright star once the model has climbed
    # and its activations grew) was observed to spike the pre-clip norm to
    # ~4e4 and eject the model from the climbing basin into the background-
    # collapse (loss 0.004 → 6.5, PSNR 54 → 46, no recovery). Skipping that
    # one batch keeps the model on the improving trajectory. 0 disables.
    GRAD_SPIKE_SKIP_NORM         = 50.0
    # The guard is OFF for the first this-many optimiser steps: early-training
    # gradients are legitimately large (the model is far from converged), so a
    # flat cutoff from step 0 would skip the whole warmup and the model would
    # never get going. The damaging spikes are a late-onset effect (~step 6k,
    # once the model is confident), so warming up past them is safe.
    GRAD_SPIKE_SKIP_WARMUP_STEPS = 2000
    # On a post-warmup spike the trainer RESTORES the last checkpoint (model +
    # optimiser state) instead of skipping — skipping only freezes a diverged
    # model (it can't recover), whereas a restore continues from before the
    # spike. After this many rollbacks the LR is clearly too hot, so the
    # trainer HALVES the learning rate and keeps going (instead of aborting).
    # The guard halves ON the Nth rollback (``n_rollbacks >= this``).
    GRAD_SPIKE_MAX_ROLLBACKS     = 2
    # …but don't halve forever: after this many halvings (LR cut to 1/2^N of
    # the original) it's hopeless, so abort with a clear message.
    GRAD_SPIKE_MAX_LR_HALVINGS   = 8

    # Non-negativity regularisation on SR (the model's deconvolved-sky
    # output). Surface brightness is physically >= 0, but the WDSR head is
    # unconstrained, so SR can go slightly negative. A soft penalty
    # ``λ · mean(relu(-SR))`` (asinh space) added to every step's loss
    # pushes the model to output >= 0 naturally across all three lanes
    # (they all branch from the same SR). 0 disables; tune by watching the
    # validation negative-pixel fraction. A penalty makes negatives
    # rare/small, not impossible — clamp the delivered product for a hard
    # guarantee.
    #
    # DISABLED (default 0): forcing SR >= 0 forbids the negative ringing
    # lobes that deconvolution/sharpening produces around bright sources,
    # which steered the model into a smooth, non-negative (blurry) basin.
    # With λ=1 the training log showed loss == loss_syn to 8 decimals (the
    # penalty was already ~0), so removing it only frees the solution space
    # — it does not change the loss number on its own. Pass a positive
    # ``--nonneg-sr-weight`` (or the WebUI field) to re-enable per run.
    NONNEG_SR_WEIGHT             = 0.0


    # Training defaults
    DEFAULT_TRAIN_STEPS          = 100_000
    # 16: reverted alongside the 96px HR crop to match the 5dece6f ("worked")
    # baseline. (Was dropped to 4 only to offset the 192px crop's 4x activation
    # area.) Note: FASRC runs pass the batch explicitly via the WebUI/CLI, so
    # this default mainly affects local runs and config coherence.
    DEFAULT_BATCH_SIZE           = 16
    DEFAULT_EVALUATE_EVERY       = 1000
    DEFAULT_VALIDATE_IMAGES      = 100
    DEFAULT_NUM_RES_BLOCKS       = 32
    DEFAULT_CHECKPOINT_DIR       = os.environ.get(
        "EUCLID_POLISH_CKPT_DIR", "./ckpt/wdsr",
    )
    # Experiment-tracking store (the "lab notebook"). Lives at the repo
    # root by default and is intentionally NOT in git — it is mirrored to
    # persistent FASRC holylabs storage instead (see euclid_polish.tracking).
    # Each campaign records the git commit it was created at as metadata.
    TRACKING_DIR                 = os.environ.get(
        "EUCLID_POLISH_TRACKING_DIR", "./tracking",
    )
    # Time-travel sandboxes: each is a git worktree at a backup's commit
    # plus isolated data/ckpt folders, so old code can be re-run without
    # touching live work. Gitignored. See euclid_polish.tracking.timetravel.
    TIMETRAVEL_DIR               = os.environ.get(
        "EUCLID_POLISH_TIMETRAVEL_DIR", "./.timetravel",
    )
    # 96 HR / 48 LR — reverted to the 5dece6f ("worked") field of view to make
    # a clean apples-to-apples comparison against the run that climbed 55→65 dB.
    # (The 192/96 enlargement was to admit the ~69 LR-px WDSR-B receptive field;
    # restore it once this FOV-matched baseline is confirmed to climb again.)
    DEFAULT_HR_CROP_SIZE         = 96

    # Reconstruction output
    VIS_RECONSTRUCTION_DIR       = os.path.join(DATA_DIR, "vis/reconstruction")
    # Persistent storage for cutouts downloaded by the real-Euclid
    # inference flow (``_job_reconstruct_euclid_cutout``). Each
    # (ra, dec, size) gets its own sub-directory containing the four
    # band FITS files + the super-resolved VIS FITS. Files persist
    # across runs so the user can revisit / re-render a position
    # without re-downloading.
    EUCLID_INFERENCE_DIR         = os.path.join(DATA_DIR, "euclid_inference")

    # Catalog-based evaluation. ``EVAL_CATALOG_DIR`` holds normalized
    # ``id,ra,dec[,grade]`` CSVs (e.g. the Natalie Lines Euclid Q1 strong-lens
    # catalog at ``lens_catalog/lenses.csv``). ``EVAL_RESULTS_DIR`` holds one
    # sub-directory per evaluation run, each with one sub-directory per object
    # (band FITS + SR.fits + eye/solar PNGs) plus a ``manifest.csv``.
    EVAL_CATALOG_DIR             = os.path.join(DATA_DIR, "eval_catalogs")
    EVAL_RESULTS_DIR             = os.path.join(DATA_DIR, "eval_results")

    # PSF extraction defaults
    DEFAULT_PSF_SIZE = 255
    DEFAULT_PSF_FWHM = 3.0
    DEFAULT_PSF_THRESHOLD = 50.0
    DEFAULT_PSF_MAX_ITERS = 10
    DEFAULT_PSF_ACCURACY = 0.001
    DEFAULT_PSF_FITS_FILENAME = "euclid_psf.fits"
    # Use FastEPSFBuilder (gridded-spline residual resampling, ~2-3x faster,
    # output numerically identical to stock photutils). Escape hatch: set
    # False to fall back to the stock photutils EPSFBuilder.
    PSF_FAST_EPSF_BUILDER = True

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

    # Reverted to the 5dece6f ("worked") density: 16.5/arcmin² → ~3 lenses per
    # 512² HR field. Well above the Collett 2015 observed-sky density (~1 per
    # 100 arcmin² at Euclid VIS depth); a training-signal knob, not sky realism.
    # Drop to 1e-2 for fully realistic sky.
    LENS_DENSITY_ARCMIN2    = 16.5
    LENS_Z_LENS_MIN         = 0.20
    LENS_Z_LENS_MAX         = 1.20
    LENS_Z_SOURCE_OFFSET    = 0.30      # minimum z_s - z_l
    LENS_Z_SOURCE_MAX       = 3.50
    # Physical half-light-radius cap on the lensed *source* galaxy (kpc).
    # Sources are drawn from COSMOS by redshift only; without a size cut a
    # genuinely large, nearby-ish galaxy can be selected and rendered at its
    # full catalog angular size, producing an unphysically big lensed source.
    # A *physical* ceiling (vs. an angular one) is the right cut: at fixed
    # physical size the rendered angular size is r_phys / D_A(z_source), so a
    # more distant source automatically appears smaller — exactly the regime
    # real strong-lensing sources occupy (compact, high-z star-forming
    # galaxies, r_e ~ 1–4 kpc). Loosen to keep more big galaxies; tighten for
    # only the most compact sources. Set ≤0 to disable the cut.
    LENS_SOURCE_MAX_PHYS_RE_KPC = 5.0
    LENS_SIGMA_V_MIN_KMS    = 150.0     # velocity-dispersion truncation
    LENS_SIGMA_V_MAX_KMS    = 350.0
    LENS_AXIS_RATIO_MIN     = 0.50      # lens-galaxy axis ratio q
    LENS_AXIS_RATIO_MAX     = 0.95
    LENS_EXT_SHEAR_SIGMA    = 0.05      # 1-D Gaussian on each γ component
    LENS_SOURCE_OFFSET_FRAC = 0.7       # source impact parameter in units of θ_E

    # ---------------------------------------------------------------------
    # TNG redshift realism (sky/redshift_model.py)
    # ---------------------------------------------------------------------
    #
    # When the generator's ``tng_redshift_mode`` is on, each injected TNG
    # stamp gets one redshift draw that sets its angular size (rebin factor
    # via D_A), Tolman dimming, and a randomized red-leaning spectral drift;
    # TNG-lit lens galaxies additionally derive σ_v from the subhalo's
    # stellar mass (Faber–Jackson) instead of the uniform prior.

    # Field n(z) for TNG stamps, truncated to [TNG_Z_MIN, TNG_Z_MAX]
    # (z_min=0.10 is where the 100 pc native pixel matches the 0.05″ HR grid).
    #
    # "volume" (default): dN/dz ∝ dV_c/dz · exp(-(z/TNG_Z_PHI_SCALE)²) —
    # the atlas holds only MASSIVE galaxies, visible across the whole survey
    # volume, so their redshift distribution is the comoving volume element
    # (Hogg 1999) times the declining number density of log M*≳11 galaxies
    # (×~10 drop from z≈0.35 to z≈2.25, Muzzin+ 2013 — matched by the
    # Gaussian factor with scale 1.5). Median z ≈ 1.2, ~4% below z = 0.4.
    #
    # "smail": n(z) ∝ z² exp(-(z/TNG_Z0)^1.5) — the full flux-limited-survey
    # population (Smail+ 1995; Euclid Red Book z_m≈0.9 with z0=0.65). Kept
    # for comparison; for the massive-only atlas it over-draws low z (≈13%
    # below z = 0.4 → far too many arcsec-scale giants per field).
    TNG_Z_FORM      = "volume"
    TNG_Z_PHI_SCALE = 1.5
    TNG_Z0    = 0.65
    TNG_Z_MIN = 0.10
    TNG_Z_MAX = 2.50
    # Field-galaxy density in PURE-TNG mode: the real sky density of
    # log M* ≥ TNG_MF_LOGM_MIN galaxies (the mass-function-rescaled TNG
    # population renders all of them, see TNG_MF_* below). Normalization:
    # φ0 ≈ 2.3e-2 Mpc⁻³ for log M* ≥ 8.5 (Baldry+ 2012 double Schechter),
    # times ∫ dV_c/dz/dΩ · exp(-(z/TNG_Z_PHI_SCALE)²) dz over [TNG_Z_MIN,
    # TNG_Z_MAX] ≈ 2.2e3 Mpc³/arcmin² ≈ 50, rounded up to 60 because the
    # Gaussian φ(z) decline is calibrated on massive galaxies and is too
    # aggressive for dwarfs → ~11 draws per 510 px field, of which the
    # undetectably faint (see TNG_FAINT_SKIP_MAG_VIS) are skipped.
    TNG_GAL_DENSITY_ARCMIN2 = 60.0
    # Skip a field draw (before the expensive 4-band stamp load) when its
    # predicted VIS magnitude (L ∝ M + luminosity distance, anchored on
    # measured atlas photometry) is fainter than this — those galaxies
    # exist but contribute nothing detectable (per-pixel 1σ ≈ 25.5,
    # per-arcsec² ≈ 28). ≤ 0 disables.
    TNG_FAINT_SKIP_MAG_VIS = 28.0
    # Optional faint-dwarf Sersic backfill (COSMOS rows with circularized
    # R_e ≤ the cut). Default 0 = OFF: the mass-function-rescaled TNG
    # population covers the small end with real morphology, and pure-TNG
    # stays fully catalog-free. Set > 0 (e.g. 102) to mix analytic Sersic
    # dwarfs back in (then run_pipeline loads COSMOS again).
    TNG_DWARF_SERSIC_DENSITY_ARCMIN2 = 0.0
    TNG_DWARF_MAX_RE_ARCSEC          = 0.5

    # Compactness correction C(z) = C0·(1+z)^BETA applied as extra
    # downsampling on top of the geometric F(z), flux-conserving (surface
    # brightness × C²) — a fixed-mass size correction: the stars and the
    # luminosity stay, only the radius shrinks. Two literature anchors:
    #   BETA = 1.0 — the z = 0 snapshot morphologies must follow the
    #     observed size evolution at fixed mass, R_e ∝ (1+z)^-0.75 (late
    #     types) to (1+z)^-1.48 (early types), van der Wel+ 2014.
    #   C0 = 1.3 — the atlas half-light radii at fixed mass measure
    #     1.24-1.41× the observed z≈0.1 late-type relation (Shen+ 2003) in
    #     every mass bin (vs early types the excess is larger), matching
    #     TNG's known ~0.1-0.2 dex size excess.
    TNG_COMPACT_C0   = 1.3
    TNG_COMPACT_BETA = 1.0
    # Mass rescaling: the atlas selection is top-heavy (log M* ≥ 9.8,
    # median 10.3), so each FIELD stamp is re-used as a smaller galaxy of
    # similar morphology — flux × s (L ∝ M) and an extra size squeeze
    # s^-ALPHA along the observed mass-size relation R ∝ M^0.25 (van der
    # Wel+ 2014); surface brightness then falls as s^0.5, the Kormendy-
    # like trend. Lens deflectors are never rescaled.
    #
    # The target mass is drawn from the real Schechter mass function
    # (per dlogM ∝ (M/M*)^(α+1) e^(-M/M*); α=-1.2, logM*=10.97 — Baldry+
    # 2012 / Muzzin+ 2013) over [LOGM_MIN, LOGM_MAX], then matched to an
    # atlas galaxy with mass within ×MASS_WINDOW above it (closest-decade
    # morphology; caps the shrink at s ≥ 1/MASS_WINDOW). The resulting
    # field population follows the observed mass distribution by
    # construction: many small galaxies, the giants only in the rare
    # Schechter tail. If the property CSV is unavailable, falls back to
    # s ~ log-uniform[RESCALE_MIN, 1] (RESCALE_MIN ≥ 1 disables).
    TNG_MF_LOGM_STAR     = 10.97
    TNG_MF_ALPHA         = -1.2
    TNG_MF_LOGM_MIN      = 8.5
    TNG_MF_LOGM_MAX      = 12.0
    TNG_MASS_WINDOW      = 30.0
    TNG_MASS_RESCALE_MIN = 0.1
    TNG_MASS_SIZE_ALPHA  = 0.25

    # Surface-brightness truncation of redshifted TNG stamps (mag/arcsec²,
    # AB, per band). The SKIRT frames carry nonzero light over the whole
    # 160 kpc box, so without a cut every stamp visually fills the field in
    # the noiseless clean target even though the outskirts are far below
    # anything Euclid can detect. 28.0 ≈ the VIS stack's 1σ per arcsec²
    # (per-pixel 1σ ≈ 25.5; deep-stacking LSB limit ~29.5, Borlaff+ 2022),
    # so everything plausibly recoverable survives. The stamp is cropped to
    # the surviving footprint. ≤ 0 disables.
    TNG_SB_TRUNCATE_MAG_ARCSEC2 = 28.0
    # Stochastic spectral-drift tilt ε ~ N(0, σ0 + σ1·ln(1+z)), applied as
    # exp(ε·ln(λ_b/λ_H)): σ0 is the z=0 colour jitter, σ1 grows the spread
    # with redshift (the rest-UV extrapolation uncertainty). The parametric-k
    # fallback is the mean red tilt used when a stamp's own SED is unusable.
    TNG_DRIFT_SIGMA0       = 0.15
    TNG_DRIFT_SIGMA_SLOPE  = 0.35
    TNG_DRIFT_PARAMETRIC_K = 1.0
    # Faber–Jackson σ_v(M*): σ = σ_ref (M*/M_ref)^slope, lognormal scatter
    # (dex), clipped. σ_ref=200 km/s at M*=1e11 M☉ with slope 0.30 tracks
    # the observed early-type relation (e.g. Zahid+ 2016).
    LENS_FJ_SIGMA_REF_KMS  = 200.0
    LENS_FJ_MSTAR_REF_MSUN = 1.0e11
    LENS_FJ_SLOPE          = 0.30
    LENS_FJ_SCATTER_DEX    = 0.07
    LENS_SIGMA_V_CLIP_KMS  = (100.0, 400.0)
    # Visibility constraint for TNG-lit lenses: require θ_E ≥ ratio × the
    # lens galaxy's apparent half-light radius, else the arcs sit inside the
    # foreground light. (A real stamp can't be shrunk like the Sersic
    # lens light, so the constraint is enforced by rejection.)
    LENS_THETA_E_MIN_RE_RATIO = 1.2
    # "Showable" lens criteria (poster/visualization, NOT the training
    # population): θ_E must clear this fraction of the deflector's VISIBLE
    # (μ-truncated) radius, and the lensed source must keep at least this
    # much VIS flux after cosmological dimming. Enforced analytically
    # before rendering via the cached native-photometry predictors
    # (tng_galaxy.predict_visible_radius_arcsec / predict_vis_flux_e),
    # with a post-render check as backstop.
    LENS_SHOWABLE_THETA_E_FRAC  = 0.5
    LENS_SHOWABLE_MIN_SRC_VIS_E = 1000.0

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
    # Dead / unresponsive pixels: a small fraction of detector pixels collect
    # ~no charge, so they read at the background floor (≈0 after sky/dark
    # subtraction) regardless of what light fell on them. Their real signal is
    # DESTROYED (replacement, not additive), so the network must inpaint the
    # value from neighbours. Rarer than hot pixels; the value is jittered
    # around zero (× the local per-pixel noise σ) so the model keys on the
    # spatial anomaly rather than a magic constant. Randomized per frame so it
    # learns position-independent robustness, not a fixed mask. 3e-4 ≈ a few
    # dozen per 510px band frame.
    DEAD_PIXEL_FRACTION     = 3.0e-4
    DEAD_PIXEL_JITTER_SIGMA = 0.3       # dead-pixel value ~ N(0, this·σ_floor)

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

    # ---------------------------------------------------------------------
    # Provenance & identity
    # ---------------------------------------------------------------------
    #
    # Every generation process and every saved artifact gets an 8-hex
    # ``ProvId`` and a frozen on-disk record. Truth lives in per-object
    # sidecars next to the data; ``PROV_DIR`` only holds the derived,
    # rebuildable index cache (never the source of truth).
    PROV_DIR            = os.path.join(DATA_DIR, "_prov")
    PROV_SCHEMA_VERSION = 1

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
    class GalaxySelection:
        """Real-field-galaxy eval selection window (``EuclidCatalog.query_galaxies``
        + ``eval.galaxy_catalog``). Picks clean, resolved, bigger-end galaxies
        that are confidently *not* lenses. ``DIAM_*`` are on-sky diameters (″),
        turned into a ``segmentation_area`` band via the VIS pixel scale;
        ``MAG_FLOOR`` drops sources fainter than this AB VIS-PSF magnitude."""
        DIAM_LO_ARCSEC: float = 2.0   # bigger-end lower bound
        DIAM_HI_ARCSEC: float = 5.0   # hard cap (must fit the LR stamp)
        MAG_FLOOR: float      = 23.0  # keep galaxies brighter than this
        MAX_RESULTS: int      = 100000  # ADQL TOP-N row cap per cone query

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
    class Tng:
        """IllustrisTNG TNG50-1 SKIRT-atlas synthetic-galaxy download layout.

        The atlas renders each subhalo as Euclid VIS + NISP (Y/J/H) FITS
        cutouts from 5 viewing orientations (``O1``…``O5``). We keep only the
        dusty frames (``TNG<id>_O<k>_Euclid_<band>.fits``); each galaxy lands
        in its own ``<SKIRT_SUBDIR>/<subhalo_id>/`` folder with a ``.done``
        marker so a re-run skips finished galaxies.
        """
        SKIRT_SUBDIR: str  = "tng_skirt"
        # Single-line file (in the user's shared FASRC home) holding the
        # personal IllustrisTNG API token; mirrors the ~/.euclid_credentials
        # convention so the secret never rides through the WebUI form / DB.
        API_KEY_FILE: str  = "~/.tng_api_key"
        # Per-galaxy completion sentinel written after a successful extract.
        DONE_MARKER: str   = ".done"

    @dataclass(frozen=True)
    class Color:
        """Visualization color constants (euclid_polish/visualization/color.py).

        Solar absolute mags (Willmer 2018, ApJS 236:47; HST F814W for VIS,
        2MASS Y/J/H for NISP) and default RGB band picks ``(R, G, B)``."""
        SOLAR_AB_MAG: ClassVar[dict[str, float]] = {
            "VIS": 4.52,   # HST F814W proxy
            "Y_E": 4.92,   # 2MASS Y proxy
            "J_E": 5.10,   # 2MASS J
            "H_E": 5.12,   # 2MASS H
        }
        RGB_SCHEMES: ClassVar[dict[str, tuple[str, str, str]]] = {
            "vis_nisp":  ("H_E", "J_E", "VIS"),   # spans full Euclid range
            "nisp_only": ("H_E", "J_E", "Y_E"),   # NIR-only, VIS-free
            "h_y_vis":   ("H_E", "Y_E", "VIS"),   # wider green-channel spacing
        }
        # Pivot wavelengths (μm) of the four bands — the Planck-spectrum
        # sampling points for the "eye" physical color mode's per-pixel
        # color-temperature fit. VIS: Euclid Collaboration (Cropper+24);
        # NISP Y/J/H: Schirmer+22 Table 4.
        PIVOT_WAVELENGTH_UM: ClassVar[dict[str, float]] = {
            "VIS": 0.715,
            "Y_E": 1.081,
            "J_E": 1.367,
            "H_E": 1.771,
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
    TNG_SKIRT_DIR          = os.path.join(DATA_DIR, Tng.SKIRT_SUBDIR)
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
                            root: str | None = None) -> str:
        """Return ``<root>/<band_name>`` — the subdir where ``band_name``'s
        star cutouts live. Defaults to :attr:`STAR_CUTOUTS_ROOT`."""
        root = root or cls.STAR_CUTOUTS_ROOT
        return os.path.join(root, band_name)
