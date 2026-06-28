"""Tests for ``euclid_polish.config``.

Covers BandConfig invariants, per-band derived quantities, and the lookup API.
Critically also verifies BAND_VIS matches the legacy scalar VIS constants
bit-for-bit, so the new multi-band path can replace the old without changing
the VIS forward model.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from euclid_polish.config import BandConfig, Config

# ---------------------------------------------------------------------------
# BandConfig basics
# ---------------------------------------------------------------------------

def test_bandconfig_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Config.BAND_VIS.read_noise_e = 99.0  # type: ignore[misc]


def test_band_names_unique_and_ordered():
    names = [b.name for b in Config.BANDS]
    assert names == ["VIS", "Y_E", "J_E", "H_E"]
    assert len(set(names)) == len(names)


def test_lr_input_band_names_match_bands_tuple():
    assert tuple(b.name for b in Config.BANDS) == Config.LR_INPUT_BAND_NAMES


def test_native_detector_scale_vis_vs_nisp():
    """VIS samples natively at 0.10″; NISP H2RGs at 0.30″. The field records
    the native detector pitch (used for cosmic-ray areal density); it is not a
    forward-model grid scale — the forward model itself works on the uniform
    0.10″ archive grid (pixel_scale_lr_arcsec) for every band."""
    assert Config.BAND_VIS.native_detector_scale_arcsec == 0.10
    assert Config.BAND_VIS.pixel_scale_lr_arcsec == 0.10
    for b in (Config.BAND_Y_E, Config.BAND_J_E, Config.BAND_H_E):
        assert b.native_detector_scale_arcsec == 0.30, b.name
        assert b.pixel_scale_lr_arcsec == 0.10, b.name
    # Native rebin factor onto the 0.05″ HR grid: VIS 2×, NISP 6×.
    hr = Config.DEFAULT_PIXEL_SCALE
    assert round(Config.BAND_VIS.native_detector_scale_arcsec / hr) == 2
    assert round(Config.BAND_H_E.native_detector_scale_arcsec / hr) == 6


def test_channel_counts():
    assert Config.NUM_LR_CHANNELS == 4
    # 4-band training: the HR target carries every band (VIS + NISP),
    # in the same order as the LR input (band k -> band k).
    assert Config.NUM_HR_CHANNELS == 4
    assert Config.HR_TARGET_BAND_NAMES == Config.LR_INPUT_BAND_NAMES
    assert Config.HR_TARGET_BAND_NAME == "VIS"


# ---------------------------------------------------------------------------
# Per-band derived quantities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("band", list(Config.BANDS))
def test_t_total_s(band: BandConfig):
    assert band.t_total_s == pytest.approx(band.exposure_time_s * band.n_exposures)


@pytest.mark.parametrize("band", list(Config.BANDS))
def test_sim_zeropoint_e_relation(band: BandConfig):
    """sim_zeropoint_e = per-second ZP + 2.5 log10(t_total)."""
    expected = band.zeropoint_ab_e_per_s + 2.5 * math.log10(band.t_total_s)
    assert band.sim_zeropoint_e == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("band", list(Config.BANDS))
def test_sky_e_per_s_per_arcsec2(band: BandConfig):
    """Sky e⁻/s/arcsec² = 10^(-0.4*(m_sky - ZP/s))."""
    expected = 10.0 ** (-0.4 * (band.sky_mag_ab_arcsec2 - band.zeropoint_ab_e_per_s))
    assert band.sky_e_per_s_per_arcsec2 == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("band", list(Config.BANDS))
def test_band_physical_ranges(band: BandConfig):
    """Quick sanity bounds — catch typos / unit slips."""
    assert 0.05 < band.pixel_scale_lr_arcsec <= 0.5
    assert 0.10 < band.psf_fwhm_arcsec   <= 1.0
    assert 20.0 < band.zeropoint_ab_e_per_s <= 30.0
    assert 15.0 < band.sky_mag_ab_arcsec2   <= 25.0
    assert 30.0 < band.exposure_time_s      <= 1000.0
    assert 1   <= band.n_exposures          <= 16
    assert 0.0 <  band.read_noise_e         <= 20.0
    assert 0.0 <= band.dark_e_per_s_per_pix <= 0.1
    assert 100. <= band.asinh_stretch_scale_e <= 1.0e5


def test_all_bands_share_pixel_scale():
    # Euclid archive delivers VIS *and* NISP at 0.10″/pix, so every band's
    # LR grid is uniform.
    for band in Config.BANDS:
        assert band.pixel_scale_lr_arcsec == 0.10


def test_vis_pixel_scale_is_native_lr():
    assert Config.BAND_VIS.pixel_scale_lr_arcsec == 0.10


# ---------------------------------------------------------------------------
# Back-compat: BAND_VIS matches legacy scalar constants exactly
# ---------------------------------------------------------------------------

def test_band_vis_matches_legacy_constants():
    v = Config.BAND_VIS
    assert v.zeropoint_ab_e_per_s == Config.VIS_AB_ZP_E_PER_S
    assert v.exposure_time_s      == Config.EXPOSURE_TIME_S
    assert v.n_exposures          == Config.N_EXPOSURES
    assert v.read_noise_e         == Config.READ_NOISE_E
    assert v.dark_e_per_s_per_pix == Config.DARK_E_PER_S_PER_PIX
    assert v.sky_mag_ab_arcsec2   == Config.SKY_MAG_AB_ARCSEC2
    assert v.pixel_scale_lr_arcsec == Config.VIS_PIXEL_SCALE_ARCSEC
    assert v.sim_zeropoint_e == pytest.approx(Config.SIM_VIS_ZEROPOINT_E)
    assert v.sky_e_per_s_per_arcsec2 == pytest.approx(Config.SKY_E_PER_S_PER_ARCSEC2)


def test_band_vis_stretch_matches_legacy_stretch():
    assert Config.BAND_VIS.asinh_stretch_scale_e == Config.STRETCH_SCALE_E


# ---------------------------------------------------------------------------
# Lookup API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["VIS", "Y_E", "J_E", "H_E"])
def test_get_band_roundtrip(name: str):
    assert Config.get_band(name).name == name


def test_get_band_unknown_raises():
    with pytest.raises(ValueError, match="Unknown band"):
        Config.get_band("BOGUS")


# ---------------------------------------------------------------------------
# Lens-population + NISP-resampling constants
# ---------------------------------------------------------------------------

def test_lens_population_ranges_consistent():
    assert Config.LENS_Z_LENS_MAX > Config.LENS_Z_LENS_MIN
    assert Config.LENS_Z_SOURCE_MAX > Config.LENS_Z_LENS_MAX
    assert Config.LENS_SIGMA_V_MAX_KMS > Config.LENS_SIGMA_V_MIN_KMS
    assert 0.0 < Config.LENS_AXIS_RATIO_MIN < Config.LENS_AXIS_RATIO_MAX <= 1.0
    assert Config.LENS_DENSITY_ARCMIN2 > 0.0


def test_nisp_resampling_constants_legacy():
    # NISP_RESAMPLE_KERNEL / NISP_LR_TO_VIS_LR_RATIO survive as constants
    # for back-compat but are unused on the uniform-0.10″ pipeline (every
    # band's resample factor is 1).
    assert Config.NISP_RESAMPLE_KERNEL in ("lanczos3", "cubic")
    assert Config.NISP_LR_TO_VIS_LR_RATIO == 3


def test_cosmos2025_paths_and_hdus():
    assert Config.COSMOS2025_CATALOG_PATH.endswith(".fits")
    assert Config.COSMOS2025_HDU_PHOTOMETRY == 1
    assert Config.COSMOS2025_HDU_LEPHARE    == 2
    assert Config.COSMOS2025_HDU_BD         == 6


def test_cosmos2025_band_to_column_map_covers_all_bands():
    mapping = Config.COSMOS2025_BAND_TO_CATALOG_COLUMN
    for band in Config.LR_INPUT_BAND_NAMES:
        assert band in mapping
        assert mapping[band].startswith(("hst-", "uvista-"))


def test_star_band_offsets_cover_all_bands():
    offsets = Config.STAR_BAND_OFFSETS_MAG
    assert offsets["VIS"] == 0.0
    for band in Config.LR_INPUT_BAND_NAMES:
        assert band in offsets
        assert isinstance(offsets[band], float)


# ---------------------------------------------------------------------------
# Sersic rendering knobs
# ---------------------------------------------------------------------------

def test_sersic_csub_base_table_is_monotonic_in_n():
    """The (n_max, csub) base lookup must be sorted by n_max ascending."""
    n_max_values = [n for (n, _) in Config.SERSIC_CSUB_BASE_BY_N]
    assert n_max_values == sorted(n_max_values), \
        "SERSIC_CSUB_BASE_BY_N should be sorted by n_max ascending"


@pytest.mark.parametrize("n_max,csub", list(Config.SERSIC_CSUB_BASE_BY_N))
def test_sersic_csub_base_entries_are_odd_positive(n_max, csub):
    assert isinstance(csub, int)
    assert csub >= 1
    assert csub % 2 == 1, "csub must be odd (centre sub-pixel = pixel centre)"


def test_sersic_compactness_bump_table_shape():
    for entry in Config.SERSIC_CSUB_COMPACTNESS_BUMPS:
        assert len(entry) == 3
        r_e_pix_max, n_min, bump = entry
        assert r_e_pix_max > 0
        assert n_min >= 0
        # Bump is either an integer (additive) or the sentinel string "x2_plus_1".
        assert isinstance(bump, (int, str))


def test_sersic_max_cap_is_odd_and_large_enough():
    cap = Config.SERSIC_CSUB_MAX_CAP
    assert isinstance(cap, int) and cap >= 21
    assert cap % 2 == 1


def test_sersic_core_radius_r_e_positive():
    assert Config.SERSIC_CORE_RADIUS_R_E > 0


def test_sersic_stamp_radius_increases_with_n():
    """High-n profiles need wider stamps because their wings are heavier."""
    low  = Config.SERSIC_STAMP_RADIUS_R_E_LOW_N
    mid  = Config.SERSIC_STAMP_RADIUS_R_E_MID_N
    high = Config.SERSIC_STAMP_RADIUS_R_E_HIGH_N
    assert low <= mid <= high


def test_default_csub_uses_config_table():
    """The runtime _default_csub should reflect a config edit."""
    from euclid_polish.sky.generation.profiles import _default_csub
    n = 4.0
    csub_default = _default_csub(n, r_e_pix=4.0)
    assert csub_default >= 1 and csub_default <= Config.SERSIC_CSUB_MAX_CAP
    assert csub_default % 2 == 1
