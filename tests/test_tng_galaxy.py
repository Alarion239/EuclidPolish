"""TNG50 SKIRT mock → injectable electron stamp: pin the three-step recipe
(block-mean rebin → quarter-turn rotate → MJy/sr → electrons) and the
surface-brightness photometry it rests on."""

import math
import os

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.euclid.photometry import (
    mjy_per_sr_to_electrons,
    mjy_per_sr_to_electrons_factor,
    pixel_solid_angle_sr,
    uJy_to_electrons,
)
from euclid_polish.sky.tng_galaxy import (
    block_mean,
    load_tng_frame,
    prepare_tng_galaxy,
    rotate_quarter,
    tng_fits_path,
)

BAND = Config.BAND_VIS
HR_SCALE = Config.DEFAULT_PIXEL_SCALE  # 0.05"

# Real galaxy folders ship in the repo's data/ for an end-to-end check.
# Discover every subhalo folder that has a complete O1 VIS frame so the sweep
# covers all locally-downloaded galaxies, not just one.
def _local_galaxies():
    root = Config.TNG_SKIRT_DIR
    if not os.path.isdir(root):
        return []
    out = []
    for gid in sorted(os.listdir(root)):
        gdir = os.path.join(root, gid)
        if os.path.isdir(gdir) and os.path.isfile(
                tng_fits_path(gdir, gid, 1, "VIS")):
            out.append((gdir, gid))
    return out


_LOCAL = _local_galaxies()
_GAL_DIR, _GAL_ID = (_LOCAL[0] if _LOCAL else (os.path.join(
    Config.TNG_SKIRT_DIR, "167396"), "167396"))
_needs_local = pytest.mark.skipif(
    not _LOCAL, reason="no local TNG sample under data/tng_skirt/")


# ----------------------------- photometry --------------------------------

def test_pixel_solid_angle_closed_form():
    s_rad = 0.05 * math.pi / 180.0 / 3600.0
    assert pixel_solid_angle_sr(0.05) == pytest.approx(s_rad ** 2, rel=1e-12)


def test_mjy_factor_matches_ujy_route():
    # 1 MJy/sr over one pixel is a known µJy flux; the SB factor must equal
    # routing that flux through uJy_to_electrons.
    omega = pixel_solid_angle_sr(HR_SCALE)
    flux_ujy = 1.0e12 * omega                      # 1 MJy/sr · Ω, in µJy
    expected = uJy_to_electrons(flux_ujy, BAND)
    assert mjy_per_sr_to_electrons_factor(BAND, HR_SCALE) == pytest.approx(
        expected, rel=1e-9)


def test_mjy_factor_scales_with_pixel_area():
    # Ω ∝ s²: doubling the pixel scale quadruples electrons-per-pixel.
    f1 = mjy_per_sr_to_electrons_factor(BAND, 0.05)
    f2 = mjy_per_sr_to_electrons_factor(BAND, 0.10)
    assert f2 == pytest.approx(4.0 * f1, rel=1e-9)


def test_mjy_to_electrons_array_is_float32():
    arr = np.ones((4, 4), dtype=np.float32)
    out = mjy_per_sr_to_electrons(arr, BAND, HR_SCALE)
    assert out.dtype == np.float32
    assert out[0, 0] == pytest.approx(
        mjy_per_sr_to_electrons_factor(BAND, HR_SCALE), rel=1e-6)


# ------------------------------- rebin -----------------------------------

def test_block_mean_preserves_surface_brightness():
    # A uniform SB field is unchanged by a block-mean (intensive quantity).
    arr = np.full((8, 8), 3.0, dtype=np.float32)
    out = block_mean(arr, 2)
    assert out.shape == (4, 4)
    assert np.allclose(out, 3.0)


def test_block_mean_value_is_block_average():
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = block_mean(arr, 2)
    # top-left 2x2 block = [[0,1],[4,5]] → mean 2.5
    assert out[0, 0] == pytest.approx(2.5)
    assert out.shape == (2, 2)


def test_block_mean_factor_one_is_copy():
    arr = np.arange(9, dtype=np.float32).reshape(3, 3)
    out = block_mean(arr, 1)
    assert np.array_equal(out, arr) and out is not arr


def test_block_mean_trims_remainder():
    arr = np.ones((5, 5), dtype=np.float32)
    out = block_mean(arr, 2)            # trims to 4x4
    assert out.shape == (2, 2)
    with pytest.raises(ValueError):
        block_mean(arr, 2, trim_remainder=False)


# ------------------------------ rotate -----------------------------------

def test_rotate_quarter_conserves_flux_and_wraps():
    arr = np.random.default_rng(0).random((6, 6)).astype(np.float32)
    for k in range(5):
        out = rotate_quarter(arr, k)
        assert out.sum() == pytest.approx(arr.sum())   # exact, no interp
    assert np.array_equal(rotate_quarter(arr, 4), arr)  # k mod 4
    assert np.array_equal(rotate_quarter(arr, 1), np.rot90(arr, 1))


# --------------------------- end-to-end ----------------------------------

@_needs_local
def test_load_tng_frame_native_float32():
    arr = load_tng_frame(tng_fits_path(_GAL_DIR, _GAL_ID, 1, "VIS"))
    assert arr.dtype == np.float32 and arr.shape == (1600, 1600)
    assert arr.dtype.byteorder in ("=", "|")     # native endianness
    assert np.isfinite(arr).all()


@_needs_local
def test_prepare_tng_galaxy_shape_and_order():
    stamp, meta = prepare_tng_galaxy(_GAL_DIR, _GAL_ID, 1, rebin_factor=4)
    assert stamp.shape == (400, 400, 4) and stamp.dtype == np.float32
    assert meta["bands"] == Config.LR_INPUT_BAND_NAMES
    assert set(meta["flux_e_per_band"]) == set(Config.LR_INPUT_BAND_NAMES)
    assert all(v > 0 for v in meta["flux_e_per_band"].values())


@_needs_local
def test_rebin_acts_as_distance_knob():
    # Coarser rebin (fixed HR scale) ⇒ fainter total flux ∝ 1/factor².
    _, m2 = prepare_tng_galaxy(_GAL_DIR, _GAL_ID, 1, rebin_factor=2)
    _, m4 = prepare_tng_galaxy(_GAL_DIR, _GAL_ID, 1, rebin_factor=4)
    f2 = m2["flux_e_per_band"]["VIS"]
    f4 = m4["flux_e_per_band"]["VIS"]
    assert f4 == pytest.approx(f2 / 4.0, rel=0.02)


@_needs_local
def test_rotation_preserves_total_electrons():
    s0, _ = prepare_tng_galaxy(_GAL_DIR, _GAL_ID, 1, rebin_factor=4, rot_k=0)
    s1, _ = prepare_tng_galaxy(_GAL_DIR, _GAL_ID, 1, rebin_factor=4, rot_k=1)
    assert s1.sum() == pytest.approx(s0.sum(), rel=1e-5)
    assert s1.shape == s0.shape


@_needs_local
@pytest.mark.parametrize("gdir,gid", _LOCAL, ids=[g for _d, g in _LOCAL])
def test_all_local_galaxies_all_orientations(gdir, gid):
    # Every downloaded galaxy, every orientation with a full band set, both
    # rebin factors → a clean, finite, non-negative, positive-flux stamp.
    ran = 0
    for o in range(1, 6):
        if not all(os.path.isfile(tng_fits_path(gdir, gid, o, b))
                   for b in ("VIS", "Y", "J", "H")):
            continue
        for f in (2, 4):
            stamp, meta = prepare_tng_galaxy(gdir, gid, o, rebin_factor=f,
                                             rot_k=o)
            assert stamp.shape == (1600 // f, 1600 // f, 4)
            assert np.isfinite(stamp).all()
            assert (stamp >= 0).all()
            assert meta["flux_e_per_band"]["VIS"] > 0
            ran += 1
    assert ran > 0, f"no complete orientations for TNG{gid}"
