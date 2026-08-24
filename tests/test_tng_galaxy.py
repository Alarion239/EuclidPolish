"""TNG50 SKIRT mock → injectable electron stamp: pin the three-step recipe
(block-mean rebin → quarter-turn rotate → MJy/sr → electrons) and the
surface-brightness photometry it rests on."""

import math
import os

import numpy as np
import pytest
from astropy.io import fits as _fits
from scipy.signal import fftconvolve

from euclid_polish.config import Config
from euclid_polish.photometry import (
    mjy_per_sr_to_electrons,
    mjy_per_sr_to_electrons_factor,
    pixel_solid_angle_sr,
    uJy_to_electrons,
)
from euclid_polish.skirt import image as skirt_image
from euclid_polish.skirt.image import (
    block_mean,
    measure_halflight_radius_px,
    rotate_arbitrary,
    rotate_quarter,
)
from euclid_polish.sky.generation.tng_galaxy import (
    circularize_psf_kernel,
    list_tng_galaxies,
    measure_vis_2fwhm_aperture_flux,
    normalise_tng_to_vis_2fwhm,
    prepare_tng_galaxy,
    sample_tng_stamp,
    tng_fits_path,
    tng_stamp_at_redshift,
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


def test_2fwhm_normalisation_uses_one_scale_and_preserves_colours():
    yy, xx = np.indices((81, 81), dtype=np.float64)
    radius2 = (yy - 40.0) ** 2 + (xx - 40.0) ** 2
    vis = np.exp(-0.5 * radius2 / 6.0 ** 2).astype(np.float32)
    stamp = np.stack([vis, 2.0 * vis, 3.0 * vis, 4.0 * vis], axis=-1)
    psf = np.exp(-0.5 * radius2 / 2.0 ** 2).astype(np.float32)
    original_totals = np.sum(stamp, axis=(0, 1), dtype=np.float64)

    scaled, meta = normalise_tng_to_vis_2fwhm(
        stamp.copy(), {}, target_flux_e=1234.5,
        psf_kernel=psf, psf_fwhm_arcsec=0.2,
        pixel_scale_arcsec=0.05,
    )

    totals = np.sum(scaled, axis=(0, 1), dtype=np.float64)
    np.testing.assert_allclose(
        totals / totals[0], original_totals / original_totals[0], rtol=2e-6,
    )
    assert meta["target_vis_2fwhm_flux_e"] == pytest.approx(1234.5)
    assert meta["achieved_vis_2fwhm_flux_e"] == pytest.approx(1234.5)
    assert meta["aperture_radius_arcsec"] == pytest.approx(0.2)
    assert meta["aperture_diameter_arcsec"] == pytest.approx(0.4)
    assert meta["shared_photometric_scale"] > 0.0
    assert circularize_psf_kernel(psf).sum() == pytest.approx(1.0)


def test_2fwhm_normalisation_rejects_zero_aperture_stamp():
    with pytest.raises(ValueError, match="no positive VIS 2FWHM"):
        normalise_tng_to_vis_2fwhm(
            np.zeros((21, 21, 4), np.float32), {}, target_flux_e=1.0,
            psf_kernel=np.ones((5, 5), np.float32),
            psf_fwhm_arcsec=0.2,
        )


@pytest.mark.parametrize(
    ("image_shape", "psf_shape"),
    [((81, 81), (9, 9)), ((80, 80), (8, 8)), ((80, 81), (7, 10))],
)
def test_compact_aperture_response_matches_full_fft(image_shape, psf_shape):
    rng = np.random.default_rng(81)
    vis = rng.random(image_shape, dtype=np.float32)
    psf = circularize_psf_kernel(rng.random(psf_shape, dtype=np.float32))
    blurred = fftconvolve(
        np.asarray(vis, dtype=np.float64),
        np.asarray(psf, dtype=np.float64),
        mode="same",
    )
    yy, xx = np.indices(image_shape, dtype=np.float64)
    cy, cx = 0.5 * (image_shape[0] - 1), 0.5 * (image_shape[1] - 1)
    aperture = np.hypot(yy - cy, xx - cx) <= 4.0
    expected = float(np.sum(blurred[aperture], dtype=np.float64))

    actual = measure_vis_2fwhm_aperture_flux(
        vis, circular_psf=psf, psf_fwhm_arcsec=0.2,
        pixel_scale_arcsec=0.05, psf_identity="fixture",
    )

    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_circular_psf_and_response_caches_reuse_identity_and_track_parity(
    monkeypatch,
):
    from euclid_polish.sky.generation import tng_galaxy

    monkeypatch.setattr(
        tng_galaxy, "_CIRCULAR_PSF_CACHE",
        type(tng_galaxy._CIRCULAR_PSF_CACHE)(),
    )
    monkeypatch.setattr(tng_galaxy, "_CIRCULAR_PSF_CACHE_BYTES", 0)
    monkeypatch.setattr(
        tng_galaxy, "_APERTURE_RESPONSE_CACHE",
        type(tng_galaxy._APERTURE_RESPONSE_CACHE)(),
    )
    monkeypatch.setattr(tng_galaxy, "_APERTURE_RESPONSE_CACHE_BYTES", 0)
    psf = np.ones((7, 7), dtype=np.float32)
    for side in (21, 21, 20):
        normalise_tng_to_vis_2fwhm(
            np.ones((side, side, 4), dtype=np.float32), {},
            target_flux_e=10.0, psf_kernel=psf,
            psf_fwhm_arcsec=0.2, pixel_scale_arcsec=0.05,
            psf_identity="empirical_vis_psf:3",
        )

    assert len(tng_galaxy._CIRCULAR_PSF_CACHE) == 1
    assert len(tng_galaxy._APERTURE_RESPONSE_CACHE) == 2
    assert tng_galaxy._CIRCULAR_PSF_CACHE_BYTES <= (
        tng_galaxy._CIRCULAR_PSF_CACHE_MAX_BYTES
    )
    assert tng_galaxy._APERTURE_RESPONSE_CACHE_BYTES <= (
        tng_galaxy._APERTURE_RESPONSE_CACHE_MAX_BYTES
    )


def test_aperture_response_uses_fft_for_empirical_psf(monkeypatch):
    from euclid_polish.sky.generation import tng_galaxy

    calls = []

    def fake_fftconvolve(left, right, *, mode):
        calls.append((left.shape, right.shape, mode))
        return np.ones(
            (left.shape[0] + right.shape[0] - 1,
             left.shape[1] + right.shape[1] - 1),
            dtype=np.float64,
        )

    monkeypatch.setattr(tng_galaxy, "fftconvolve", fake_fftconvolve)
    monkeypatch.setattr(
        tng_galaxy, "_APERTURE_RESPONSE_CACHE",
        type(tng_galaxy._APERTURE_RESPONSE_CACHE)(),
    )
    monkeypatch.setattr(tng_galaxy, "_APERTURE_RESPONSE_CACHE_BYTES", 0)

    response = tng_galaxy._aperture_response(
        np.ones((9, 9), dtype=bool),
        np.ones((101, 101), dtype=np.float32),
        psf_identity="large-empirical-fixture",
        psf_fwhm_arcsec=0.2,
        pixel_scale_arcsec=0.05,
        centre_parity=(1, 1),
    )

    assert calls == [((9, 9), (101, 101), "full")]
    assert response.shape == (109, 109)


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


# ------------------------------ rotate -----------------------------------

def test_rotate_quarter_conserves_flux_and_wraps():
    arr = np.random.default_rng(0).random((6, 6)).astype(np.float32)
    for k in range(5):
        out = rotate_quarter(arr, k)
        assert out.sum() == pytest.approx(arr.sum())   # exact, no interp
    assert np.array_equal(rotate_quarter(arr, 4), arr)  # k mod 4
    assert np.array_equal(rotate_quarter(arr, 1), np.rot90(arr, 1))


def _fake_tng_galaxy(tng_dir, gid, *, size=64):
    """A minimal 4-band / 5-orientation fake galaxy with a centred asymmetric
    feature (so rotation is detectable and flux isn't clipped at the corners)."""
    d = os.path.join(tng_dir, gid)
    os.makedirs(d, exist_ok=True)
    c = size // 2
    arr = np.zeros((size, size), dtype=">f4")
    arr[c - 8:c + 8, c - 2:c + 2] = 500.0          # vertical bar
    arr[c - 2:c + 10, c - 6:c + 6] += 200.0        # off-centre block → asymmetric
    for o in (1, 2, 3, 4, 5):
        for b in ("VIS", "Y", "J", "H"):
            _fits.PrimaryHDU(arr).writeto(
                os.path.join(d, f"TNG{gid}_O{o}_Euclid_{b}.fits"), overwrite=True)
    open(os.path.join(d, Config.Tng.DONE_MARKER), "w").close()
    return d


def test_rotate_arbitrary_preserves_flux_and_nonneg():
    img = np.zeros((64, 64), np.float32)
    img[24:40, 28:36] = (np.random.default_rng(0).random((16, 8)) + 0.1).astype(np.float32)
    out = rotate_arbitrary(img, 37.0)
    assert out.shape == img.shape
    assert (out >= 0).all()                            # SB stays non-negative
    assert out.sum() == pytest.approx(img.sum(), rel=0.03)   # flux ~conserved
    assert not np.allclose(out, img)                   # actually rotated


def test_tng_module_keeps_legacy_image_helper_imports():
    from euclid_polish.sky.generation import tng_galaxy

    assert tng_galaxy.block_mean is block_mean
    assert tng_galaxy.rotate_arbitrary is rotate_arbitrary


def test_prepare_arbitrary_rotation_gated_on_rebin(tmp_path):
    d = _fake_tng_galaxy(str(tmp_path / "tng"), "424242", size=64)
    # rebin ≥ 4 + an angle → arbitrary spline rotation.
    _, m_hi = prepare_tng_galaxy(d, "424242", 1, rebin_factor=4, rot_angle=30.0)
    assert m_hi["arbitrary_rotation"] is True
    assert m_hi["rot_angle"] == pytest.approx(30.0)
    # rebin < 4 → falls back to the exact quarter-turn even with an angle given.
    _, m_lo = prepare_tng_galaxy(d, "424242", 1, rebin_factor=2, rot_angle=30.0)
    assert m_lo["arbitrary_rotation"] is False and m_lo["rot_angle"] is None
    # No angle → unchanged quarter-turn path regardless of rebin.
    _, m_none = prepare_tng_galaxy(d, "424242", 1, rebin_factor=4)
    assert m_none["arbitrary_rotation"] is False
    # Arbitrary rotation ~conserves total flux vs the un-rotated stamp.
    _, m0 = prepare_tng_galaxy(d, "424242", 1, rebin_factor=4)
    assert (m_hi["flux_e_per_band"]["VIS"]
            == pytest.approx(m0["flux_e_per_band"]["VIS"], rel=0.05))


# --------------------------- end-to-end ----------------------------------

@_needs_local
def test_load_skirt_frame_native_float32():
    arr = skirt_image.load_skirt_frame(
        tng_fits_path(_GAL_DIR, _GAL_ID, 1, "VIS")
    )
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


# ---------------------------------------------------------------------------
# Enumeration + random sampling (for injection into synthetic scenes)
# ---------------------------------------------------------------------------

def _write_fake_galaxy(tng_dir, gid, *, size=24, done=True):
    """A tiny stand-in galaxy: 5 orientations × 4 bands of MJy/sr frames with a
    bright core, plus a .done marker. ``size`` is divisible by 1/2/3/4."""
    d = os.path.join(tng_dir, gid)
    os.makedirs(d, exist_ok=True)
    for o in (1, 2, 3, 4, 5):
        for b in ("VIS", "Y", "J", "H"):
            arr = np.zeros((size, size), dtype=">f4")
            arr[size // 2 - 2:size // 2 + 2, size // 2 - 2:size // 2 + 2] = 500.0
            _fits.PrimaryHDU(arr).writeto(
                os.path.join(d, f"TNG{gid}_O{o}_Euclid_{b}.fits"), overwrite=True)
    if done:
        open(os.path.join(d, Config.Tng.DONE_MARKER), "w").close()


def test_list_tng_galaxies(tmp_path):
    tng = str(tmp_path)
    _write_fake_galaxy(tng, "111")
    _write_fake_galaxy(tng, "222")
    _write_fake_galaxy(tng, "333", done=False)        # no .done → excluded
    gals = list_tng_galaxies(tng)
    assert [g[1] for g in gals] == ["111", "222"]     # numeric sort
    assert all(os.path.isdir(g[0]) for g in gals)
    assert list_tng_galaxies(str(tmp_path / "nope")) == []



def test_vis_brightness_normalization_preserves_tng_band_ratios(tmp_path):
    tng = str(tmp_path)
    _write_fake_galaxy(tng, "111")
    galaxy_dir = os.path.join(tng, "111")
    _, native = tng_stamp_at_redshift(
        galaxy_dir, "111", 1, 0.8, np.random.default_rng(9),
        sb_cut_mag_arcsec2=0.0,
    )
    target_vis = 2.5 * native["flux_e_per_band"]["VIS"]
    _, normalized = tng_stamp_at_redshift(
        galaxy_dir, "111", 1, 0.8, np.random.default_rng(9),
        sb_cut_mag_arcsec2=0.0,
        target_vis_flux_e=target_vis,
    )
    assert normalized["flux_e_per_band"]["VIS"] == pytest.approx(target_vis)
    for band in Config.LR_INPUT_BAND_NAMES[1:]:
        native_ratio = (
            native["flux_e_per_band"][band]
            / native["flux_e_per_band"]["VIS"]
        )
        normalized_ratio = (
            normalized["flux_e_per_band"][band]
            / normalized["flux_e_per_band"]["VIS"]
        )
        assert normalized_ratio == pytest.approx(native_ratio, rel=1e-6)


# --------------------- half-light radius / target sizing -------------------

def test_measure_halflight_radius_top_hat():
    # A filled disk of radius R has half-light radius R/√2.
    n = 201
    yy, xx = np.ogrid[:n, :n]
    r = np.hypot(yy - n // 2, xx - n // 2)
    R = 40.0
    frame = (r <= R).astype(np.float32)
    re = measure_halflight_radius_px(frame)
    assert abs(re - R / np.sqrt(2.0)) < 1.5      # ≈ 28.3 px


def test_measure_halflight_radius_empty_is_nan():
    assert math.isnan(measure_halflight_radius_px(np.zeros((10, 10))))
    # negatives don't count as light
    assert math.isnan(measure_halflight_radius_px(-np.ones((10, 10))))


def _fresh_radius_grid_cache(monkeypatch, *, max_bytes, max_seen=16):
    monkeypatch.setattr(
        skirt_image, "_RADIUS_INT_GRID",
        type(skirt_image._RADIUS_INT_GRID)(),
    )
    monkeypatch.setattr(
        skirt_image, "_RADIUS_INT_GRID_SEEN",
        type(skirt_image._RADIUS_INT_GRID_SEEN)(),
    )
    monkeypatch.setattr(skirt_image, "_RADIUS_INT_GRID_BYTES", 0)
    monkeypatch.setattr(skirt_image, "_RADIUS_INT_GRID_MAX_BYTES", max_bytes)
    monkeypatch.setattr(
        skirt_image, "_RADIUS_INT_GRID_SEEN_MAX_SHAPES", max_seen,
    )


def test_radius_grid_cache_is_byte_bounded_and_reuses_hot_shape(monkeypatch):
    _fresh_radius_grid_cache(monkeypatch, max_bytes=5_000, max_seen=4)

    first = skirt_image.radius_int_grid((16, 16))
    assert not skirt_image._RADIUS_INT_GRID
    second = skirt_image.radius_int_grid((16, 16))
    assert np.array_equal(first, second)
    assert second.nbytes == skirt_image._RADIUS_INT_GRID_BYTES
    assert skirt_image.radius_int_grid((16, 16)) is second

    # Reused shapes enter the LRU and force byte-based eviction. One-off shapes
    # only enter the small admission-history set, which is bounded separately.
    for side in range(17, 31):
        skirt_image.radius_int_grid((side, side))
        skirt_image.radius_int_grid((side, side))
    for side in range(31, 50):
        skirt_image.radius_int_grid((side, side))

    assert skirt_image._RADIUS_INT_GRID_BYTES <= 5_000
    assert sum(
        grid.nbytes for grid in skirt_image._RADIUS_INT_GRID.values()
    ) == skirt_image._RADIUS_INT_GRID_BYTES
    assert len(skirt_image._RADIUS_INT_GRID_SEEN) <= 4


def test_radius_grid_larger_than_budget_is_never_retained(monkeypatch):
    _fresh_radius_grid_cache(monkeypatch, max_bytes=1_024)

    expected = skirt_image.radius_int_grid((16, 16))
    actual = skirt_image.radius_int_grid((16, 16))

    assert np.array_equal(actual, expected)
    assert actual.nbytes > skirt_image._RADIUS_INT_GRID_MAX_BYTES
    assert not skirt_image._RADIUS_INT_GRID
    assert skirt_image._RADIUS_INT_GRID_BYTES == 0


def test_sample_tng_stamp_with_target_size_records_meta(tmp_path):
    # The sampled Euclid radius is a nominal continuous-space similarity scale.
    tng = str(tmp_path)
    _write_fake_galaxy(tng, "111", size=240)        # measurable core
    gals = list_tng_galaxies(tng)
    native = measure_halflight_radius_px(skirt_image.load_skirt_frame(
        tng_fits_path(gals[0][0], "111", 1, "VIS")
    ))
    radius_lookup = {("111", orientation): native for orientation in range(1, 6)}
    res = sample_tng_stamp(
        gals, np.random.default_rng(0), target_re_arcsec=1.0,
        radius_lookup_map=radius_lookup,
    )
    assert res is not None
    _stamp, meta = res
    assert "target_re_arcsec" in meta and meta["target_re_arcsec"] == 1.0
    assert meta["nominal_re_arcsec"] == pytest.approx(1.0)
    assert "native_halflight_px" in meta
    assert meta["radius_remeasured"] is False
    assert "achieved_re_arcsec" not in meta
