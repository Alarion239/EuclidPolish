"""Redshift realism for TNG stamps: distances → downsample factor, Tolman
dimming + randomized spectral drift, mass → σ_v for TNG-lit lenses, and the
θ_E ≥ κ·R_e lens-visibility constraint."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.sky.generation.redshift_model import (
    PIVOT_WAVELENGTH_UM,
    TNG_NATIVE_PC_PER_PIXEL,
    angular_diameter_distance,
    band_drift_factors,
    compactness_factor,
    load_tng_properties,
    physical_pc_to_arcsec,
    predicted_vis_mag,
    rebin_factor_for_redshift,
    sample_galaxy_redshift,
    sample_target_logmass,
    sigma_v_from_stellar_mass,
    tolman_dimming_factor,
)
from euclid_polish.sky.generation.sky_simulator import (
    SkySimulator,
    SkySimulatorConfig,
)
from euclid_polish.sky.generation.tng_galaxy import (
    native_halflight_px,
    sample_tng_stamp,
    tng_stamp_at_redshift,
)
from tests._tiny_catalog import TinyCosmosCatalog

# ---------------------------------------------------------------------------
# Cosmological geometry → downsample factor
# ---------------------------------------------------------------------------

def test_angular_diameter_distance_reference_values():
    # Flat ΛCDM H0=70, Ωm=0.3: textbook checkpoints.
    assert angular_diameter_distance(0.5) == pytest.approx(1255.0, rel=0.02)
    assert angular_diameter_distance(1.0) == pytest.approx(1650.0, rel=0.02)


def test_rebin_factor_tracks_distance():
    f05 = rebin_factor_for_redshift(0.5)
    f10 = rebin_factor_for_redshift(1.0)
    assert 2.8 < f05 < 3.3        # ≈3.0 at z=0.5 on the 0.05″ grid
    assert 3.7 < f10 < 4.3        # ≈4.0 at z=1.0
    assert f05 < f10


def test_rebin_factor_floors_at_one_nearby():
    # Below z≈0.10 the native 100 pc pixel subtends more than 0.05″.
    assert rebin_factor_for_redshift(0.05) == 1.0


def test_rebin_factor_turns_over_with_da():
    # D_A peaks near z≈1.6: very distant galaxies stop shrinking.
    f16 = rebin_factor_for_redshift(1.6)
    f30 = rebin_factor_for_redshift(3.0)
    assert f30 < f16
    assert f16 < 4.6


def test_physical_pc_to_arcsec():
    # An 8 kpc giant at z=0.5 spans ≈1.3″.
    assert physical_pc_to_arcsec(8000.0, 0.5) == pytest.approx(1.31, rel=0.05)


# ---------------------------------------------------------------------------
# Field n(z)
# ---------------------------------------------------------------------------

def test_redshift_sampler_volume_form_range_median_and_low_z_tail():
    rng = np.random.default_rng(42)
    zs = np.array([sample_galaxy_redshift(rng) for _ in range(6000)])
    assert zs.min() >= Config.TNG_Z_MIN
    assert zs.max() <= Config.TNG_Z_MAX
    # dN/dz ∝ dV_c/dz · exp(-(z/1.5)²) → median ≈ 1.15; the atlas's massive
    # galaxies rarely sit nearby, so arcsec-scale giants stay rare.
    assert 1.0 < float(np.median(zs)) < 1.35
    assert (zs < 0.4).mean() < 0.07


def test_redshift_sampler_smail_form():
    rng = np.random.default_rng(42)
    zs = np.array([sample_galaxy_redshift(rng, form="smail")
                   for _ in range(4000)])
    # Full flux-limited population (Smail+ 1995): median ≈ 0.9 — kept as an
    # option; over-draws low z for the massive-only atlas.
    assert 0.7 < float(np.median(zs)) < 1.1


def test_redshift_sampler_unknown_form_raises():
    with pytest.raises(ValueError, match="form"):
        sample_galaxy_redshift(np.random.default_rng(0), form="nope")


# ---------------------------------------------------------------------------
# Dimming + spectral drift
# ---------------------------------------------------------------------------

def test_tolman_dimming():
    assert tolman_dimming_factor(0.0) == 1.0
    assert tolman_dimming_factor(1.0) == pytest.approx(0.125)


def test_drift_identity_at_z0():
    factors, meta = band_drift_factors([1.0, 2.0, 3.0, 4.0], 0.0, rng=None)
    assert meta["drift_mode"] == "sed_interp"
    np.testing.assert_allclose(factors, np.ones(4), rtol=1e-12)


def test_drift_flat_sed_is_pure_dimming():
    factors, _ = band_drift_factors([2.0] * 4, 0.7, rng=None)
    np.testing.assert_allclose(factors, tolman_dimming_factor(0.7), rtol=1e-9)


def test_drift_red_sed_suppresses_blue_bands_most():
    # A red continuum (f_ν rising with λ) sampled bluer at z>0: every band
    # dims, VIS most, H least → colours drift red.
    factors, meta = band_drift_factors(
        [1.0, 2.0, 3.0, 4.0], 0.5, rng=None, include_dimming=False)
    assert meta["drift_mode"] == "sed_interp"
    assert np.all(factors < 1.0)
    assert factors[0] < factors[3]
    assert factors.argmax() == 3             # H least affected (red drift)


def test_drift_parametric_fallback_on_bad_sed():
    factors, meta = band_drift_factors(
        [0.0, 2.0, 3.0, 4.0], 0.5, rng=None, include_dimming=False)
    assert meta["drift_mode"] == "parametric"
    assert factors[0] < factors[3] == pytest.approx(1.0)   # anchored at H


def test_drift_stochastic_tilt_reproducible_and_red_biased_spread():
    sed = [1.0, 2.0, 3.0, 4.0]
    f1, m1 = band_drift_factors(sed, 0.8, np.random.default_rng(7))
    f2, m2 = band_drift_factors(sed, 0.8, np.random.default_rng(7))
    np.testing.assert_allclose(f1, f2)
    assert m1["drift_eps"] == m2["drift_eps"] != 0.0
    # ε is zero-mean around the SED's own response: both signs must occur.
    rng = np.random.default_rng(3)
    eps = [band_drift_factors(sed, 0.8, rng)[1]["drift_eps"]
           for _ in range(200)]
    assert min(eps) < 0.0 < max(eps)


def test_pivot_wavelengths_monotone():
    assert list(PIVOT_WAVELENGTH_UM) == sorted(PIVOT_WAVELENGTH_UM)


# ---------------------------------------------------------------------------
# Mass → σ_v (Faber–Jackson) + property catalog
# ---------------------------------------------------------------------------

def test_sigma_v_faber_jackson_reference_points():
    assert sigma_v_from_stellar_mass(1.0e11) == pytest.approx(200.0)
    # Most massive subhalo in the local catalog (≈3.9e11 M☉) → ≈300 km/s.
    assert sigma_v_from_stellar_mass(3.95e11) == pytest.approx(302.0, rel=0.02)
    assert sigma_v_from_stellar_mass(1.0e15) == Config.LENS_SIGMA_V_CLIP_KMS[1]
    assert math.isnan(sigma_v_from_stellar_mass(float("nan")))
    assert math.isnan(sigma_v_from_stellar_mass(-5.0))


def test_sigma_v_scatter_uses_rng():
    vals = {sigma_v_from_stellar_mass(1.0e11, np.random.default_rng(s))
            for s in range(8)}
    assert len(vals) > 1
    assert all(100.0 <= v <= 400.0 for v in vals)


def _write_props_csv(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("id,sfr,mass_stars,m_halo,reff\n")
        for gid, mstar in rows:
            f.write(f"{gid},0.1,{mstar},1e12,3.0\n")


def test_load_tng_properties(tmp_path):
    csv_path = str(tmp_path / "tng_properties.csv")
    _write_props_csv(csv_path, [("111", 1.0e11), ("222", 5.0e10)])
    props = load_tng_properties(csv_path)
    assert props["111"]["mass_stars"] == pytest.approx(1.0e11)
    assert props["222"]["reff"] == pytest.approx(3.0)
    assert load_tng_properties(str(tmp_path / "missing.csv")) == {}


def test_load_tng_properties_real_repo_csv():
    path = os.path.join("data", "_tng_infographics", "tng_properties.csv")
    if not os.path.isfile(path):
        pytest.skip("local tng_properties.csv not present")
    props = load_tng_properties(path)
    assert len(props) > 100
    masses = [p["mass_stars"] for p in props.values()
              if np.isfinite(p.get("mass_stars", float("nan")))]
    assert all(m > 1e8 for m in masses)


# ---------------------------------------------------------------------------
# TNG stamps at redshift (synthetic atlas)
# ---------------------------------------------------------------------------

def _write_fake_tng_galaxy(tng_dir, gid, *, size=24):
    from astropy.io import fits
    d = os.path.join(tng_dir, gid)
    os.makedirs(d, exist_ok=True)
    for o in (1, 2, 3, 4, 5):
        for b in ("VIS", "Y", "J", "H"):
            arr = np.zeros((size, size), dtype=">f4")
            arr[size // 2 - 2:size // 2 + 2,
                size // 2 - 2:size // 2 + 2] = 500.0
            fits.PrimaryHDU(arr).writeto(
                os.path.join(d, f"TNG{gid}_O{o}_Euclid_{b}.fits"),
                overwrite=True)
    open(os.path.join(d, Config.Tng.DONE_MARKER), "w").close()


def test_tng_stamp_at_redshift_size_and_dimming(tmp_path):
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111", size=24)
    gdir = os.path.join(tng, "111")

    z = 0.5
    native_re_px = native_halflight_px(gdir, "111", 1)
    stamp, meta = tng_stamp_at_redshift(
        gdir, "111", 1, z, rng=None, native_re_px=native_re_px,
    )
    expected_rebin = int(round(
        rebin_factor_for_redshift(z) * compactness_factor(z)))
    assert meta["rebin_factor"] == expected_rebin
    assert stamp.shape == (24 // expected_rebin, 24 // expected_rebin, 4)
    assert meta["z"] == z
    assert meta["compactness"] == pytest.approx(compactness_factor(z))
    # Flat fake SED (equal MJy/sr in all bands) → in f_ν the four bands are
    # NOT equal (per-band zeropoints differ), but the drift on the stamp's
    # own SED is deterministic with rng=None; every factor must include the
    # (1+z)⁻³ dimming, i.e. be well below 1.
    factors = np.asarray(meta["redshift_band_factors"])
    assert np.all(factors < 0.7)
    assert np.all(factors > 0.0)
    assert meta["dimming"] == pytest.approx(tolman_dimming_factor(z))
    assert meta["apparent_re_arcsec"] == pytest.approx(
        physical_pc_to_arcsec(
            meta["native_halflight_px"] * TNG_NATIVE_PC_PER_PIXEL, z)
        / compactness_factor(z), rel=1e-6)


def test_compactness_factor_size_evolution():
    assert compactness_factor(0.0) == pytest.approx(Config.TNG_COMPACT_C0)
    assert compactness_factor(1.0) == pytest.approx(
        Config.TNG_COMPACT_C0 * 2.0 ** Config.TNG_COMPACT_BETA)
    assert compactness_factor(1.0, c0=1.0, beta=0.0) == 1.0


def test_compactness_squeeze_conserves_flux(tmp_path):
    # The squeeze shrinks the stamp but must keep the total flux pinned to
    # the continuous geometric prediction (SB x C^2 boost): same luminosity,
    # smaller radius. Fake stamp: bright core only, so truncation is neutral.
    from unittest import mock

    from astropy.io import fits
    tng = str(tmp_path / "tng")
    d = os.path.join(tng, "888")
    os.makedirs(d)
    for b in ("VIS", "Y", "J", "H"):
        arr = np.zeros((96, 96), dtype=">f4")
        arr[40:56, 40:56] = 300.0
        fits.PrimaryHDU(arr).writeto(
            os.path.join(d, f"TNG888_O1_Euclid_{b}.fits"))
    open(os.path.join(d, Config.Tng.DONE_MARKER), "w").close()

    z = 0.5
    squeezed, ms = tng_stamp_at_redshift(d, "888", 1, z, rng=None)
    with mock.patch("euclid_polish.sky.generation.tng_galaxy.compactness_factor",
                    lambda z, **k: 1.0):
        plain, mp = tng_stamp_at_redshift(d, "888", 1, z, rng=None)
    assert ms["rebin_factor"] > mp["rebin_factor"]      # more compact
    assert ms["flux_e_per_band"]["VIS"] == pytest.approx(
        mp["flux_e_per_band"]["VIS"], rel=0.05)          # same total light


def test_tng_stamp_sb_truncation_crops_faint_outskirts(tmp_path):
    from astropy.io import fits

    from euclid_polish.sky.generation.tng_galaxy import tng_stamp_at_redshift
    # Bright 4-px core + a whole box of ultra-faint "outskirts": the wings
    # sit far below the mu=28 cut, so the stamp must crop to the core.
    tng = str(tmp_path / "tng")
    d = os.path.join(tng, "777")
    os.makedirs(d)
    for o in (1,):
        for b in ("VIS", "Y", "J", "H"):
            arr = np.full((96, 96), 1e-12, dtype=">f4")     # faint everywhere
            arr[46:50, 46:50] = 500.0                       # bright core
            fits.PrimaryHDU(arr).writeto(
                os.path.join(d, f"TNG777_O{o}_Euclid_{b}.fits"))
    open(os.path.join(d, Config.Tng.DONE_MARKER), "w").close()

    stamp, meta = tng_stamp_at_redshift(d, "777", 1, 0.5, rng=None)
    full = 96 // meta["rebin_factor"]
    assert stamp.shape[0] < full                  # cropped below the full box
    assert stamp.shape == meta["shape"]
    assert meta["flux_e_per_band"]["VIS"] > 0     # the core survives
    # Disabling the cut keeps the full box.
    stamp_full, _ = tng_stamp_at_redshift(d, "777", 1, 0.5, rng=None,
                                          sb_cut_mag_arcsec2=0.0)
    assert stamp_full.shape[0] == full


def test_mass_rescale_dims_and_shrinks(tmp_path):
    # mass_scale s: flux x s (L ∝ M), size / s^alpha — a smaller galaxy of
    # similar morphology, NOT a flux-conserving squeeze.
    from astropy.io import fits

    from euclid_polish.sky.generation.tng_galaxy import tng_stamp_at_redshift
    tng = str(tmp_path / "tng")
    d = os.path.join(tng, "999")
    os.makedirs(d)
    for b in ("VIS", "Y", "J", "H"):
        arr = np.zeros((96, 96), dtype=">f4")
        arr[40:56, 40:56] = 300.0
        fits.PrimaryHDU(arr).writeto(os.path.join(d, f"TNG999_O1_Euclid_{b}.fits"))
    open(os.path.join(d, Config.Tng.DONE_MARKER), "w").close()

    full, mf = tng_stamp_at_redshift(d, "999", 1, 0.5, rng=None)
    small, ms = tng_stamp_at_redshift(d, "999", 1, 0.5, rng=None,
                                      mass_scale=0.25)
    assert ms["mass_scale"] == 0.25
    assert ms["rebin_factor"] >= mf["rebin_factor"]
    assert ms["flux_e_per_band"]["VIS"] == pytest.approx(
        0.25 * mf["flux_e_per_band"]["VIS"], rel=0.1)
    with pytest.raises(ValueError, match="mass_scale"):
        tng_stamp_at_redshift(d, "999", 1, 0.5, rng=None, mass_scale=1.5)


@pytest.mark.skip(reason="superseded by the joint COSMOS population prior")
def test_z_mode_field_galaxies_draw_mass_scale(tmp_path):
    # MF-weighted rescaling: the fake atlas holds only 1-2e11 subhalos while
    # the Schechter target is mostly ~1e9-1e10, so the drawn mass scales are
    # small and varied.
    sim = _z_mode_sim(tmp_path)
    _, meta = sim.simulate_field(np.random.default_rng(9), n_tng=6,
                                 n_sersic=0, n_stars=0, n_lenses=0)
    s = [r["mass_scale"] for r in meta["galaxies"]]
    assert all(0.0 < v <= 1.0 for v in s)
    assert len(set(s)) > 1                      # actually randomized
    assert min(s) < 0.5                         # small galaxies dominate


def test_predicted_vis_mag_faint_skip():
    # Pinned to the measured pipeline anchor; dwarfs at high z fall beyond
    # the skip cut, bright nearby ones stay well inside it.
    assert predicted_vis_mag(10.55, 0.5) == pytest.approx(21.36, abs=0.01)
    assert predicted_vis_mag(9.0, 2.0) > Config.TNG_FAINT_SKIP_MAG_VIS
    assert predicted_vis_mag(10.5, 0.3) < 22.0


@pytest.mark.skip(reason="superseded by the joint COSMOS population prior")
def test_target_logmass_schechter():
    rng = np.random.default_rng(5)
    lm = np.array([sample_target_logmass(rng) for _ in range(6000)])
    assert lm.min() >= Config.TNG_MF_LOGM_MIN
    assert lm.max() <= Config.TNG_MF_LOGM_MAX
    # The smooth data-calibrated slope strongly favors faint/small targets;
    # giants remain a continuous rare tail rather than a separate component.
    assert 8.7 < float(np.median(lm)) < 9.1
    assert (lm > 10.5).mean() < 0.05
    assert (lm > 11.0).mean() < 0.015
    assert pytest.approx(-1.76) == Config.TNG_MF_ALPHA
    assert pytest.approx(100.0) == Config.TNG_GAL_DENSITY_ARCMIN2


def test_sample_tng_stamp_z_mode(tmp_path):
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    galaxies = [(os.path.join(tng, "111"), "111")]
    res = sample_tng_stamp(galaxies, np.random.default_rng(0), z=1.0)
    assert res is not None
    stamp, meta = res
    assert meta["z"] == 1.0
    assert meta["rebin_factor"] >= 3
    assert stamp.ndim == 3 and stamp.shape[2] == 4


# ---------------------------------------------------------------------------
# Generator integration: redshift mode
# ---------------------------------------------------------------------------

def _z_mode_sim(tmp_path, *, lens_density=0.0, sersic_density=0.0):
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    _write_fake_tng_galaxy(tng, "222")
    csv_path = str(tmp_path / "tng_properties.csv")
    _write_props_csv(csv_path, [("111", 2.0e11), ("222", 1.0e11)])
    cat = TinyCosmosCatalog(n_galaxies=400, seed=0) if sersic_density > 0.0 else None
    cfg = SkySimulatorConfig(
        image_size=64, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        lens_density_arcmin2=lens_density,
        sersic_density_arcmin2=sersic_density,
        tng_density_arcmin2=Config.DEFAULT_GAL_DENSITY_ARCMIN2,
        tng_galaxy_dir=tng,
        tng_redshift_mode=True, tng_properties_csv=csv_path)
    return SkySimulator(cat, cfg)


@pytest.mark.skip(reason="superseded by the joint COSMOS population prior")
def test_generator_z_mode_field_galaxies(tmp_path):
    sim = _z_mode_sim(tmp_path)
    img, meta = sim.simulate_field(np.random.default_rng(1),
                                   n_tng=5, n_sersic=0, n_stars=0, n_lenses=0)
    assert img.data.sum() > 0
    recs = meta["galaxies"]
    # Undetectably faint draws (predicted m_VIS > the skip cut) drop their
    # slot, so up to 5 render — and every rendered one is a TNG stamp.
    assert 1 <= len(recs) <= 5
    assert all(r["render"] == "tng" for r in recs)
    for r in recs:
        assert Config.TNG_Z_MIN <= r["z"] <= Config.TNG_Z_MAX
        assert r["rebin_factor"] >= 1
        assert np.isfinite(r["drift_eps"])
        # z-mode replaces the log-uniform target-size draw entirely.
        assert math.isnan(r["target_re_arcsec"])


@pytest.mark.skip(reason="analytic Sersic generation was removed")
def test_z_mode_tng_only_no_sersic(tmp_path):
    # When sersic_density_arcmin2=0, all galaxy records are TNG stamps.
    sim = _z_mode_sim(tmp_path)
    _, meta = sim.simulate_field(np.random.default_rng(2),
                                 n_tng=4, n_sersic=0, n_stars=0, n_lenses=0)
    assert all(r["render"] == "tng" for r in meta["galaxies"])


@pytest.mark.skip(reason="superseded by the joint COSMOS population prior")
def test_generator_z_mode_lens_mass_and_visibility(tmp_path):
    sim = _z_mode_sim(tmp_path, lens_density=1.0)
    rng = np.random.default_rng(3)
    tng_lenses = []
    for _ in range(40):
        _, meta = sim.simulate_field(rng, n_tng=0, n_sersic=0, n_stars=0,
                                     n_lenses=1)
        tng_lenses += [r for r in meta["lenses"]
                       if r["lens_light_render"] == "tng"]
        if len(tng_lenses) >= 5:
            break
    assert tng_lenses, "no TNG-lit lens drawn in 40 fields with tng loaded"
    kappa = sim.config.lens_theta_e_min_re_ratio
    for r in tng_lenses:
        # σ_v derived from the subhalo's stellar mass (FJ, with scatter).
        assert np.isfinite(r["sigma_v_kms"])
        assert 100.0 <= r["sigma_v_kms"] <= 400.0
        assert any(r["lens_mstar_msun"] == pytest.approx(m, rel=1e-6)
                   for m in (2.0e11, 1.0e11))
        # Visibility: the Einstein radius clears the lens light.
        assert np.isfinite(r["lens_apparent_re_arcsec"])
        assert (r["theta_E_arcsec"]
                >= kappa * r["lens_apparent_re_arcsec"] - 1e-9)
        assert 0.10 < r["theta_E_arcsec"] < 3.5
        # Arc-prominence diagnostics for downstream showability cuts.
        assert r["lens_visible_r_arcsec"] > 0
        assert r["source_flux_vis_e"] >= 0


def test_analytic_showability_predictors(tmp_path):
    # The pre-render predictors must track the rendered stamp: flux within
    # ~10% (truncation losses ignored -> slight over-prediction), visible
    # radius below the rendered half-size (mean-profile approximation ->
    # permissive, the post-render check is the backstop).
    from astropy.io import fits

    from euclid_polish.sky.generation.tng_galaxy import (
        predict_vis_flux_e,
        predict_visible_radius_arcsec,
        tng_stamp_at_redshift,
    )
    tng = str(tmp_path / "tng")
    d = os.path.join(tng, "555")
    os.makedirs(d)
    for b in ("VIS", "Y", "J", "H"):
        arr = np.zeros((96, 96), dtype=">f4")
        arr[36:60, 36:60] = 200.0
        fits.PrimaryHDU(arr).writeto(os.path.join(d, f"TNG555_O1_Euclid_{b}.fits"))
    open(os.path.join(d, Config.Tng.DONE_MARKER), "w").close()

    z = 0.7
    stamp, meta = tng_stamp_at_redshift(d, "555", 1, z, rng=None)
    fpred = predict_vis_flux_e(d, "555", 1, z)
    assert fpred == pytest.approx(meta["flux_e_per_band"]["VIS"], rel=0.1)
    rpred = predict_visible_radius_arcsec(d, "555", 1, z)
    assert 0.0 < rpred <= stamp.shape[0] * 0.05 / 2 * 1.2


@pytest.mark.skip(reason="superseded by the joint COSMOS population prior")
def test_lens_require_showable_smoke(tmp_path):
    # With the flag on, lens systems are pre-filtered analytically and the
    # rendered record still satisfies the showable thresholds (the fake
    # stamps are compact and bright, so the cut is easy to pass).
    from scripts.fasrc_poster_cutout import _lens_is_showable
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    csv_path = str(tmp_path / "tng_properties.csv")
    _write_props_csv(csv_path, [("111", 2.0e11)])
    cfg = SkySimulatorConfig(
        image_size=64, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        sersic_density_arcmin2=0.0,
        tng_density_arcmin2=Config.DEFAULT_GAL_DENSITY_ARCMIN2,
        lens_density_arcmin2=1.0, tng_galaxy_dir=tng,
        tng_properties_csv=csv_path, lens_require_showable=True)
    sim = SkySimulator(None, cfg)
    rng = np.random.default_rng(4)
    for _ in range(30):
        _, meta = sim.simulate_field(rng, n_tng=0, n_sersic=0, n_stars=0,
                                     n_lenses=1)
        if meta["lenses"]:
            assert _lens_is_showable(meta["lenses"][0])
            break
    else:
        pytest.fail("no showable lens produced in 30 fields")


def test_poster_lens_showability_cut():
    from scripts.fasrc_poster_cutout import (
        LENS_MIN_SOURCE_VIS_E,
        LENS_MIN_THETA_E_VISIBLE_FRAC,
        _lens_is_showable,
    )
    base = {"theta_E_arcsec": 1.5, "lens_visible_r_arcsec": 2.0,
            "source_flux_vis_e": 5000.0}
    assert _lens_is_showable(base)
    # Arcs buried inside the deflector light → rejected.
    assert not _lens_is_showable({**base, "lens_visible_r_arcsec": 4.0})
    # Source dimmed into oblivion → rejected.
    assert not _lens_is_showable({**base, "source_flux_vis_e": 100.0})
    # Legacy/Sersic records (no diagnostics) pass through unchecked.
    assert _lens_is_showable({"theta_E_arcsec": 0.4})
    assert LENS_MIN_THETA_E_VISIBLE_FRAC > 0 and LENS_MIN_SOURCE_VIS_E > 0


@pytest.mark.skip(reason="catalog-backed lens population was removed")
def test_tng_only_has_no_lens_population(tmp_path):
    # catalog=None → no catalog-backed lens priors.
    sim = _z_mode_sim(tmp_path)
    assert sim.config.tng_redshift_mode
    assert sim.lens_population is None   # catalog-backed priors unused


@pytest.mark.skip(reason="analytic Sersic generation was removed")
def test_mixed_tng_sersic_in_z_mode(tmp_path):
    # With sersic_density > 0 AND tng_density > 0, both populations appear.
    sim = _z_mode_sim(tmp_path, sersic_density=5.0)
    _, meta = sim.simulate_field(np.random.default_rng(7), n_tng=2, n_sersic=4,
                                 n_stars=0, n_lenses=0)
    renders = {g["render"] for g in meta["galaxies"]}
    assert "tng" in renders
    assert "sersic" in renders


@pytest.mark.skip(reason="superseded by galaxy_density_arcmin2")
def test_tng_density_drives_poisson_count(tmp_path):
    # The explicit tng_density_arcmin2 drives the TNG Poisson rate.
    sim = _z_mode_sim(tmp_path)
    lams = []

    class SpyRng:
        def __init__(self, rng): self._r = rng
        def poisson(self, lam):
            lams.append(float(lam))
            return self._r.poisson(lam)
        def __getattr__(self, name): return getattr(self._r, name)

    sim.simulate_field(SpyRng(np.random.default_rng(0)))
    area = sim._field_area_arcmin2()
    # First Poisson draw is the Sersic count (= 0 × area = 0), second is TNG.
    assert len(lams) >= 2
    assert lams[0] == pytest.approx(0.0)  # sersic_density=0
    assert lams[1] == pytest.approx(sim.config.tng_density_arcmin2 * area)


@pytest.mark.skip(reason="a joint COSMOS population prior is now required")
def test_tng_only_works_without_catalog(tmp_path):
    # sersic_density=0 never renders anything Sersic, so COSMOS is not needed:
    # field galaxies, stars AND lens systems all come out of catalog=None.
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    _write_fake_tng_galaxy(tng, "222")
    csv_path = str(tmp_path / "tng_properties.csv")
    _write_props_csv(csv_path, [("111", 2.0e11), ("222", 1.0e11)])
    cfg = SkySimulatorConfig(
        image_size=64, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        sersic_density_arcmin2=0.0,
        tng_density_arcmin2=Config.DEFAULT_GAL_DENSITY_ARCMIN2,
        lens_density_arcmin2=1.0, tng_galaxy_dir=tng,
        tng_properties_csv=csv_path)
    sim = SkySimulator(None, cfg)
    rng = np.random.default_rng(5)
    lenses = []
    for _ in range(20):
        img, meta = sim.simulate_field(rng, n_tng=3, n_sersic=0, n_stars=1,
                                       n_lenses=1)
        assert img.data.sum() > 0
        assert all(r["render"] == "tng" for r in meta["galaxies"])
        lenses += meta["lenses"]
        if lenses:
            break
    assert lenses, "no lens system rendered in 20 catalog-free fields"
    L = lenses[0]
    assert L["lens_light_render"] == "tng"
    assert L["source_render"] == "tng"
    assert L["lens_subhalo_id"] in ("111", "222")
    assert Config.LENS_Z_LENS_MIN <= L["z_lens"] <= Config.LENS_Z_LENS_MAX
    assert L["z_source"] >= L["z_lens"] + Config.LENS_Z_SOURCE_OFFSET
    assert (L["theta_E_arcsec"]
            >= sim.config.lens_theta_e_min_re_ratio
            * L["lens_apparent_re_arcsec"] - 1e-9)


@pytest.mark.skip(reason="analytic Sersic generation was removed")
def test_catalog_none_requires_zero_sersic_density(tmp_path):
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    # Default sersic_density_arcmin2 > 0 → must provide a catalog.
    with pytest.raises(ValueError, match="sersic_density_arcmin2"):
        SkySimulator(None, SkySimulatorConfig(
            tng_density_arcmin2=1.0, tng_galaxy_dir=tng))


def test_sample_lens_geometry_priors():
    from euclid_polish.sky.generation.lens_population import sample_lens_geometry
    rng = np.random.default_rng(11)
    for _ in range(20):
        lp = sample_lens_geometry(rng, 250.0)
        assert lp is not None
        assert lp.lens_galaxy is None and lp.source_galaxy is None
        assert Config.LENS_Z_LENS_MIN <= lp.z_lens <= Config.LENS_Z_LENS_MAX
        assert (lp.z_lens + Config.LENS_Z_SOURCE_OFFSET
                <= lp.z_source <= Config.LENS_Z_SOURCE_MAX)
        assert 0.10 < lp.theta_E_arcsec < 3.5
        assert Config.LENS_AXIS_RATIO_MIN <= lp.lens_q <= Config.LENS_AXIS_RATIO_MAX
        r = math.hypot(lp.src_dx_arcsec, lp.src_dy_arcsec)
        assert r <= Config.LENS_SOURCE_OFFSET_FRAC * lp.theta_E_arcsec + 1e-12


@pytest.mark.skip(reason="redshift and size now come from the COSMOS row")
def test_tng_without_redshift_mode_uses_log_uniform_re(tmp_path):
    # tng_redshift_mode=False: R_e is drawn log-uniformly from tng_re_arcsec_range;
    # no z is assigned (nan) and target_re_arcsec is the sampled value.
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    cat = TinyCosmosCatalog(n_galaxies=200, seed=0)
    cfg = SkySimulatorConfig(
        image_size=64, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        lens_density_arcmin2=0.0, tng_density_arcmin2=1.0,
        tng_redshift_mode=False, tng_galaxy_dir=tng)
    sim = SkySimulator(cat, cfg)
    assert not sim.config.tng_redshift_mode
    _, meta = sim.simulate_field(np.random.default_rng(0),
                                 n_tng=8, n_sersic=0, n_stars=0, n_lenses=0)
    tng_recs = [r for r in meta["galaxies"] if r["render"] == "tng"]
    assert tng_recs
    lo, hi = cfg.tng_re_arcsec_range
    for r in tng_recs:
        assert math.isnan(r["z"])                             # no redshift assigned
        assert np.isfinite(r["target_re_arcsec"])             # log-uniform sizing
        assert lo <= r["target_re_arcsec"] <= hi
