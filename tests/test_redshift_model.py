"""Redshift realism for TNG stamps: distances → downsample factor, Tolman
dimming + randomized spectral drift, mass → σ_v for TNG-lit lenses, and the
θ_E ≥ κ·R_e lens-visibility constraint."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.sky.redshift_model import (
    PIVOT_WAVELENGTH_UM,
    TNG_NATIVE_PC_PER_PIXEL,
    angular_diameter_distance,
    band_drift_factors,
    load_tng_properties,
    physical_pc_to_arcsec,
    rebin_factor_for_redshift,
    sample_galaxy_redshift,
    sigma_v_from_stellar_mass,
    tolman_dimming_factor,
)
from euclid_polish.sky.tng_galaxy import (
    sample_tng_stamp,
    tng_stamp_at_redshift,
)
from euclid_polish.sky.multiband_generator import (
    MultiBandGeneratorConfig,
    MultiBandSimulator,
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

def test_redshift_sampler_range_and_median():
    rng = np.random.default_rng(42)
    zs = np.array([sample_galaxy_redshift(rng) for _ in range(4000)])
    assert zs.min() >= Config.TNG_Z_MIN
    assert zs.max() <= Config.TNG_Z_MAX
    # n(z) ∝ z² exp(-(z/0.65)^1.5) → median around z≈0.9.
    assert 0.6 < float(np.median(zs)) < 1.2


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
    stamp, meta = tng_stamp_at_redshift(gdir, "111", 1, z, rng=None)
    expected_rebin = int(round(rebin_factor_for_redshift(z)))
    assert meta["rebin_factor"] == expected_rebin
    assert stamp.shape == (24 // expected_rebin, 24 // expected_rebin, 4)
    assert meta["z"] == z
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
            meta["native_halflight_px"] * TNG_NATIVE_PC_PER_PIXEL, z),
        rel=1e-6)


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

def _z_mode_sim(tmp_path, *, lens_density=0.0, tng_fraction=1.0):
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    _write_fake_tng_galaxy(tng, "222")
    csv_path = str(tmp_path / "tng_properties.csv")
    _write_props_csv(csv_path, [("111", 2.0e11), ("222", 1.0e11)])
    cat = TinyCosmosCatalog(n_galaxies=400, seed=0)
    cfg = MultiBandGeneratorConfig(
        image_size=64, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        lens_density_arcmin2=lens_density,
        tng_fraction=tng_fraction, tng_galaxy_dir=tng,
        tng_redshift_mode=True, tng_properties_csv=csv_path)
    return MultiBandSimulator(cat, cfg)


def test_generator_z_mode_field_galaxies(tmp_path):
    sim = _z_mode_sim(tmp_path)
    img, meta = sim.simulate_field(np.random.default_rng(1),
                                   n_galaxies=5, n_stars=0, n_lenses=0,
                                   n_big=0)
    assert img.data.sum() > 0
    recs = meta["galaxies"]
    assert [r["render"] for r in recs] == ["tng"] * 5
    for r in recs:
        assert Config.TNG_Z_MIN <= r["z"] <= Config.TNG_Z_MAX
        assert r["rebin_factor"] >= 1
        assert np.isfinite(r["drift_eps"])
        # z-mode replaces the COSMOS target-size draw entirely.
        assert math.isnan(r["target_re_arcsec"])


def test_z_mode_has_no_big_galaxy_population(tmp_path):
    # The realistic n(z) already yields big nearby galaxies — the separate
    # fixed-density "big" population is legacy-only, even when explicitly
    # requested via n_big.
    sim = _z_mode_sim(tmp_path)
    _, meta = sim.simulate_field(np.random.default_rng(2),
                                 n_galaxies=2, n_stars=0, n_lenses=0,
                                 n_big=4)
    assert meta["n_big_galaxies"] == 0
    assert not any(r["big"] for r in meta["galaxies"])


def test_generator_z_mode_lens_mass_and_visibility(tmp_path):
    sim = _z_mode_sim(tmp_path, lens_density=1.0)
    rng = np.random.default_rng(3)
    tng_lenses = []
    for _ in range(40):
        _, meta = sim.simulate_field(rng, n_galaxies=0, n_stars=0,
                                     n_lenses=1, n_big=0)
        tng_lenses += [r for r in meta["lenses"]
                       if r["lens_light_render"] == "tng"]
        if len(tng_lenses) >= 5:
            break
    assert tng_lenses, "no TNG-lit lens drawn in 40 fields at tng_fraction=1"
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


def test_pure_tng_mode_forces_redshift_mode(tmp_path):
    sim = _z_mode_sim(tmp_path)          # tng_fraction=1
    assert sim.pure_tng
    assert sim.config.tng_redshift_mode
    assert sim.lens_population is None   # catalog-backed priors unused


def test_pure_tng_mode_works_without_catalog(tmp_path):
    # tng_fraction=1 never renders anything Sersic, so COSMOS is optional:
    # field galaxies, stars AND lens systems all come out of catalog=None.
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    _write_fake_tng_galaxy(tng, "222")
    csv_path = str(tmp_path / "tng_properties.csv")
    _write_props_csv(csv_path, [("111", 2.0e11), ("222", 1.0e11)])
    cfg = MultiBandGeneratorConfig(
        image_size=64, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        lens_density_arcmin2=1.0, tng_fraction=1.0, tng_galaxy_dir=tng,
        tng_properties_csv=csv_path)
    sim = MultiBandSimulator(None, cfg)
    rng = np.random.default_rng(5)
    lenses = []
    for _ in range(20):
        img, meta = sim.simulate_field(rng, n_galaxies=3, n_stars=1,
                                       n_lenses=1, n_big=0)
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


def test_catalog_none_requires_pure_tng(tmp_path):
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    with pytest.raises(ValueError, match="pure-TNG"):
        MultiBandSimulator(None, MultiBandGeneratorConfig(
            tng_fraction=0.5, tng_galaxy_dir=tng))
    # No downloaded galaxies → not pure either, whatever the fraction.
    with pytest.raises(ValueError, match="pure-TNG"):
        MultiBandSimulator(None, MultiBandGeneratorConfig(
            tng_fraction=1.0, tng_galaxy_dir=str(tmp_path / "empty")))


def test_sample_lens_geometry_priors():
    from euclid_polish.sky.lens_population import sample_lens_geometry
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


def test_z_mode_off_keeps_legacy_metadata(tmp_path):
    # Fractional tng_fraction without tng_redshift_mode: the legacy
    # COSMOS-target-size path is untouched (tng_fraction=1 would force
    # pure/redshift mode instead).
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    cat = TinyCosmosCatalog(n_galaxies=200, seed=0)
    cfg = MultiBandGeneratorConfig(
        image_size=64, pixel_scale=Config.DEFAULT_PIXEL_SCALE,
        lens_density_arcmin2=0.0, tng_fraction=0.9, tng_galaxy_dir=tng)
    sim = MultiBandSimulator(cat, cfg)
    assert not sim.pure_tng and not sim.config.tng_redshift_mode
    _, meta = sim.simulate_field(np.random.default_rng(0),
                                 n_galaxies=8, n_stars=0, n_lenses=0)
    tng_recs = [r for r in meta["galaxies"] if r["render"] == "tng"]
    assert tng_recs
    for r in tng_recs:
        assert math.isnan(r["z"])                       # no redshift assigned
        assert np.isfinite(r["target_re_arcsec"])       # legacy sizing intact