"""Redshift geometry, photometry, and typed TNG-rendering contracts."""

from __future__ import annotations

import math
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from astropy.io import fits

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
    sigma_v_from_stellar_mass,
    tolman_dimming_factor,
)
from euclid_polish.sky.generation.tng_galaxy import TNGRenderer
from euclid_polish.sky.generation.tng_types import PhysicalRedshiftGeometry, TNGView


def test_angular_diameter_distance_reference_values():
    assert angular_diameter_distance(0.5) == pytest.approx(1255.0, rel=0.02)
    assert angular_diameter_distance(1.0) == pytest.approx(1650.0, rel=0.02)


def test_rebin_factor_tracks_distance_and_turns_over():
    factor_05 = rebin_factor_for_redshift(0.5)
    factor_10 = rebin_factor_for_redshift(1.0)
    assert 2.8 < factor_05 < 3.3
    assert 3.7 < factor_10 < 4.3
    assert factor_05 < factor_10
    assert rebin_factor_for_redshift(0.05) == 1.0
    assert rebin_factor_for_redshift(3.0) < rebin_factor_for_redshift(1.6) < 4.6


def test_physical_pc_to_arcsec():
    assert physical_pc_to_arcsec(8000.0, 0.5) == pytest.approx(1.31, rel=0.05)


def test_tolman_dimming():
    assert tolman_dimming_factor(0.0) == 1.0
    assert tolman_dimming_factor(1.0) == pytest.approx(0.125)


def test_drift_identity_at_z0_and_flat_sed_dimming():
    factors, metadata = band_drift_factors(
        [1.0, 2.0, 3.0, 4.0], 0.0, rng=None
    )
    assert metadata["drift_mode"] == "sed_interp"
    np.testing.assert_allclose(factors, np.ones(4), rtol=1e-12)
    flat, _ = band_drift_factors([2.0] * 4, 0.7, rng=None)
    np.testing.assert_allclose(flat, tolman_dimming_factor(0.7), rtol=1e-9)


def test_drift_red_sed_suppresses_blue_bands_most():
    factors, metadata = band_drift_factors(
        [1.0, 2.0, 3.0, 4.0],
        0.5,
        rng=None,
        include_dimming=False,
    )
    assert metadata["drift_mode"] == "sed_interp"
    assert np.all(factors < 1.0)
    assert factors[0] < factors[3]
    assert factors.argmax() == 3


def test_drift_parametric_fallback_on_bad_sed():
    factors, metadata = band_drift_factors(
        [0.0, 2.0, 3.0, 4.0],
        0.5,
        rng=None,
        include_dimming=False,
    )
    assert metadata["drift_mode"] == "parametric"
    assert factors[0] < factors[3] == pytest.approx(1.0)


def test_drift_stochastic_tilt_is_reproducible_and_two_sided():
    sed = [1.0, 2.0, 3.0, 4.0]
    first, first_meta = band_drift_factors(
        sed, 0.8, np.random.default_rng(7)
    )
    second, second_meta = band_drift_factors(
        sed, 0.8, np.random.default_rng(7)
    )
    np.testing.assert_allclose(first, second)
    assert first_meta["drift_eps"] == second_meta["drift_eps"] != 0.0
    rng = np.random.default_rng(3)
    epsilon = [
        band_drift_factors(sed, 0.8, rng)[1]["drift_eps"]
        for _ in range(200)
    ]
    assert min(epsilon) < 0.0 < max(epsilon)


def test_pivot_wavelengths_monotone():
    assert list(PIVOT_WAVELENGTH_UM) == sorted(PIVOT_WAVELENGTH_UM)


def test_sigma_v_faber_jackson_reference_points():
    assert sigma_v_from_stellar_mass(1.0e11) == pytest.approx(200.0)
    assert sigma_v_from_stellar_mass(3.95e11) == pytest.approx(302.0, rel=0.02)
    assert sigma_v_from_stellar_mass(1.0e15) == Config.LENS_SIGMA_V_CLIP_KMS[1]
    assert math.isnan(sigma_v_from_stellar_mass(float("nan")))
    assert math.isnan(sigma_v_from_stellar_mass(-5.0))


def test_sigma_v_scatter_uses_rng():
    values = {
        sigma_v_from_stellar_mass(1.0e11, np.random.default_rng(seed))
        for seed in range(8)
    }
    assert len(values) > 1
    assert all(100.0 <= value <= 400.0 for value in values)


def _write_props_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    lines = ["id,sfr,mass_stars,m_halo,reff"]
    lines.extend(
        f"{subhalo_id},0.1,{mass},1e12,3.0" for subhalo_id, mass in rows
    )
    path.write_text("\n".join(lines) + "\n")


def test_load_tng_properties_refreshes_when_file_changes(tmp_path):
    csv_path = tmp_path / "tng_properties.csv"
    _write_props_csv(csv_path, [("111", 1.0e11), ("222", 5.0e10)])
    properties = load_tng_properties(str(csv_path))
    assert properties["111"]["mass_stars"] == pytest.approx(1.0e11)
    assert properties["222"]["reff"] == pytest.approx(3.0)
    _write_props_csv(
        csv_path,
        [("111", 1.0e11), ("222", 5.0e10), ("333", 2.5e10)],
    )
    assert load_tng_properties(str(csv_path))["333"]["mass_stars"] == (
        pytest.approx(2.5e10)
    )
    assert load_tng_properties(str(tmp_path / "missing.csv")) == {}


def test_load_tng_properties_real_repo_csv():
    path = os.path.join("data", "_tng_infographics", "tng_properties.csv")
    if not os.path.isfile(path):
        pytest.skip("local tng_properties.csv not present")
    properties = load_tng_properties(path)
    assert len(properties) > 100
    masses = [
        row["mass_stars"]
        for row in properties.values()
        if np.isfinite(row.get("mass_stars", float("nan")))
    ]
    assert all(mass > 1e8 for mass in masses)


def _write_frame(path: Path, data: np.ndarray) -> None:
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=">f4"))
    hdu.header["BUNIT"] = "MJy/sr"
    hdu.header["CDELT1"] = 100.0
    hdu.header["CUNIT1"] = "pc"
    hdu.header["CDELT2"] = 100.0
    hdu.header["CUNIT2"] = "pc"
    hdu.writeto(path, overwrite=True)


def _write_fake_tng_galaxy(
    root: Path,
    subhalo_id: str,
    *,
    size: int = 96,
    faint_outskirts: bool = False,
) -> TNGView:
    directory = root / subhalo_id
    directory.mkdir(parents=True)
    base = np.full(
        (size, size), 1e-12 if faint_outskirts else 0.0, dtype=np.float32
    )
    centre = size // 2
    base[centre - 8 : centre + 8, centre - 8 : centre + 8] = 300.0
    for band in ("VIS", "Y", "J", "H"):
        _write_frame(
            directory / f"TNG{subhalo_id}_O1_Euclid_{band}.fits", base
        )
    (directory / Config.Tng.DONE_MARKER).touch()
    return TNGView(directory, subhalo_id, 1, native_re_px=8.0)


def test_physical_redshift_render_records_geometry_and_dimming(tmp_path):
    view = _write_fake_tng_galaxy(tmp_path, "111", size=96)
    renderer = TNGRenderer(pixel_scale_arcsec=0.05)
    redshift = 0.5

    rendered = renderer.render_for_physical_redshift(
        view,
        redshift,
        rng=None,
        surface_brightness_cut_mag_arcsec2=0.0,
    )
    geometry = rendered.trace.geometry

    assert isinstance(geometry, PhysicalRedshiftGeometry)
    assert geometry.rebin_factor == int(
        round(rebin_factor_for_redshift(redshift) * compactness_factor(redshift))
    )
    assert rendered.shape == (
        96 // geometry.rebin_factor,
        96 // geometry.rebin_factor,
        4,
    )
    assert rendered.trace.redshift is not None
    assert rendered.trace.redshift.dimming_factor == pytest.approx(
        tolman_dimming_factor(redshift)
    )
    assert geometry.apparent_re_arcsec == pytest.approx(
        physical_pc_to_arcsec(
            view.native_re_px * TNG_NATIVE_PC_PER_PIXEL, redshift
        )
        / compactness_factor(redshift),
        rel=1e-6,
    )


def test_compactness_factor_size_evolution():
    assert compactness_factor(0.0) == pytest.approx(Config.TNG_COMPACT_C0)
    assert compactness_factor(1.0) == pytest.approx(
        Config.TNG_COMPACT_C0 * 2.0**Config.TNG_COMPACT_BETA
    )
    assert compactness_factor(1.0, c0=1.0, beta=0.0) == 1.0


def test_compactness_squeeze_conserves_flux(tmp_path):
    view = _write_fake_tng_galaxy(tmp_path, "888", size=96)
    redshift = 0.5
    squeezed = TNGRenderer().render_for_physical_redshift(
        view,
        redshift,
        surface_brightness_cut_mag_arcsec2=0.0,
    )
    with mock.patch(
        "euclid_polish.sky.generation.tng_galaxy.compactness_factor",
        lambda redshift: 1.0,
    ):
        plain = TNGRenderer().render_for_physical_redshift(
            view,
            redshift,
            surface_brightness_cut_mag_arcsec2=0.0,
        )
    squeezed_geometry = squeezed.trace.geometry
    plain_geometry = plain.trace.geometry
    assert isinstance(squeezed_geometry, PhysicalRedshiftGeometry)
    assert isinstance(plain_geometry, PhysicalRedshiftGeometry)
    assert squeezed_geometry.rebin_factor > plain_geometry.rebin_factor
    assert squeezed.flux_e("VIS") == pytest.approx(
        plain.flux_e("VIS"), rel=0.05
    )


def test_surface_brightness_truncation_crops_faint_outskirts(tmp_path):
    view = _write_fake_tng_galaxy(
        tmp_path, "777", size=96, faint_outskirts=True
    )
    renderer = TNGRenderer()

    cropped = renderer.render_for_physical_redshift(view, 0.5)
    full = renderer.render_for_physical_redshift(
        view, 0.5, surface_brightness_cut_mag_arcsec2=0.0
    )

    assert cropped.shape[0] < full.shape[0]
    assert cropped.flux_e("VIS") > 0.0


def test_analytic_showability_predictors_track_render(tmp_path):
    view = _write_fake_tng_galaxy(tmp_path, "555", size=96)
    renderer = TNGRenderer()
    redshift = 0.7
    rendered = renderer.render_for_physical_redshift(view, redshift)

    predicted_flux = renderer.predict_vis_flux_e(view, redshift)
    predicted_radius = renderer.predict_visible_radius_arcsec(view, redshift)

    assert predicted_flux == pytest.approx(rendered.flux_e("VIS"), rel=0.1)
    assert 0.0 < predicted_radius <= rendered.shape[0] * 0.05 / 2.0 * 1.2


def test_predicted_vis_mag_faint_skip():
    assert predicted_vis_mag(10.55, 0.5) == pytest.approx(21.36, abs=0.01)
    assert predicted_vis_mag(9.0, 2.0) > Config.TNG_FAINT_SKIP_MAG_VIS
    assert predicted_vis_mag(10.5, 0.3) < 22.0


def test_poster_lens_showability_cut():
    from scripts.fasrc_poster_cutout import (
        LENS_MIN_SOURCE_VIS_E,
        LENS_MIN_THETA_E_VISIBLE_FRAC,
        _lens_is_showable,
    )

    base = {
        "theta_E_arcsec": 1.5,
        "lens_visible_r_arcsec": 2.0,
        "source_flux_vis_e": 5000.0,
    }
    assert _lens_is_showable(base)
    assert not _lens_is_showable({**base, "lens_visible_r_arcsec": 4.0})
    assert not _lens_is_showable({**base, "source_flux_vis_e": 100.0})
    assert _lens_is_showable({"theta_E_arcsec": 0.4})
    assert LENS_MIN_THETA_E_VISIBLE_FRAC > 0
    assert LENS_MIN_SOURCE_VIS_E > 0
