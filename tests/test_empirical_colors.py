"""Empirical colour pipeline: SFR rank transport, per-band scaling, lenses."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.image import Image, Role
from euclid_polish.sky.generation.cosmos_tng_prior import (
    JointGalaxyPopulationPrior,
    conditional_ssfr_quantiles,
    sfr_rank_transport_weights,
)
from euclid_polish.sky.generation.sky_simulator import (
    SkySimulator,
    SkySimulatorConfig,
)
from euclid_polish.sky.generation.source_catalog import _galaxy_row
from euclid_polish.tng.radius_manifest import build_manifest
from euclid_polish.tng.renderer import TNGRenderer
from euclid_polish.tng.types import (
    EmpiricalColorTransform,
    NominalRadiusGeometry,
    RenderedTNG,
    TNGRedshiftTransform,
    TNGRenderTrace,
    TNGRotation,
    TNGView,
)
from tests.test_euclid_galaxy_prior import active_payload
from tests.test_sky_simulator import _write_fake_tng_galaxy


def _electron_ratio(band_name: str, flux_ratio: float) -> float:
    band = Config.get_band(band_name)
    vis = Config.get_band("VIS")
    return flux_ratio * 10.0 ** (
        0.4 * (band.sim_zeropoint_e - vis.sim_zeropoint_e)
    )


# --------------------------------------------------------------------------- #
# SFR rank transport
# --------------------------------------------------------------------------- #

def test_sfr_rank_transport_widens_to_the_effective_donor_floor():
    ranks = np.linspace(0.0, 1.0, 200)

    probabilities, used_bandwidth, effective = sfr_rank_transport_weights(
        ranks, 0.5, bandwidth=1e-4, minimum_effective_donors=64,
    )

    assert probabilities.shape == ranks.shape
    assert np.isclose(np.sum(probabilities), 1.0)
    assert effective >= 64.0 - 1e-6
    assert used_bandwidth > 1e-4


def test_sfr_rank_transport_balance_multiplies_a_flat_kernel():
    ranks = np.full(4, 0.5)
    balance = np.asarray([1.0, 2.0, 3.0, 4.0])

    probabilities, _, _ = sfr_rank_transport_weights(
        ranks, 0.5, bandwidth=1.0, minimum_effective_donors=1,
        balance_weights=balance,
    )

    assert probabilities == pytest.approx(balance / balance.sum())


def test_sfr_rank_transport_reaches_the_censored_zero_sfr_mass():
    log_sfr = np.asarray([np.nan, np.nan, -0.5, 0.0, 0.5, 1.0])
    zero = np.asarray([True, True, False, False, False, False])
    ranks = conditional_ssfr_quantiles(
        log_sfr, np.full(log_sfr.shape, "all"), zero_sfr=zero,
    )
    censored_rank = ranks[0]
    assert ranks[1] == censored_rank  # shared bottom point mass

    probabilities, _, _ = sfr_rank_transport_weights(
        ranks, censored_rank, bandwidth=0.05, minimum_effective_donors=1,
    )

    assert probabilities[0] == probabilities[1] == np.max(probabilities)


def test_sfr_rank_transport_rejects_invalid_targets():
    with pytest.raises(ValueError, match="quantile-transport"):
        sfr_rank_transport_weights(
            np.asarray([0.2, 0.8]), float("nan"), bandwidth=0.1,
        )


# --------------------------------------------------------------------------- #
# Renderer per-band colour scaling
# --------------------------------------------------------------------------- #

def _rendered_fixture(values: tuple[float, float, float, float]) -> RenderedTNG:
    data = np.zeros((9, 9, 4), dtype=np.float32)
    for index, value in enumerate(values):
        data[3:6, 3:6, index] = value
    image = Image(
        data=data,
        pixel_scale_arcsec=Config.DEFAULT_PIXEL_SCALE,
        band_names=tuple(Config.LR_INPUT_BAND_NAMES),
        is_clean=True,
        role=Role.CLEAN,
    )
    trace = TNGRenderTrace(
        view=TNGView(Path("unused"), "fixture", 1, 20.0),
        rotation=TNGRotation(),
        geometry=NominalRadiusGeometry(
            target_re_arcsec=20.0 * Config.DEFAULT_PIXEL_SCALE * 0.5,
            scale_factor=0.5,
            radius_rendering="test_shrink_only",
            radius_renderer_fingerprint="test-fixture",
        ),
    )
    return RenderedTNG(image=image, trace=trace)


def test_apply_empirical_colors_hits_target_ratios_and_keeps_vis_bitwise():
    renderer = TNGRenderer()
    rendered = _rendered_fixture((10.0, 3.0, 5.0, 7.0))
    vis_before = rendered.plane("VIS").copy()
    target_ratios = (1.4, 1.1, 0.8)

    colored = renderer.apply_empirical_colors(
        rendered,
        target_ratios=target_ratios,
        color_pooling="forest",
        neighborhood_rows=123,
        calibration_fingerprint="c" * 64,
    )

    np.testing.assert_array_equal(colored.plane("VIS"), vis_before)
    vis_flux = colored.flux_e("VIS")
    for band_name, ratio in zip(("Y_E", "J_E", "H_E"), target_ratios,
                                strict=True):
        assert colored.flux_e(band_name) / vis_flux == pytest.approx(
            _electron_ratio(band_name, ratio), rel=1e-6,
        )
    assert colored.trace.color is not None
    fields = colored.record_fields()
    assert fields["target_ratio_y"] == pytest.approx(1.4)
    assert fields["vis_minus_y_mag"] == pytest.approx(2.5 * math.log10(1.4))
    assert fields["color_pooling"] == "forest"
    assert fields["color_neighborhood_rows"] == 123
    assert fields["color_calibration_fingerprint"] == "c" * 64
    assert fields["color_band_factors"][0] == 1.0


def test_apply_empirical_colors_refuses_an_empty_band():
    renderer = TNGRenderer()
    rendered = _rendered_fixture((10.0, 0.0, 5.0, 7.0))

    with pytest.raises(ValueError, match="Y_E"):
        renderer.apply_empirical_colors(
            rendered, target_ratios=(1.0, 1.0, 1.0),
        )


def test_empirical_color_transform_requires_a_unit_vis_factor():
    with pytest.raises(ValueError, match="VIS anchor"):
        EmpiricalColorTransform(
            target_ratios=(1.0, 1.0, 1.0),
            band_factors=(1.1, 1.0, 1.0, 1.0),
        )


def test_achromatic_redshift_transform_requires_equal_factors():
    with pytest.raises(ValueError, match="achromatic"):
        TNGRedshiftTransform(
            redshift=0.5,
            band_factors=(0.3, 0.3, 0.3, 0.4),
            drift_mode="achromatic",
            drift_epsilon=0.0,
            dimming_factor=0.3,
        )


# --------------------------------------------------------------------------- #
# Achromatic physical rendering (lens path)
# --------------------------------------------------------------------------- #

def _write_physical_view(root: Path) -> TNGView:
    directory = root / "222"
    directory.mkdir()
    side = 96
    centre = side // 2
    surface_brightness = np.zeros((side, side), dtype=np.float32)
    surface_brightness[centre - 12:centre + 12, centre - 6:centre + 6] = 400.0
    for band_index, band in enumerate(("VIS", "Y", "J", "H"), start=1):
        hdu = fits.PrimaryHDU(
            np.asarray(surface_brightness * band_index, dtype=">f4")
        )
        hdu.header["BUNIT"] = "MJy/sr"
        hdu.header["CDELT1"] = 100.0
        hdu.header["CUNIT1"] = "pc"
        hdu.header["CDELT2"] = 100.0
        hdu.header["CUNIT2"] = "pc"
        hdu.writeto(directory / f"TNG222_O{1}_Euclid_{band}.fits")
    return TNGView(
        galaxy_dir=directory,
        subhalo_id="222",
        orientation=1,
        native_re_px=20.0,
        radius_manifest_fingerprint="fixture-manifest",
    )


def test_achromatic_physical_render_applies_pure_tolman_dimming(tmp_path):
    renderer = TNGRenderer()
    view = _write_physical_view(tmp_path)
    redshift = 0.5
    dimming = (1.0 + redshift) ** -3

    rendered = renderer.render_physical_at_redshift(
        view, redshift, rng=None, chromatic=False,
        surface_brightness_cut_mag_arcsec2=99.0,
    )

    transform = rendered.trace.redshift
    assert transform is not None
    assert transform.drift_mode == "achromatic"
    assert transform.drift_epsilon == 0.0
    assert transform.band_factors == pytest.approx((dimming,) * 4)
    chromatic = renderer.render_physical_at_redshift(
        view, redshift, rng=None, chromatic=True,
        surface_brightness_cut_mag_arcsec2=99.0,
    )
    assert chromatic.trace.redshift is not None
    assert chromatic.trace.redshift.drift_mode == "sed_interp"
    assert chromatic.trace.redshift.band_factors != transform.band_factors


def test_achromatic_flux_prediction_uses_tolman_only(tmp_path):
    renderer = TNGRenderer()
    view = _write_physical_view(tmp_path)
    redshift = 0.8

    achromatic = renderer.predict_vis_flux_e(
        view, redshift, chromatic=False,
    )
    chromatic = renderer.predict_vis_flux_e(view, redshift, chromatic=True)

    assert achromatic > 0.0
    assert achromatic != pytest.approx(chromatic)
    radius = renderer.predict_visible_radius_arcsec(
        view, redshift, chromatic=False,
    )
    assert radius >= 0.0


# --------------------------------------------------------------------------- #
# Staged joint-prior generation end to end
# --------------------------------------------------------------------------- #

def _joint_simulator(tmp_path) -> SkySimulator:
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    properties = tmp_path / "tng_properties.csv"
    properties.write_text(
        "id,sfr,mass_stars,m_halo,reff\n111,1,1e10,1e12,2\n"
    )
    radius_manifest = tmp_path / "tng_radius_manifest.json"
    build_manifest(
        tng, properties_path=str(properties),
        output_path=str(radius_manifest),
    )
    prior = JointGalaxyPopulationPrior(active_payload())
    return SkySimulator(
        prior,
        SkySimulatorConfig(
            image_size=64,
            pixel_scale=Config.DEFAULT_PIXEL_SCALE,
            galaxy_density_arcmin2=1.0,
            star_density_arcmin2=0.0,
            lens_density_arcmin2=0.0,
            tng_galaxy_dir=tng,
            tng_properties_csv=str(properties),
            tng_radius_manifest_path=str(radius_manifest),
        ),
    )


def test_joint_prior_fields_carry_empirical_colors_and_no_redshift(tmp_path):
    simulator = _joint_simulator(tmp_path)

    _image, meta = simulator.simulate_field(
        np.random.default_rng(5),
        n_galaxies=3,
        n_stars=0,
        n_lenses=0,
    )

    assert len(meta["galaxies"]) == 3
    for record in meta["galaxies"]:
        assert math.isnan(record["z"])
        assert record["sfr_class"] in ("quenched", "star_forming")
        assert 0.0 < record["sfr_rank"] < 1.0
        assert np.isfinite(record["sfr_log10"])
        assert 0.0 <= record["morphology_target_sfr_rank"] <= 1.0
        assert np.isfinite(record["morphology_sfr_rank_delta"])
        assert record["color_pooling"] == "forest"
        assert record["color_neighborhood_rows"] > 0
        assert record["color_calibration_fingerprint"] == "c" * 64
        # The composited stamp realizes the drawn flux ratios exactly.
        vis_flux, y_flux, j_flux, h_flux = record["flux_e_per_band"]
        for band_name, flux, ratio_key in (
            ("Y_E", y_flux, "target_ratio_y"),
            ("J_E", j_flux, "target_ratio_j"),
            ("H_E", h_flux, "target_ratio_h"),
        ):
            assert flux / vis_flux == pytest.approx(
                _electron_ratio(band_name, record[ratio_key]), rel=1e-5,
            )
        csv_row = _galaxy_row(0, record)
        assert csv_row["z"] == ""
        assert csv_row["sfr_class"] == record["sfr_class"]
        assert np.isfinite(float(csv_row["vis_minus_y_mag"]))


def test_lens_component_coloring_is_class_conditioned(tmp_path):
    simulator = _joint_simulator(tmp_path)
    stamp = _rendered_fixture((10.0, 3.0, 5.0, 7.0))
    rng = np.random.default_rng(41)

    quenched_stamp, quenched_draw = simulator._colored_lens_component(
        stamp, "quenched", rng,
    )
    forming_stamp, forming_draw = simulator._colored_lens_component(
        stamp, "star_forming", rng,
    )

    assert quenched_draw.sfr_class == "quenched"
    assert forming_draw.sfr_class == "star_forming"
    for colored, draw in (
        (quenched_stamp, quenched_draw),
        (forming_stamp, forming_draw),
    ):
        assert colored.trace.color is not None
        np.testing.assert_array_equal(
            colored.plane("VIS"), stamp.plane("VIS"),
        )
        assert colored.flux_e("Y_E") / colored.flux_e("VIS") == pytest.approx(
            _electron_ratio("Y_E", draw.ratio_y), rel=1e-6,
        )
