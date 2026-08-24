import json

import numpy as np
import pytest

from euclid_polish.sky.generation.cosmos_tng_prior import (
    CosmosTngPrior,
    F814WToVisTransfer,
    conditional_mass_quantiles,
    conditional_ssfr_quantiles,
    cross_validated_mass_bandwidth,
    joint_quantile_transport_weights,
)


def _write_prior(path):
    np.savez(
        path,
        catalog_id=np.arange(4),
        mag_hst_f814w=np.array([21.0, 23.0, 25.0, 27.0]),
        z_phot=np.array([0.3, 0.8, 1.1, 1.8]),
        logmass_lephare=np.array([10.5, 10.0, 9.5, 9.0]),
        logssfr_lephare=np.array([-12.0, -10.0, -9.5, -9.0]),
        re_combined_arcsec=np.array([0.8, 0.4, np.nan, 0.12]),
        generator_ready=np.array([True, True, False, True]),
    )


def test_prior_requires_strict_generator_ready_rows_and_fit(tmp_path):
    path = tmp_path / "prior.npz"
    _write_prior(path)
    with pytest.raises(ValueError, match="fitted F814W"):
        CosmosTngPrior(path, photometric_fit_path=tmp_path / "missing.json")

    transfer = F814WToVisTransfer(source="embedded:test")
    prior = CosmosTngPrior(path, photometric_transfer=transfer)
    assert len(prior) == 3
    draws = [prior.sample(np.random.default_rng(i)) for i in range(20)]
    assert all(not draw.imputed_size for draw in draws)
    assert {draw.catalog_id for draw in draws}.issubset({"0", "1", "3"})
    assert all(0.0 < draw.mass_quantile < 1.0 for draw in draws)
    assert all(0.0 < draw.ssfr_quantile < 1.0 for draw in draws)
    assert all(np.isfinite(draw.logssfr) for draw in draws)
    assert {draw.activity_class for draw in draws} <= {
        "quenched", "star_forming",
    }


def test_prior_does_not_clip_f814w_before_vis_transfer(tmp_path):
    path = tmp_path / "prior.npz"
    np.savez(
        path,
        catalog_id=np.arange(2),
        mag_hst_f814w=np.array([17.5, 29.5]),
        z_phot=np.array([0.3, 1.1]),
        logmass_lephare=np.array([10.5, 9.5]),
        logssfr_lephare=np.array([-12.0, -9.5]),
        re_combined_arcsec=np.array([0.8, 0.2]),
        generator_ready=np.array([True, True]),
    )

    prior = CosmosTngPrior(
        path,
        photometric_transfer=F814WToVisTransfer(source="embedded:test"),
    )

    assert len(prior) == 2
    assert set(prior.f814w.tolist()) == {17.5, 29.5}


def test_multicone_fit_maps_f814w_to_vis_brightness(tmp_path):
    path = tmp_path / "prior.npz"
    _write_prior(path)
    fit = tmp_path / "fit.json"
    fit.write_text(json.dumps({
        "inputs": {"euclid_cone_count": 6},
        "fit": {
            "vis_minus_f814w_mag": 0.2,
            "magnitude_slope": 1.0,
            "scatter_mag": 0.0,
        },
        "local_normalization_sensitivity_fit": {
            "vis_minus_f814w_mag": 9.0,
            "magnitude_slope": 0.1,
            "scatter_mag": 0.0,
        },
    }))
    draw = CosmosTngPrior(path, photometric_fit_path=fit).sample(
        np.random.default_rng(2)
    )
    assert draw.target_vis_mag == pytest.approx(draw.mag_hst_f814w + 0.2)
    assert draw.brightness_transfer.startswith(
        "fixed_normalization_fit:"
    )


def test_embedded_transfer_does_not_need_fit_file(tmp_path):
    path = tmp_path / "prior.npz"
    _write_prior(path)
    transfer = F814WToVisTransfer(
        offset_mag=0.3,
        magnitude_slope=0.8,
        scatter_mag=0.0,
        source="embedded:test-fit",
    )

    draw = CosmosTngPrior(
        path,
        photometric_fit_path=tmp_path / "missing.json",
        photometric_transfer=transfer,
    ).sample(np.random.default_rng(2))

    expected = 24.0 + 0.8 * (draw.mag_hst_f814w - 24.0) + 0.3
    assert draw.target_vis_mag == pytest.approx(expected)
    assert draw.brightness_transfer == "embedded:test-fit"


def test_mass_bandwidth_cross_validation_includes_kernel_normalization():
    rng = np.random.default_rng(91)
    logmass = rng.normal(10.2, 0.18, size=300)

    bandwidth = cross_validated_mass_bandwidth(logmass)

    assert 0.03 <= bandwidth < 0.3


def test_conditional_mass_quantiles_preserve_classes():
    masses = np.array([8.0, 9.0, 10.0, 10.5, 11.0, 11.5])
    classes = np.array([
        "star_forming", "star_forming", "star_forming",
        "quenched", "quenched", "quenched",
    ])
    quantiles = conditional_mass_quantiles(masses, classes)
    assert quantiles[:3] == pytest.approx([1 / 6, 3 / 6, 5 / 6])
    assert quantiles[3:] == pytest.approx([1 / 6, 3 / 6, 5 / 6])


def test_ssfr_quantiles_keep_zero_sfr_as_censored_point_mass():
    logssfr = np.array([np.nan, np.nan, -12.0, -11.5, -10.0, -9.0])
    classes = np.array([
        "quenched", "quenched", "quenched", "quenched",
        "star_forming", "star_forming",
    ])
    zero_sfr = np.array([True, True, False, False, False, False])

    quantiles = conditional_ssfr_quantiles(
        logssfr, classes, zero_sfr=zero_sfr,
    )

    assert quantiles[:4] == pytest.approx([0.25, 0.25, 0.625, 0.875])
    assert quantiles[4:] == pytest.approx([0.25, 0.75])


def test_joint_mass_ssfr_transport_uses_both_ranks_and_diversity_floor():
    mass = np.tile((np.arange(10) + 0.5) / 10, 10)
    ssfr = np.repeat((np.arange(10) + 0.5) / 10, 10)

    weights, mass_bandwidth, ssfr_bandwidth, effective = (
        joint_quantile_transport_weights(
            mass,
            ssfr,
            0.05,
            0.95,
            mass_bandwidth=0.03,
            ssfr_bandwidth=0.03,
            minimum_effective_donors=64,
        )
    )

    assert weights.sum() == pytest.approx(1.0)
    assert mass_bandwidth > 0.03
    assert ssfr_bandwidth > 0.03
    assert effective >= 64.0 - 1e-6
    assert int(np.argmax(weights)) == 90
