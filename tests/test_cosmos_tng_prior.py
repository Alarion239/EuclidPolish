import json

import numpy as np
import pytest

from euclid_polish.sky.generation.cosmos_tng_prior import (
    CosmosTngPrior,
    F814WToVisTransfer,
)


def _write_prior(path):
    np.savez(
        path,
        catalog_id=np.arange(4),
        mag_hst_f814w=np.array([21.0, 23.0, 25.0, 27.0]),
        z_phot=np.array([0.3, 0.8, 1.1, 1.8]),
        logmass_lephare=np.array([10.5, 10.0, 9.5, 9.0]),
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
