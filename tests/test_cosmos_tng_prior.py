import json

import numpy as np
import pytest

from euclid_polish.sky.generation.cosmos_tng_prior import CosmosTngPrior


def _write_prior(path):
    np.savez(
        path,
        catalog_id=np.arange(4),
        mag_hst_f814w=np.array([21.0, 23.0, 25.0, 27.0]),
        z_phot=np.array([0.3, 0.8, 1.1, 1.8]),
        logmass_lephare=np.array([10.5, 10.0, 9.5, 9.0]),
        re_combined_arcsec=np.array([0.8, 0.4, np.nan, 0.12]),
    )


def test_physical_prior_keeps_faint_rows_and_imputes_only_size(tmp_path):
    path = tmp_path / "prior.npz"
    _write_prior(path)
    prior = CosmosTngPrior(
        path, photometric_fit_path=tmp_path / "missing.json"
    )
    assert len(prior) == 4
    rng = np.random.default_rng(4)
    draws = [prior.sample(rng) for _ in range(200)]
    imputed = [draw for draw in draws if draw.catalog_id == "2"]
    assert imputed
    assert all(draw.imputed_size for draw in imputed)
    assert all(draw.target_vis_mag == draw.mag_hst_f814w for draw in draws)
    assert all(draw.target_vis_flux_e > 0 for draw in draws)


def test_multicone_fit_maps_f814w_to_vis_brightness(tmp_path):
    path = tmp_path / "prior.npz"
    _write_prior(path)
    fit = tmp_path / "fit.json"
    fit.write_text(json.dumps({
        "inputs": {"euclid_cone_count": 6},
        "local_normalization_sensitivity_fit": {
            "vis_minus_f814w_mag": 0.2,
            "magnitude_slope": 1.0,
            "scatter_mag": 0.0,
        },
    }))
    draw = CosmosTngPrior(path, photometric_fit_path=fit).sample(
        np.random.default_rng(2)
    )
    assert draw.target_vis_mag == pytest.approx(draw.mag_hst_f814w + 0.2)
    assert draw.brightness_transfer.startswith(
        "local_normalization_sensitivity_fit:"
    )
