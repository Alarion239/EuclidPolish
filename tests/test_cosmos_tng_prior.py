import numpy as np

from euclid_polish.sky.generation.cosmos_tng_prior import CosmosTngPrior


def test_joint_prior_keeps_faint_rows_and_imputes_missing_values(tmp_path):
    path = tmp_path / "prior.npz"
    np.savez(
        path,
        catalog_id=np.arange(4),
        mag_VIS=np.array([21.0, 23.0, 25.0, 27.0]),
        mag_Y_E=np.array([20.5, 22.5, 999.0, 26.5]),
        mag_J_E=np.array([20.2, 22.2, 999.0, 26.2]),
        mag_H_E=np.array([20.0, 22.0, 999.0, 26.0]),
        z_phot=np.array([0.3, 0.8, 1.1, 1.8]),
        logmass_lephare=np.array([10.5, 10.0, 9.5, 9.0]),
        re_combined_arcsec=np.array([0.8, 0.4, np.nan, 0.12]),
    )
    prior = CosmosTngPrior(path)
    assert len(prior) == 4
    rng = np.random.default_rng(4)
    draws = [prior.sample(rng) for _ in range(200)]
    imputed = [draw for draw in draws if draw.catalog_id == "2"]
    assert imputed
    assert all(draw.imputed_photometry and draw.imputed_size for draw in imputed)
    assert all(np.all(np.isfinite(draw.magnitudes)) for draw in draws)
    assert all(np.all(np.asarray(draw.flux_e_per_band) > 0) for draw in draws)
