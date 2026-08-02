"""Tests for the single-population COSMOS-conditioned TNG generator."""
from __future__ import annotations

import os

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.sky.generation.cosmos_tng_prior import CosmosTngDraw
from euclid_polish.sky.generation.sky_simulator import (
    SkySimulator,
    SkySimulatorConfig,
    _sample_star_band_magnitudes,
)


class StaticPrior:
    def sample(self, _rng):
        return CosmosTngDraw(
            catalog_id="cosmos-1",
            mag_hst_f814w=23.0,
            target_vis_mag=23.1,
            target_vis_flux_e=1200.0,
            z=0.8,
            logmass=10.0,
            re_arcsec=0.2,
            imputed_size=False,
            brightness_transfer="test",
            mass_quantile=0.5,
            activity_class="star_forming",
        )

    def proxy_logmass(self, quantile, activity_class):
        assert quantile == pytest.approx(0.5)
        assert activity_class == "star_forming"
        return 8.0


def _write_fake_tng_galaxy(tng_dir, gid, *, size=24):
    directory = os.path.join(tng_dir, gid)
    os.makedirs(directory, exist_ok=True)
    for orientation in range(1, 6):
        for band in ("VIS", "Y", "J", "H"):
            frame = np.zeros((size, size), dtype=">f4")
            frame[size // 2 - 2:size // 2 + 2,
                  size // 2 - 2:size // 2 + 2] = 500.0
            fits.PrimaryHDU(frame).writeto(
                os.path.join(
                    directory,
                    f"TNG{gid}_O{orientation}_Euclid_{band}.fits",
                ),
                overwrite=True,
            )
    open(os.path.join(directory, Config.Tng.DONE_MARKER), "w").close()


def _simulator(tmp_path):
    tng = str(tmp_path / "tng")
    _write_fake_tng_galaxy(tng, "111")
    properties = tmp_path / "tng_properties.csv"
    properties.write_text(
        "id,sfr,mass_stars,m_halo,reff\n111,1,1e10,1e12,2\n"
    )
    return SkySimulator(
        StaticPrior(),
        SkySimulatorConfig(
            image_size=64,
            pixel_scale=Config.DEFAULT_PIXEL_SCALE,
            galaxy_density_arcmin2=1.0,
            star_density_arcmin2=0.0,
            lens_density_arcmin2=0.0,
            tng_galaxy_dir=tng,
            tng_properties_csv=str(properties),
        ),
    )


def test_joint_population_renders_only_tng(tmp_path):
    simulator = _simulator(tmp_path)
    image, meta = simulator.simulate_field(
        np.random.default_rng(3),
        n_galaxies=4,
        n_stars=0,
        n_lenses=0,
    )
    assert isinstance(image, Image)
    assert image.shape == (64, 64, 4)
    assert len(meta["galaxies"]) == 4
    assert {row["render"] for row in meta["galaxies"]} == {"tng"}
    assert {row["population_prior"] for row in meta["galaxies"]} == {
        "cosmos2025_joint"
    }
    assert all(row["catalog_id"] == "cosmos-1" for row in meta["galaxies"])
    assert all(
        row["native_tng_logmass"] == pytest.approx(10.0)
        and row["morphology_proxy_logmass"] == pytest.approx(8.0)
        and row["morphology_activity_class"] == "star_forming"
        and row["morphology_effective_donors"] == pytest.approx(1.0)
        for row in meta["galaxies"]
    )
    assert image.data.sum() > 0


def test_explicit_zero_sources_is_empty(tmp_path):
    image, meta = _simulator(tmp_path).simulate_field(
        np.random.default_rng(0),
        n_galaxies=0,
        n_stars=0,
        n_lenses=0,
    )
    assert np.all(image.data == 0)
    assert meta["n_galaxies"] == 0


def test_prior_required_for_nonzero_density(tmp_path):
    with pytest.raises(ValueError, match="population_prior"):
        SkySimulator(
            None,
            SkySimulatorConfig(
                galaxy_density_arcmin2=1.0,
                lens_density_arcmin2=0.0,
                tng_galaxy_dir=str(tmp_path),
            ),
        )


def test_star_colours_remain_correlated():
    rng = np.random.default_rng(123)
    with pytest.raises(ValueError, match="empirical stellar prior"):
        _sample_star_band_magnitudes(rng, 20.0)
