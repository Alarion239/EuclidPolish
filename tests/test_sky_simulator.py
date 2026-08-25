"""Tests for the single-population COSMOS-conditioned TNG generator."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.provenance import ConfigSnapshot
from euclid_polish.sky.generation.cosmos_tng_prior import CosmosTngDraw
from euclid_polish.sky.generation.sky_simulator import (
    SkySimulator,
    SkySimulatorConfig,
)
from euclid_polish.tng import (
    TNGAtlas,
    TNGGalaxy,
    TNGPropertyCatalog,
    TNGRadiusManifest,
)
from euclid_polish.tng.radius_manifest import build_manifest
from euclid_polish.tng.renderer import (
    TNG_RADIUS_RENDERER_FINGERPRINT,
    TNG_RADIUS_RENDERING,
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
            ssfr_quantile=0.5,
            activity_class="star_forming",
            logssfr=-10.0,
        )

    def proxy_logmass(self, quantile, activity_class):
        assert quantile == pytest.approx(0.5)
        assert activity_class == "star_forming"
        return 8.0


class OneOversizedDrawPrior(StaticPrior):
    def __init__(self):
        self.calls = 0

    def sample(self, rng):
        self.calls += 1
        draw = super().sample(rng)
        return replace(draw, re_arcsec=10.0) if self.calls == 1 else draw


def _write_fake_tng_galaxy(tng_dir, gid, *, size=24):
    directory = os.path.join(tng_dir, gid)
    os.makedirs(directory, exist_ok=True)
    for orientation in range(1, 6):
        for band in ("VIS", "Y", "J", "H"):
            frame = np.zeros((size, size), dtype=">f4")
            frame[size // 2 - 8:size // 2 + 8,
                  size // 2 - 8:size // 2 + 8] = 500.0
            hdu = fits.PrimaryHDU(frame)
            hdu.header["BUNIT"] = "MJy/sr"
            hdu.header["CDELT1"] = 100.0
            hdu.header["CUNIT1"] = "pc"
            hdu.header["CDELT2"] = 100.0
            hdu.header["CUNIT2"] = "pc"
            hdu.writeto(
                os.path.join(
                    directory,
                    f"TNG{gid}_O{orientation}_Euclid_{band}.fits",
                ),
                overwrite=True,
            )
    open(os.path.join(directory, Config.Tng.DONE_MARKER), "w").close()


def _simulator(tmp_path, *, galaxy_density_arcmin2: float = 1.0):
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
    return SkySimulator(
        StaticPrior(),
        SkySimulatorConfig(
            image_size=64,
            pixel_scale=Config.DEFAULT_PIXEL_SCALE,
            galaxy_density_arcmin2=galaxy_density_arcmin2,
            star_density_arcmin2=0.0,
            lens_density_arcmin2=0.0,
            tng_galaxy_dir=tng,
            tng_properties_csv=str(properties),
            tng_radius_manifest_path=str(radius_manifest),
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
    assert all(row["radius_manifest_fingerprint"] for row in meta["galaxies"])
    assert all(
        row["tng_render_trace"]["subhalo_id"] == "111"
        for row in meta["galaxies"]
    )
    assert simulator._tng_max_output_side == 129
    assert simulator.config.tng_radius_rendering == TNG_RADIUS_RENDERING
    assert (
        simulator.config.tng_radius_renderer_fingerprint
        == TNG_RADIUS_RENDERER_FINGERPRINT
    )
    config_snapshot = ConfigSnapshot.from_dataclass(simulator.config)
    assert config_snapshot.fields["tng_radius_rendering"] == TNG_RADIUS_RENDERING
    assert (
        config_snapshot.fields["tng_radius_renderer_fingerprint"]
        == TNG_RADIUS_RENDERER_FINGERPRINT
    )
    assert all(
        row["target_re_arcsec"] == pytest.approx(0.2)
        and row["arbitrary_rotation"] is True
        and np.isfinite(row["rot_angle"])
        and np.isnan(row["achieved_re_arcsec"])
        and row["radius_rendering"] == TNG_RADIUS_RENDERING
        and row["radius_renderer_fingerprint"]
        == TNG_RADIUS_RENDERER_FINGERPRINT
        for row in meta["galaxies"]
    )
    assert all(
        row["native_tng_logmass"] == pytest.approx(10.0)
        and row["native_tng_sfr"] == pytest.approx(1.0)
        and row["native_tng_logssfr"] == pytest.approx(-10.0)
        and row["native_tng_zero_sfr"] is False
        and row["morphology_proxy_logmass"] == pytest.approx(8.0)
        and row["morphology_activity_class"] == "star_forming"
        and row["morphology_effective_donors"] == pytest.approx(1.0)
        and row["target_ssfr_quantile"] == pytest.approx(0.5)
        and row["tng_ssfr_quantile"] == pytest.approx(0.5)
        for row in meta["galaxies"]
    )
    assert image.data.sum() > 0


def test_explicit_galaxies_open_atlas_when_configured_density_is_zero(tmp_path):
    simulator = _simulator(tmp_path, galaxy_density_arcmin2=0.0)

    _image, meta = simulator.simulate_field(
        np.random.default_rng(17),
        n_galaxies=1,
        n_stars=0,
        n_lenses=0,
    )

    assert simulator.tng_atlas is not None
    assert len(meta["galaxies"]) == 1


def test_oversized_radius_draw_is_rejected_and_redrawn(tmp_path):
    simulator = _simulator(tmp_path)
    prior = OneOversizedDrawPrior()
    simulator.population_prior = prior

    record = simulator._add_tng_galaxy(
        np.zeros((64, 64, 4), dtype=np.float32),
        np.random.default_rng(31),
    )

    assert prior.calls == 2
    assert record is not None
    assert record["target_re_arcsec"] == pytest.approx(0.2)
    assert record["radius_scale_factor"] <= 1.0


def test_random_morphology_selection_excludes_donors_that_need_enlargement(
    tmp_path,
):
    simulator = _simulator(tmp_path)
    small = TNGGalaxy(Path(tmp_path, "small"), "1")
    large = TNGGalaxy(Path(tmp_path, "large"), "2")
    simulator.tng_atlas = TNGAtlas(
        root=tmp_path,
        galaxies=(small, large),
        properties=TNGPropertyCatalog({}, (None, 0, 0)),
        radii=TNGRadiusManifest(
            {
                (galaxy.subhalo_id, orientation): radius
                for galaxy, radius in ((small, 2.0), (large, 20.0))
                for orientation in range(1, 6)
            },
            "r" * 64,
        ),
    )
    simulator._atlas_logm = np.asarray([9.0, 10.0])
    simulator._atlas_sfr = np.asarray([1.0, 1.0])
    simulator._atlas_logssfr = np.asarray([-9.0, -10.0])
    simulator._atlas_zero_sfr = np.asarray([False, False])
    simulator._atlas_activity_class = np.asarray([
        "star_forming", "star_forming",
    ])
    simulator._atlas_mass_quantile = np.asarray([0.0, 1.0])
    simulator._atlas_ssfr_quantile = np.asarray([0.0, 1.0])
    simulator._morphology_use_counts = np.zeros(2, dtype=np.int64)

    galaxy, metadata = simulator._pick_random_field_galaxy(
        np.random.default_rng(9), target_re_arcsec=0.5,
    )

    assert galaxy == large
    assert metadata["selection_probability"] == pytest.approx(1.0)
    assert metadata["effective_donors"] == pytest.approx(1.0)


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
