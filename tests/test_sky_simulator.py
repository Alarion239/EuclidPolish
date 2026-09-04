"""Tests for the single-population COSMOS-conditioned TNG generator."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.image import Image
from euclid_polish.provenance import ConfigSnapshot
from euclid_polish.sky.generation.compositing import composite_stamp
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


def _simulator(
    tmp_path,
    *,
    galaxy_density_arcmin2: float = 1.0,
    galaxy_off_field_padding_hr_pix: int = 68,
):
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
            galaxy_off_field_padding_hr_pix=galaxy_off_field_padding_hr_pix,
            star_density_arcmin2=0.0,
            lens_density_arcmin2=0.0,
            tng_galaxy_dir=tng,
            tng_properties_csv=str(properties),
            tng_radius_manifest_path=str(radius_manifest),
        ),
    )


def test_off_field_default_matches_32_block_half_receptive_field():
    config = SkySimulatorConfig()

    assert config.galaxy_off_field_padding_hr_pix == 68
    assert 68 * config.pixel_scale == pytest.approx(3.4)


def test_composite_stamp_adds_only_the_exact_intersection():
    canvas = np.zeros((4, 4, 1), dtype=np.float32)
    stamp = np.arange(25, dtype=np.float32).reshape(5, 5, 1)

    assert composite_stamp(canvas, stamp, x0=-1.0, y0=2.0) is True
    expected = np.zeros_like(canvas)
    expected[:4, :2, :] = stamp[:4, 3:5, :]
    np.testing.assert_array_equal(canvas, expected)
    assert composite_stamp(canvas, stamp, x0=-20.0, y0=2.0) is False


def test_off_field_positions_tile_only_the_exterior_frame(tmp_path):
    simulator = _simulator(tmp_path)
    side = simulator.config.image_size
    padding = simulator.config.galaxy_off_field_padding_hr_pix

    positions = [
        simulator._random_off_field_pix(np.random.default_rng(seed))
        for seed in range(200)
    ]

    assert all(
        -padding <= x < side + padding
        and -padding <= y < side + padding
        and not (0.0 <= x < side and 0.0 <= y < side)
        for x, y in positions
    )
    assert {"left", "right", "top", "bottom"} == {
        "left" if x < 0.0 else "right" if x >= side
        else "top" if y < 0.0 else "bottom"
        for x, y in positions
    }


def test_off_field_reach_rejects_before_donor_selection(tmp_path, monkeypatch):
    simulator = _simulator(tmp_path)
    canvas = np.zeros((64, 64, 4), dtype=np.float32)

    monkeypatch.setattr(
        simulator,
        "_pick_field_galaxy",
        lambda *_args, **_kwargs: pytest.fail("donor selection must not run"),
    )
    record = simulator._add_tng_galaxy(
        canvas,
        np.random.default_rng(4),
        position=(-17.0, 32.0),
        off_field=True,
    )

    assert record is None  # StaticPrior has R_e=0.2": 4 R_e = 16 HR pixels.
    assert not np.any(canvas)


def test_off_field_rng_preserves_existing_streams_and_zero_padding(tmp_path):
    base = _simulator(
        tmp_path / "base",
        galaxy_density_arcmin2=2000.0,
        galaxy_off_field_padding_hr_pix=0,
    )
    extended = _simulator(
        tmp_path / "extended",
        galaxy_density_arcmin2=2000.0,
        galaxy_off_field_padding_hr_pix=68,
    )
    for simulator in (base, extended):
        simulator.config.star_density_arcmin2 = 2000.0
        simulator.config.lens_density_arcmin2 = 2000.0
        simulator._draw_star = lambda rng: {"token": float(rng.random())}
        simulator._add_lens = lambda _canvas, rng: {
            "token": float(rng.random())
        }

    base_rng = np.random.default_rng(29)
    extended_rng = np.random.default_rng(29)
    _base_image, base_meta = base.simulate_field(base_rng)
    _extended_image, extended_meta = extended.simulate_field(extended_rng)

    def interior_signature(meta):
        return [
            (
                row["x_pix"], row["y_pix"], row["subhalo_id"],
                row["orientation"], row["rot_angle"],
            )
            for row in meta["galaxies"]
        ]

    assert interior_signature(base_meta) == interior_signature(extended_meta)
    assert base_meta["stars"] == extended_meta["stars"]
    assert base_meta["lenses"] == extended_meta["lenses"]
    assert base_rng.random() == extended_rng.random()
    assert base_meta["off_field_galaxies"] == []
    assert base_meta["n_off_field_galaxy_proposals"] == 0
    assert extended_meta["n_off_field_galaxies"] == len(
        extended_meta["off_field_galaxies"]
    )
    assert extended_meta["n_off_field_galaxies"] > 0
    assert all(row["off_field"] for row in extended_meta["off_field_galaxies"])


def test_off_field_poisson_thinning_is_nested_and_metadata_is_separate(
    tmp_path,
):
    lower = _simulator(tmp_path / "lower", galaxy_density_arcmin2=500.0)
    higher = _simulator(tmp_path / "higher", galaxy_density_arcmin2=800.0)
    for simulator in (lower, higher):
        simulator.config.galaxy_thinning_max_density_arcmin2 = 1000.0

        def fake_add(
            self, _canvas, rng, *, position=None, off_field=False, **_kwargs,
        ):
            x_pix, y_pix = (
                position if position is not None else self._random_pix(rng)
            )
            return {
                "x_pix": float(x_pix),
                "y_pix": float(y_pix),
                "off_field": bool(off_field),
                "proposal": int(rng.integers(0, 2**63)),
            }

        simulator._add_tng_galaxy = MethodType(fake_add, simulator)

    _, lower_meta = lower.simulate_field(
        np.random.default_rng(7), n_stars=0, n_lenses=0,
    )
    _, higher_meta = higher.simulate_field(
        np.random.default_rng(7), n_stars=0, n_lenses=0,
    )

    lower_ids = {row["proposal"] for row in lower_meta["off_field_galaxies"]}
    higher_ids = {row["proposal"] for row in higher_meta["off_field_galaxies"]}
    assert lower_ids
    assert lower_ids < higher_ids
    assert lower_meta["n_galaxies"] == len(lower_meta["galaxies"])
    assert lower_meta["n_off_field_galaxies"] == len(
        lower_meta["off_field_galaxies"]
    )
    assert all(
        not (0.0 <= row["x_pix"] < 64 and 0.0 <= row["y_pix"] < 64)
        for row in lower_meta["off_field_galaxies"]
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
