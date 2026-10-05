"""SkySimulator donor eligibility and donor-balance state, end to end.

Uses a six-galaxy on-disk fake atlas whose donors span the shrink-only
radius range, so eligibility genuinely filters donors and the balance weights
change which donor a draw picks. No FASRC atlas is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.sky.generation.cosmos_tng_prior import (
    CosmosTngDraw,
    JointGalaxyPopulationPrior,
)
from euclid_polish.sky.generation.sky_simulator import (
    SkySimulator,
    SkySimulatorConfig,
)
from euclid_polish.sky.generation.source_catalog import SourceCatalogWriter
from euclid_polish.tng import TNGAtlas
from euclid_polish.tng.radius_manifest import build_manifest
from tests.test_euclid_galaxy_prior import active_payload
from tests.test_tng_atlas_eligibility import (
    _old_eligible_galaxies,
    _old_eligible_morphology_indices,
    _old_require_galaxy,
)

#: Square half-widths (native pixels) of each fake donor's light; the
#: measured native R_e spans about 2-12 px (0.1-0.6 arcsec at 0.05"/px).
_DONOR_HALF_WIDTHS = (4, 6, 8, 10, 12, 16)


def write_fake_atlas(root: Path) -> dict[str, str]:
    """Write a measured six-donor atlas; return the simulator config paths."""
    tng_dir = root / "tng"
    rows = ["id,sfr,mass_stars,m_halo,reff"]
    for index, half in enumerate(_DONOR_HALF_WIDTHS):
        subhalo_id = str(100 + index)
        directory = tng_dir / subhalo_id
        directory.mkdir(parents=True, exist_ok=True)
        for orientation in range(1, 6):
            width = max(2, half - (orientation - 1))
            frame = np.zeros((48, 48), dtype=">f4")
            frame[24 - width:24 + width, 24 - width:24 + width] = 500.0
            for band in ("VIS", "Y", "J", "H"):
                hdu = fits.PrimaryHDU(frame * (1.0 + 0.1 * len(band)))
                hdu.header["BUNIT"] = "MJy/sr"
                hdu.header["CDELT1"] = 100.0
                hdu.header["CUNIT1"] = "pc"
                hdu.header["CDELT2"] = 100.0
                hdu.header["CUNIT2"] = "pc"
                hdu.writeto(
                    directory / f"TNG{subhalo_id}_O{orientation}_Euclid_{band}.fits",
                    overwrite=True,
                )
        (directory / Config.Tng.DONE_MARKER).touch()
        rows.append(f"{subhalo_id},{0.1 * (index + 1)},{1e9 * (index + 1)},1e12,2")
    properties = root / "tng_properties.csv"
    properties.write_text("\n".join(rows) + "\n")
    manifest = root / "tng_radius_manifest.json"
    build_manifest(
        str(tng_dir), properties_path=str(properties), output_path=str(manifest),
    )
    return {
        "tng_galaxy_dir": str(tng_dir),
        "tng_properties_csv": str(properties),
        "tng_radius_manifest_path": str(manifest),
    }


class LegacyPrior:
    """A CosmosTngDraw prior driving the mass-sSFR transport donor pick."""

    def sample(self, rng):
        return CosmosTngDraw(
            catalog_id="legacy",
            mag_hst_f814w=23.0,
            target_vis_mag=23.1,
            target_vis_flux_e=1200.0,
            z=0.8,
            logmass=10.0,
            re_arcsec=float(rng.uniform(0.05, 0.5)),
            imputed_size=False,
            brightness_transfer="test",
            mass_quantile=float(rng.uniform(0.0, 1.0)),
            ssfr_quantile=float(rng.uniform(0.0, 1.0)),
            activity_class="star_forming",
            logssfr=-10.0,
        )


def joint_prior() -> JointGalaxyPopulationPrior:
    return JointGalaxyPopulationPrior(active_payload())


def make_simulator(
    paths: dict[str, str],
    prior=None,
    *,
    image_size: int = 192,
    off_field_padding: int = 68,
    star_density_arcmin2: float = 0.0,
) -> SkySimulator:
    """A simulator over the fake atlas; the joint (production) prior by default."""
    prior = joint_prior() if prior is None else prior
    density = float(getattr(prior, "surface_density_arcmin2", 300.0))
    return SkySimulator(
        prior,
        SkySimulatorConfig(
            image_size=image_size,
            pixel_scale=Config.DEFAULT_PIXEL_SCALE,
            galaxy_density_arcmin2=density,
            galaxy_off_field_padding_hr_pix=off_field_padding,
            star_density_arcmin2=star_density_arcmin2,
            lens_density_arcmin2=0.0,
            **paths,
        ),
    )


@pytest.fixture(scope="module")
def atlas_paths(tmp_path_factory) -> dict[str, str]:
    return write_fake_atlas(tmp_path_factory.mktemp("fake_atlas"))


def _fields(simulator: SkySimulator, seed: int, n: int, csv_path: Path):
    """Generate ``n`` fields from one stream; return pixels, metas, CSV text."""
    rng = np.random.default_rng(seed)
    pixels, metas = [], []
    with SourceCatalogWriter(str(csv_path)) as sources:
        for index in range(n):
            image, meta = simulator.simulate_field(rng)
            pixels.append(np.asarray(image.data).tobytes())
            metas.append(repr(meta))
            sources.add_field(index, meta)
    return pixels, metas, csv_path.read_text()


@pytest.mark.parametrize("prior_factory", [joint_prior, LegacyPrior])
def test_fields_are_bit_identical_to_the_linear_scan_eligibility(
    atlas_paths, tmp_path, monkeypatch, prior_factory,
):
    eligible_sizes = []
    original = TNGAtlas.eligible_indices

    def spy(self, target_re_arcsec, pixel_scale_arcsec):
        indices = original(self, target_re_arcsec, pixel_scale_arcsec)
        eligible_sizes.append(indices.size)
        return indices

    monkeypatch.setattr(TNGAtlas, "eligible_indices", spy)
    new = _fields(
        make_simulator(atlas_paths, prior_factory()), 41, 4, tmp_path / "new.csv",
    )
    monkeypatch.undo()

    # The donor support genuinely filtered some draws.
    assert eligible_sizes and min(eligible_sizes) < len(_DONOR_HALF_WIDTHS)

    monkeypatch.setattr(
        SkySimulator, "_eligible_morphology_indices",
        _old_eligible_morphology_indices,
    )
    monkeypatch.setattr(TNGAtlas, "_require_galaxy", _old_require_galaxy)
    monkeypatch.setattr(TNGAtlas, "eligible_galaxies", _old_eligible_galaxies)
    old = _fields(
        make_simulator(atlas_paths, prior_factory()), 41, 4, tmp_path / "old.csv",
    )

    assert new[0] == old[0]                     # pixels, byte for byte
    assert new[1] == old[1]                     # every metadata value
    assert new[2] == old[2]                     # source-sidecar rows
    assert sum(meta.count("'type': 'galaxy'") for meta in new[1]) >= 8


def test_reset_donor_balance_forgets_previous_fields(atlas_paths, tmp_path):
    fresh = make_simulator(atlas_paths)
    used = make_simulator(atlas_paths)
    _fields(used, 3, 3, tmp_path / "warmup.csv")
    assert used._morphology_use_counts.sum() > 0

    used.reset_donor_balance()

    assert not used._morphology_use_counts.any()
    assert used._morphology_use_counts.shape == (len(_DONOR_HALF_WIDTHS),)
    expected = _fields(fresh, 11, 3, tmp_path / "fresh.csv")
    actual = _fields(used, 11, 3, tmp_path / "reset.csv")
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert actual[2] == expected[2]


def test_donor_balance_otherwise_carries_over(atlas_paths, tmp_path):
    """Without the reset, earlier picks change later donors (why it exists)."""
    fresh = make_simulator(atlas_paths)
    used = make_simulator(atlas_paths)
    _fields(used, 3, 3, tmp_path / "warmup.csv")

    expected = _fields(fresh, 11, 3, tmp_path / "fresh.csv")
    actual = _fields(used, 11, 3, tmp_path / "carried.csv")
    assert actual[1] != expected[1]


def test_reset_donor_balance_without_an_atlas_is_a_no_op():
    simulator = SkySimulator(
        None,
        SkySimulatorConfig(
            image_size=96,
            galaxy_density_arcmin2=0.0,
            star_density_arcmin2=0.0,
            lens_density_arcmin2=0.0,
        ),
    )
    simulator.reset_donor_balance()
    assert simulator._morphology_use_counts.shape == (0,)
