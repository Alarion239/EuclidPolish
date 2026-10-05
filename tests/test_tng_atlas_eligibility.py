"""Donor eligibility on a FASRC-sized atlas: precomputed, exact, O(1) lookups.

Every field-galaxy draw asks the atlas which donors can render a target R_e
without enlargement. The answer (indices AND their ascending order) feeds
``rng.choice``, so the precomputed lookups must reproduce the original
linear-scan implementation exactly, which is copied verbatim below.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from euclid_polish.sky.generation import sky_simulator as sky_simulator_module
from euclid_polish.sky.generation.sky_simulator import SkySimulator
from euclid_polish.tng import (
    TNGAtlas,
    TNGGalaxy,
    TNGPropertyCatalog,
    TNGRadiusManifest,
)
from euclid_polish.tng.atlas import _positive_finite

PIXEL_SCALE = 0.05
N_GALAXIES = 1000


# ---------------------------------------------------------------------------
# Verbatim copies of the linear-scan implementation (HEAD 09ae9ef), rewired
# only to call each other instead of the live methods.
# ---------------------------------------------------------------------------

def _old_require_galaxy(self, galaxy: TNGGalaxy) -> TNGGalaxy:
    if not isinstance(galaxy, TNGGalaxy):
        raise TypeError("galaxy must be a TNGGalaxy")
    if galaxy not in self.galaxies:
        raise ValueError(
            f"TNG{galaxy.subhalo_id} is not a complete galaxy in this atlas"
        )
    return galaxy


def _old_max_native_re_px(self, galaxy: TNGGalaxy) -> float:
    selected = _old_require_galaxy(self, galaxy)
    return self.radii.max_radius(selected.subhalo_id)


def _old_eligible_galaxies(
    self,
    target_re_arcsec: float,
    pixel_scale_arcsec: float,
) -> tuple[TNGGalaxy, ...]:
    target = _positive_finite(target_re_arcsec, "target_re_arcsec")
    pixel_scale = _positive_finite(
        pixel_scale_arcsec, "pixel_scale_arcsec"
    )
    minimum_native_re_px = target / pixel_scale
    return tuple(
        galaxy
        for galaxy in self.galaxies
        if _old_max_native_re_px(self, galaxy) >= minimum_native_re_px
    )


def _old_eligible_morphology_indices(
    self,
    target_re_arcsec: float,
) -> np.ndarray:
    atlas = self.tng_atlas
    if (
        atlas is None
        or not atlas
        or not np.isfinite(target_re_arcsec)
        or target_re_arcsec <= 0.0
    ):
        raise ValueError("TNG shrink-only donor selection is unavailable")
    eligible_galaxies = _old_eligible_galaxies(
        atlas,
        target_re_arcsec,
        self.config.pixel_scale,
    )
    eligible_ids = {galaxy.subhalo_id for galaxy in eligible_galaxies}
    eligible = np.asarray(
        [
            index
            for index, galaxy in enumerate(atlas.galaxies)
            if galaxy.subhalo_id in eligible_ids
        ],
        dtype=np.int64,
    )
    if not eligible.size:
        maximum_arcsec = float(
            max(_old_max_native_re_px(atlas, galaxy) for galaxy in atlas)
            * self.config.pixel_scale
        )
        raise sky_simulator_module._NoRenderableTNGDonorError(
            f"no TNG donor can render R_e={target_re_arcsec:g} arcsec "
            f"without enlargement; atlas maximum is {maximum_arcsec:g} "
            "arcsec"
        )
    return eligible


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _fake_atlas(n: int = N_GALAXIES, seed: int = 20261005) -> TNGAtlas:
    """A FASRC-sized atlas: realistic path lengths, shuffled ids, ties."""
    rng = np.random.default_rng(seed)
    root = Path(
        "/n/netscratch/hernquist_lab/Everyone/euclid_polish/tng_skirt_atlas"
    )
    ids = rng.choice(np.arange(1, 900_000), size=n, replace=False)
    galaxies = tuple(
        TNGGalaxy(root / f"{int(gid):06d}", str(int(gid))) for gid in ids
    )
    # Quantised radii create exact ties between donors; the continuous part
    # covers the whole shrink-only range.
    radii = {}
    for galaxy in galaxies:
        views = rng.uniform(0.5, 40.0, size=5)
        if rng.random() < 0.3:
            views = np.round(views * 4.0) / 4.0
        for orientation, radius in enumerate(views, start=1):
            radii[(galaxy.subhalo_id, orientation)] = float(radius)
    return TNGAtlas(
        root=root,
        galaxies=galaxies,
        properties=TNGPropertyCatalog({}, (None, 0, 0)),
        radii=TNGRadiusManifest(radii, "f" * 64),
    )


@pytest.fixture(scope="module")
def atlas() -> TNGAtlas:
    return _fake_atlas()


def _targets(atlas: TNGAtlas) -> list[float]:
    """Random targets plus exact and one-ulp boundary targets."""
    rng = np.random.default_rng(7)
    maxima = sorted(
        atlas.radii.max_radius(galaxy.subhalo_id) for galaxy in atlas
    )
    boundary = [maxima[k] * PIXEL_SCALE for k in (0, 1, 250, 500, 998, 999)]
    targets = [
        1e-9,
        *rng.uniform(0.02, 2.1, size=12).tolist(),
        *boundary,
        *(float(np.nextafter(t, 0.0)) for t in boundary[2:4]),
        *(float(np.nextafter(t, np.inf)) for t in boundary[2:4]),
    ]
    return targets


def _simulator_with(atlas: TNGAtlas) -> SkySimulator:
    simulator = object.__new__(SkySimulator)
    simulator.tng_atlas = atlas
    simulator.config = SimpleNamespace(pixel_scale=PIXEL_SCALE)
    return simulator


# ---------------------------------------------------------------------------
# Bit-identity against the linear scan
# ---------------------------------------------------------------------------

def test_eligible_indices_match_the_linear_scan_exactly(atlas):
    for target in _targets(atlas):
        expected = _old_eligible_galaxies(atlas, target, PIXEL_SCALE)
        indices = atlas.eligible_indices(target, PIXEL_SCALE)

        assert np.all(np.diff(indices) > 0)              # ascending, unique
        assert tuple(atlas.galaxies[i] for i in indices) == expected
        assert atlas.eligible_galaxies(target, PIXEL_SCALE) == expected


def test_simulator_eligibility_matches_the_linear_scan_exactly(atlas):
    simulator = _simulator_with(atlas)
    for target in _targets(atlas):
        expected = _old_eligible_morphology_indices(simulator, target)
        actual = simulator._eligible_morphology_indices(target)

        assert actual.dtype == expected.dtype == np.int64
        np.testing.assert_array_equal(actual, expected)


def test_no_donor_error_is_unchanged(atlas):
    simulator = _simulator_with(atlas)
    with pytest.raises(
        sky_simulator_module._NoRenderableTNGDonorError,
    ) as old:
        _old_eligible_morphology_indices(simulator, 5.0)
    with pytest.raises(
        sky_simulator_module._NoRenderableTNGDonorError,
    ) as new:
        simulator._eligible_morphology_indices(5.0)
    assert str(new.value) == str(old.value)


@pytest.mark.parametrize("target", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_targets_are_still_rejected(atlas, target):
    with pytest.raises(ValueError):
        atlas.eligible_indices(target, PIXEL_SCALE)
    with pytest.raises(ValueError):
        _simulator_with(atlas)._eligible_morphology_indices(target)


# ---------------------------------------------------------------------------
# Cost: no per-galaxy work on the hot path
# ---------------------------------------------------------------------------

def test_eligibility_does_no_per_galaxy_lookups(atlas, monkeypatch):
    calls = {"max_radius": 0, "eq": 0}
    original_max_radius = TNGRadiusManifest.max_radius
    original_eq = TNGGalaxy.__eq__

    def counting_max_radius(self, subhalo_id):
        calls["max_radius"] += 1
        return original_max_radius(self, subhalo_id)

    def counting_eq(self, other):
        calls["eq"] += 1
        return original_eq(self, other)

    monkeypatch.setattr(TNGRadiusManifest, "max_radius", counting_max_radius)
    monkeypatch.setattr(TNGGalaxy, "__eq__", counting_eq)

    atlas.eligible_indices(0.4, PIXEL_SCALE)
    _simulator_with(atlas)._eligible_morphology_indices(0.4)
    assert calls == {"max_radius": 0, "eq": 0}

    # Membership is a hash lookup: an equal, non-identical copy of the LAST
    # galaxy costs one comparison, where the tuple scan cost N.
    last = atlas.galaxies[-1]
    copy = TNGGalaxy(Path(str(last.directory)), last.subhalo_id)
    assert copy is not last
    views = atlas.eligible_views(copy, 1e-6, PIXEL_SCALE)
    assert len(views) == 5
    assert calls["eq"] <= 6                       # one per _require_galaxy


# ---------------------------------------------------------------------------
# Membership semantics are unchanged
# ---------------------------------------------------------------------------

def test_require_galaxy_semantics(atlas):
    first = atlas.galaxies[0]
    equal = TNGGalaxy(Path(str(first.directory)), first.subhalo_id)
    assert atlas.view(equal, 3).subhalo_id == first.subhalo_id
    assert atlas.max_native_re_px(equal) == atlas.radii.max_radius(
        first.subhalo_id
    )

    foreign = TNGGalaxy(Path("/elsewhere") / first.subhalo_id, first.subhalo_id)
    with pytest.raises(ValueError, match="not a complete galaxy"):
        atlas.view(foreign, 1)
    with pytest.raises(ValueError, match="not a complete galaxy"):
        atlas.max_native_re_px(foreign)
    with pytest.raises(TypeError, match="must be a TNGGalaxy"):
        atlas.eligible_views(first.subhalo_id, 0.1, PIXEL_SCALE)


def test_precomputed_lookups_do_not_change_atlas_equality(atlas):
    rebuilt = TNGAtlas(
        root=atlas.root,
        galaxies=atlas.galaxies,
        properties=atlas.properties,
        radii=atlas.radii,
    )
    assert rebuilt == atlas


def test_empty_atlas_has_no_eligible_donors(tmp_path):
    empty = TNGAtlas(
        root=tmp_path,
        galaxies=(),
        properties=TNGPropertyCatalog({}, (None, 0, 0)),
        radii=TNGRadiusManifest({}, "e" * 64),
    )
    assert empty.eligible_indices(0.1, PIXEL_SCALE).size == 0
    assert empty.eligible_galaxies(0.1, PIXEL_SCALE) == ()
