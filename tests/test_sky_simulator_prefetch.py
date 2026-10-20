"""The donor read-ahead: while one galaxy renders, the field looks ahead to
its next galaxy's donor view and reads its files in the background. It is
only a hint, so fields must come out exactly as without it, and the look-
ahead must guess the donor the real draw takes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from euclid_polish.sky.generation.sky_simulator import SkySimulator
from tests.test_sky_simulator_donor_balance import (
    _fields,
    make_simulator,
    write_fake_atlas,
)


@pytest.fixture(scope="module")
def atlas_paths(tmp_path_factory) -> dict[str, str]:
    return write_fake_atlas(tmp_path_factory.mktemp("prefetch_atlas"))


def test_fields_are_identical_with_and_without_the_read_ahead(
    atlas_paths, tmp_path, monkeypatch,
):
    ahead = make_simulator(atlas_paths)
    pixels_ahead, metas_ahead, csv_ahead = _fields(ahead, 11, 4, tmp_path / "a.csv")
    info = ahead.tng_renderer.cache_info()
    assert info["prefetch_hits"] > 0

    monkeypatch.setattr(SkySimulator, "_prefetch_next_donor", lambda *a, **k: None)
    plain = make_simulator(atlas_paths)
    pixels, metas, csv = _fields(plain, 11, 4, tmp_path / "b.csv")
    assert plain.tng_renderer.cache_info()["prefetch_hits"] == 0
    assert pixels_ahead == pixels
    assert csv_ahead == csv


def test_the_look_ahead_guesses_the_donor_the_draw_takes(atlas_paths, monkeypatch):
    simulator = make_simulator(atlas_paths)
    peeked, loaded = [], []
    renderer = simulator.tng_renderer
    real_prefetch = renderer.prefetch
    real_load = renderer._native_source_keyed

    def prefetch(view):
        peeked.append((view.subhalo_id, view.orientation))
        real_prefetch(view)

    def load(view):
        loaded.append((view.subhalo_id, view.orientation))
        return real_load(view)

    monkeypatch.setattr(renderer, "prefetch", prefetch)
    monkeypatch.setattr(renderer, "_native_source_keyed", load)
    rng = np.random.default_rng(5)
    for _ in range(3):
        simulator.simulate_field(rng)
    assert loaded
    info = renderer.cache_info()
    reads = info["prefetch_hits"] + info["prefetch_misses"]
    assert reads > 0
    # A donor read from disk is the one guessed, unless a galaxy before it
    # was redrawn (its stamp failed photometry) after the guess.
    assert info["prefetch_hits"] >= 0.9 * reads
    assert set(loaded) <= set(peeked)


def test_a_peek_leaves_the_donor_balance_alone(atlas_paths):
    simulator = make_simulator(atlas_paths)
    simulator.simulate_field(np.random.default_rng(2))
    counts = simulator._morphology_use_counts.copy()
    view = simulator._peek_tng_view(np.random.default_rng(99), position=None, off_field=False)
    assert view is not None
    assert np.array_equal(simulator._morphology_use_counts, counts)


def test_a_failing_look_ahead_never_stops_the_field(atlas_paths, monkeypatch):
    simulator = make_simulator(atlas_paths)

    def broken(*args, **kwargs):
        raise RuntimeError("look-ahead bug")

    monkeypatch.setattr(simulator, "_peek_tng_view", broken)
    image, meta = simulator.simulate_field(np.random.default_rng(3))
    assert meta["n_galaxies"] > 0 and np.isfinite(image.data).all()
