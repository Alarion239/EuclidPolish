"""Real-tile listings and sky-atlas layers are memoised on the on-disk stamp of
what they read — never on a short timer. An atlas click (``/api/sky/at``)
and a layers load stay warm (no lister re-run, no FITS headers re-read)
until something they depend on changes; a change is seen without an
explicit ``invalidate()``."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.web.helpers import jwst_euclid, model_catalog, real_tiles, sky_atlas
from tests import _real_fixtures as fx


@pytest.fixture
def world(tmp_path, monkeypatch):
    store = fx.point_store(tmp_path, monkeypatch)
    fx.stub_regime(tmp_path, monkeypatch)
    identifier = fx.make_nexus_field(n_tiles=2)
    fx.make_poster(store["poster"])
    return {**store, "nexus_field": identifier}


def _counting(monkeypatch, source: str) -> list[int]:
    calls: list[int] = []
    lister = real_tiles._LISTERS[source]

    def counted():
        calls.append(1)
        return lister()

    monkeypatch.setitem(real_tiles._LISTERS, source, counted)
    return calls


def _later(monkeypatch, seconds: float) -> None:
    """Pretend ``seconds`` passed (any leftover timer would expire)."""
    now = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: now + seconds)


def test_an_unchanged_listing_is_never_re_read(world, monkeypatch):
    calls = _counting(monkeypatch, "nexus")
    assert len(real_tiles.list_entries("nexus")) == 2
    _later(monkeypatch, 3600.0)
    assert len(real_tiles.list_entries("nexus")) == 2
    assert len(calls) == 1


def test_a_rewritten_manifest_is_seen_without_invalidate(world, monkeypatch):
    calls = _counting(monkeypatch, "nexus")
    assert len(real_tiles.list_entries("nexus")) == 2
    path = jwst_euclid.nexus_field_root() / world["nexus_field"] / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["tiles"] = manifest["tiles"][:1]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert len(real_tiles.list_entries("nexus")) == 1
    assert len(calls) == 2


def test_a_new_cached_tile_directory_is_seen(world, monkeypatch):
    assert real_tiles.list_entries("tile") == []
    directory = real_tiles.tiles_root() / "ra268.46250_dec+65.19917"
    directory.mkdir(parents=True)
    np.save(directory / "lr_e.npy", np.zeros((4, 4, 4), np.float32))
    (directory / "manifest.json").write_text(json.dumps(
        {"ra": fx.NEXUS_RA, "dec": fx.NEXUS_DEC, "shape": [4, 4, 4]}), encoding="utf-8")
    assert [entry.id for entry in real_tiles.list_entries("tile")] == [directory.name]


def test_a_new_output_of_an_existing_tile_updates_its_layer_state(world, monkeypatch):
    first = sky_atlas.layer_features("nexus-tiles")
    assert {f["props"]["state"] for f in first["features"]} == {"missing"}
    _later(monkeypatch, 3600.0)
    assert sky_atlas.layer_features("nexus-tiles") is first          # warm, no timer
    # A nested write (outputs/<source>/<id>/<slug>.json) for a tile id whose
    # directory already exists must still be seen.
    entry = real_tiles.list_entries("nexus")[0]
    spec = model_catalog.resolve_spec(model_catalog.SPEC_PRODUCTION)
    directory = model_catalog.output_dir("nexus", entry.id)
    directory.mkdir(parents=True)
    (directory / "placeholder").write_text("x", encoding="utf-8")
    assert sky_atlas.layer_features("nexus-tiles") is not first
    model_catalog.save_output("nexus", entry.id, spec, np.zeros((8, 8, 4), np.float32),
                              lr_header=None, lr_sha=None)
    states = {f["id"]: f["props"]["state"]
              for f in sky_atlas.layer_features("nexus-tiles")["features"]}
    assert states[entry.id] == "current"


def test_warm_layers_and_point_lookups_do_not_re_list(world, monkeypatch):
    sky_atlas.layers_payload()
    sky_atlas.at(fx.NEXUS_RA, fx.NEXUS_DEC)
    counts = {source: _counting(monkeypatch, source) for source in real_tiles.SOURCES}
    _later(monkeypatch, 3600.0)
    sky_atlas.layers_payload()
    at = sky_atlas.at(fx.NEXUS_RA, fx.NEXUS_DEC)
    assert at["nexus"], at
    assert {source: len(calls) for source, calls in counts.items()} == dict.fromkeys(
        real_tiles.SOURCES, 0)


def test_source_stamp_changes_with_nested_files(world):
    before = real_tiles.source_stamp("nexus")
    assert real_tiles.source_stamp("nexus") == before
    tiles = jwst_euclid.nexus_field_root() / world["nexus_field"] / "tiles"
    (tiles / "starfull_combiner_0000.fits").write_bytes(b"x")
    assert real_tiles.source_stamp("nexus") != before


def test_eval_listing_sees_a_per_object_file_without_a_manifest_change(world, monkeypatch):
    sub = fx.make_eval_store()
    [entry] = real_tiles.list_entries("eval")
    assert entry.extras["legacy_sr"] is None
    directory = Path(Config.EVAL_RESULTS_DIR) / sub
    (directory / "SR.fits").write_bytes(b"x")          # manifest.csv untouched
    [entry] = real_tiles.list_entries("eval")
    assert entry.extras["legacy_sr"] is not None
