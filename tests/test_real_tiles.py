"""Real tile store (contract C9, plan WP-B2 T2): every real source behind one
``RealTile`` (LR electrons + celestial WCS), cheap listings, the poster WCS
construction, and caching a new 25.6″ tile from the containing Q1 tile."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from euclid_polish.config import Config
from euclid_polish.photometry import adu_per_s_to_electrons_factor
from euclid_polish.sky.observation import q1_mer_tiles
from euclid_polish.web.helpers import jwst_euclid, real_tiles
from tests import _real_fixtures as fx


@pytest.fixture
def store(tmp_path, monkeypatch):
    return fx.point_store(tmp_path, monkeypatch)


def test_sources_are_listed_even_when_empty(store):
    payload = real_tiles.sources_payload()
    ids = [item["id"] for item in payload["sources"]]
    assert ids == list(real_tiles.SOURCES)
    assert all(item["count"] == 0 and not item["ready"] for item in payload["sources"])


def test_nexus_tiles_list_with_ids_positions_polygons_and_jwst(store):
    fx.make_nexus_field(n_tiles=2)
    entries = real_tiles.list_entries("nexus")
    assert [entry.id for entry in entries] == ["f200w-0000", "f200w-0001"]
    first = entries[0]
    assert first.field == "EDF-N" and first.has_jwst and first.model_ready
    assert first.ref == "nexus/f200w-0000"
    # the polygon comes from the LR header and contains the tile centre
    assert q1_mer_tiles.point_in_polygon(first.ra, first.dec, first.polygon)
    tile = real_tiles.get_tile("nexus", "f200w-0001")
    assert tile.lr_e.shape == (40, 40, 4) and tile.lr_e.dtype == np.float32
    centre = WCS(tile.wcs_header).pixel_to_world_values(19.5, 19.5)
    assert centre[0] == pytest.approx(entries[1].ra, abs=1e-6)
    planes = real_tiles.jwst_planes(first)
    assert planes[0]["band"] == "F200W" and planes[0]["data"].shape == (133, 133)
    payload = first.to_dict()
    assert payload["tiers"] == ["lr", "jwst"] and payload["model_ready"]


def test_eval_objects_skip_synthetic_rows(store):
    sub = fx.make_eval_store()
    entries = real_tiles.list_entries("eval")
    assert [entry.id for entry in entries] == [sub]
    assert entries[0].extras["grade"] == "A" and entries[0].extras["kind"] == "lens"
    tile = real_tiles.get_tile("eval", sub)
    assert tile.lr_e.shape == (20, 20, 4)
    assert tile.wcs_header["CRVAL1"] == pytest.approx(268.30)


def test_poster_wcs_is_constructed_north_up_at_the_target(store):
    fx.make_poster(store["poster"])
    (entry,) = real_tiles.list_entries("poster")
    assert entry.id == "target_181255_test" and entry.extras["wcs_constructed"]
    tile = real_tiles.get_tile("poster", entry.id)
    wcs = WCS(tile.wcs_header)
    ra, dec = wcs.pixel_to_world_values(15.5, 15.5)                 # grid centre
    assert ra == pytest.approx(273.2308875, abs=1e-9) and dec == pytest.approx(68.3636556, abs=1e-9)
    assert wcs.pixel_scale_matrix[0, 0] < 0 < wcs.pixel_scale_matrix[1, 1]  # east left
    sr, header = real_tiles.poster_legacy_sr(entry)
    assert sr.shape == (64, 64, 4)
    sr_ra, sr_dec = WCS(header).celestial.pixel_to_world_values(31.5, 31.5)
    assert sr_ra == pytest.approx(ra, abs=1e-9) and sr_dec == pytest.approx(dec, abs=1e-9)


def test_legacy_real_field_lists_every_sub_tile_with_offset_wcs(store):
    field_id = fx.make_real_field(side=32, grid=2)
    entries = real_tiles.list_entries("field")
    assert [entry.id for entry in entries] == [f"{field_id}-{i:03d}" for i in range(4)]
    tile = real_tiles.get_tile("field", f"{field_id}-003")      # row 1, col 1
    source = WCS(fits.getheader(
        Config.EUCLID_INFERENCE_DIR + f"/real_fields/{field_id}/original_stack.fits")).celestial
    expected = source.pixel_to_world_values(32, 32)
    got = WCS(tile.wcs_header).pixel_to_world_values(0, 0)
    assert got[0] == pytest.approx(float(expected[0]), abs=1e-9)
    assert got[1] == pytest.approx(float(expected[1]), abs=1e-9)


def test_refs_parse_and_validate():
    assert real_tiles.parse_refs("nexus/f200w-0001, eval/abc,nexus/f200w-0001") == [
        ("nexus", "f200w-0001"), ("eval", "abc")]
    with pytest.raises(real_tiles.RealTileError):
        real_tiles.parse_refs("mars/1")
    with pytest.raises(real_tiles.RealTileError):
        real_tiles.parse_refs("nexus/../../etc")
    with pytest.raises(real_tiles.RealTileError) as info:
        real_tiles.get_entry("nexus", "f200w-9999")
    assert info.value.code == 404


def test_cache_tile_uses_the_containing_q1_tile_and_registers_nisp(store, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(jwst_euclid, "fetch_q1_cutout", fx.fake_cutout_writer(calls))
    manifest = real_tiles.cache_tile(fx.NEXUS_RA, fx.NEXUS_DEC)
    assert calls == list(Config.LR_INPUT_BAND_NAMES)
    assert manifest["euclid_tile_index"] == "102158584" and manifest["field"] == "EDF-N"
    directory = real_tiles.tiles_root() / manifest["id"]
    lr = np.load(directory / "lr_e.npy")
    assert lr.shape == (256, 256, 4) and lr.dtype == np.float32
    vis_factor = adu_per_s_to_electrons_factor(24.6, Config.get_band("VIS"))
    np.testing.assert_allclose(lr[..., 0], vis_factor, rtol=1e-6)
    h_factor = adu_per_s_to_electrons_factor(29.9, Config.get_band("H_E"))
    np.testing.assert_allclose(lr[..., 3], h_factor, rtol=1e-5)
    with fits.open(directory / "lr.fits") as hdul:
        assert hdul[0].data.shape == (4, 256, 256)
        wcs = WCS(hdul[0].header).celestial
    ra, dec = wcs.pixel_to_world_values(127.5, 127.5)
    assert abs(ra - fx.NEXUS_RA) * np.cos(np.radians(dec)) * 3600 < 0.1
    assert abs(dec - fx.NEXUS_DEC) * 3600 < 0.1
    (entry,) = real_tiles.list_entries("tile")                 # listing invalidated
    assert entry.id == manifest["id"] and entry.model_ready
    # raw bands are reused on a second call
    real_tiles.cache_tile(fx.NEXUS_RA, fx.NEXUS_DEC)
    assert calls == list(Config.LR_INPUT_BAND_NAMES)


def test_cache_tile_refuses_positions_outside_q1(store, monkeypatch):
    def never(**_kwargs):
        raise AssertionError("no archive request outside Q1")

    monkeypatch.setattr(jwst_euclid, "fetch_q1_cutout", never)
    with pytest.raises(ValueError, match="outside the Euclid Q1"):
        real_tiles.cache_tile(150.1, 2.2)                       # COSMOS


def test_cache_tile_reports_an_archive_failure(store, monkeypatch):
    monkeypatch.setattr(jwst_euclid, "fetch_q1_cutout",
                        lambda **_kwargs: (False, "archive down"))
    with pytest.raises(RuntimeError, match="archive down"):
        real_tiles.cache_tile(fx.NEXUS_RA, fx.NEXUS_DEC)
