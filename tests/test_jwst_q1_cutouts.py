"""Euclid cutouts from the Q1 MER tile whose POLYGON contains the position
(never the nearest tile centre), the NEXUS manifest footprints, the pair LR
input and pair inference through a model spec (plan WP-B2 handoffs d/f)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astroquery.esa import jwst as esa_jwst

from euclid_polish.config import Config
from euclid_polish.sky.observation import q1_mer_tiles
from euclid_polish.web.helpers import jwst_euclid, model_catalog, real_field, real_tiles
from tests import _real_fixtures as fx


@pytest.fixture(autouse=True)
def _fresh_products():
    jwst_euclid._MOSAIC_PRODUCTS.clear()
    yield
    jwst_euclid._MOSAIC_PRODUCTS.clear()


def _two_tiles():
    # ``far`` contains the point but its centre is far away; ``near`` has the
    # nearer centre but does not contain the point (the old failure mode).
    far = q1_mer_tiles.Q1Tile(tile="111", ra=10.0, dec=0.0, polygon=(
        (9.0, -1.0), (10.6, -1.0), (10.6, 1.0), (9.0, 1.0)))
    near = q1_mer_tiles.Q1Tile(tile="222", ra=10.8, dec=0.0, polygon=(
        (10.7, -0.2), (10.9, -0.2), (10.9, 0.2), (10.7, 0.2)))
    return (far, near)


def test_cutouts_come_from_the_containing_tile(monkeypatch, tmp_path):
    monkeypatch.setattr(q1_mer_tiles, "load_tiles", lambda *_a: _two_tiles())
    requested = []
    monkeypatch.setattr(jwst_euclid, "_mosaic_product",
                        lambda tile, instrument, filter_name: (
                            {"file_path": "/data/euclid", "file_name": f"MOSAIC-{tile}.fits",
                             "tile_index": tile}, ""))

    def fake_get_cutout(**kwargs):
        requested.append(kwargs)
        Path(kwargs["output_file"]).write_bytes(b"x" * 2880)

    monkeypatch.setattr(jwst_euclid, "_euclid_get_cutout", fake_get_cutout)
    ok, error = jwst_euclid.fetch_q1_cutout(ra=10.55, dec=0.0, band_name="VIS",
                                            output_file=str(tmp_path / "vis.fits"),
                                            cutout_size_vis_pixels=256)
    assert ok, error
    (call,) = requested
    assert call["id"] == 111 and call["file_path"] == "/data/euclid/MOSAIC-111.fits"
    assert call["instrument"] == "VIS"
    assert call["radius"].to_value("arcsec") == pytest.approx(12.8)


def test_cutouts_outside_q1_are_refused_before_any_query(monkeypatch, tmp_path):
    monkeypatch.setattr(jwst_euclid, "_mosaic_product", lambda *a: pytest.fail("queried"))
    ok, error = jwst_euclid.fetch_q1_cutout(ra=150.1, dec=2.2, band_name="H_E",
                                            output_file=str(tmp_path / "h.fits"),
                                            cutout_size_vis_pixels=264)
    assert not ok and "outside the Q1" in error


def test_mosaic_product_is_resolved_by_tile_index_and_cached(monkeypatch):
    queries = []

    def fake_query(query):
        queries.append(query)
        return Table(rows=[
            ("/p", "OTHER.fits", 102158584, "Q1_R1", "DpdMerFlagMosaic", "IMAGE"),
            ("/p", "BGSUB-MOSAIC-NIR-H.fits", 102158584, "Q1_R1", "DpdMerBksMosaic", "IMAGE"),
        ], names=("file_path", "file_name", "tile_index", "release_name", "product_type",
                  "technique")), ""

    monkeypatch.setattr(jwst_euclid, "query_mosaic_tiles", fake_query)
    product, error = jwst_euclid._mosaic_product("102158584", "NISP", "NIR_H")
    assert error == "" and product["file_name"] == "BGSUB-MOSAIC-NIR-H.fits"
    assert "tile_index = 102158584" in queries[0] and "filter_name = 'NIR_H'" in queries[0]
    assert "instrument_name = 'NISP'" in queries[0]
    again, _ = jwst_euclid._mosaic_product("102158584", "NISP", "NIR_H")
    assert again == product and len(queries) == 1
    assert jwst_euclid._mosaic_product("../1", "VIS", None)[0] is None
    monkeypatch.setattr(jwst_euclid, "query_mosaic_tiles", lambda q: (None, "TAP down"))
    product, error = jwst_euclid._mosaic_product("102158585", "VIS", None)
    assert product is None and "TAP down" in error


def test_nexus_download_records_polygons_tile_ids_and_footprint(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path / "data"))
    header = fx.wcs_header(fx.NEXUS_RA, fx.NEXUS_DEC, 0.03, (850, 850))
    wcs = WCS(header)
    source = np.ones((850, 850), dtype=np.float32)
    monkeypatch.setattr(jwst_euclid, "_download_nexus_mosaic", lambda *a, **k: tmp_path / "m.fits")
    monkeypatch.setattr(jwst_euclid, "_nexus_source_tiles",
                        lambda *a, **k: (source, header, wcs, [(0, 0, 850, 850)]))

    def write_jwst(_data, _header, _wcs, _bounds, destination, **_kwargs):
        fits.PrimaryHDU(source, header=header).writeto(destination, overwrite=True)
        return source, header, wcs, fx.NEXUS_RA, fx.NEXUS_DEC

    monkeypatch.setattr(jwst_euclid, "_write_nexus_source_tile", write_jwst)
    monkeypatch.setattr(jwst_euclid, "fetch_q1_cutout", fx.fake_cutout_writer())
    manifest = jwst_euclid.download_nexus_field(filter_name="F200W")
    (tile,) = manifest["tiles"]
    assert tile["euclid_tile_index"] == "102158584"
    assert len(tile["polygon"]) == 4
    assert q1_mer_tiles.point_in_polygon(tile["ra_deg"], tile["dec_deg"], tile["polygon"])
    assert len(manifest["footprint"]) == 4
    polygons = jwst_euclid.nexus_tile_polygons(manifest["field_id"])
    assert polygons[0] == tile["polygon"]


def _saved_pair(root: Path, identifier: str = "pair-1") -> str:
    directory = jwst_euclid.pair_root() / identifier
    directory.mkdir(parents=True)
    header = fx.wcs_header(fx.NEXUS_RA, fx.NEXUS_DEC, 0.1, (300, 300), bunit="ADU/s")
    header["MAGZERO"] = 24.6
    fits.PrimaryHDU(np.ones((300, 300), np.float32), header=header).writeto(
        directory / "euclid_vis.fits")
    fits.PrimaryHDU(np.ones((1000, 1000), np.float32),
                    header=fx.wcs_header(fx.NEXUS_RA, fx.NEXUS_DEC, 0.03, (1000, 1000),
                                         bunit="MJy/sr")).writeto(directory / "jwst_native.fits")
    for name in ("euclid_vis.png", "jwst_native.png"):
        (directory / name).write_bytes(b"png")
    (directory / "manifest.json").write_text(json.dumps({
        "field_id": identifier, "ra_deg": fx.NEXUS_RA, "dec_deg": fx.NEXUS_DEC,
        "size_arcsec": 25.6, "target_name": "test pair",
        "files": {"euclid": "euclid_vis.fits", "jwst_native": "jwst_native.fits",
                  "euclid_png": "euclid_vis.png", "jwst_png": "jwst_native.png"},
        "jwst_bands": [{"filter": "F200W", "file": "jwst_native.fits"}],
    }), encoding="utf-8")
    return identifier


def test_pair_lr_input_and_inference_through_a_spec(tmp_path, monkeypatch):
    fx.point_store(tmp_path, monkeypatch)
    fx.stub_regime(tmp_path, monkeypatch)
    identifier = _saved_pair(tmp_path)
    (entry,) = real_tiles.list_entries("pair")
    assert not entry.model_ready and entry.bands == ("VIS",)
    calls: list[str] = []
    monkeypatch.setattr(jwst_euclid, "fetch_q1_cutout", fx.fake_cutout_writer(calls))
    cube, header, manifest = jwst_euclid.pair_lr_input(identifier)
    assert cube.shape == (256, 256, 4) and calls == list(Config.LR_INPUT_BAND_NAMES)
    assert manifest["lr_input"]["euclid_tile_index"] == "102158584"
    assert np.all(np.isfinite(cube))
    real_tiles.invalidate()
    (entry,) = real_tiles.list_entries("pair")
    assert entry.model_ready and entry.has_jwst
    # a second call reuses the cached input
    jwst_euclid.pair_lr_input(identifier)
    assert len(calls) == 4
    result = jwst_euclid.run_starfull_pair_inference(identifier, runner=fx.StubRunner())
    record = result["inference"]
    assert record["spec"] == "production" and record["combiner_kind"] == "spatial_gate"
    sr_path = jwst_euclid.pair_root() / identifier / record["files"]["starfull"]
    sr_header = fits.getheader(sr_path)
    lr_wcs = WCS(header).celestial
    expected = lr_wcs.pixel_to_world_values(-0.25, -0.25)
    got = WCS(sr_header).celestial.pixel_to_world_values(0, 0)
    assert abs(float(got[0]) - float(expected[0])) < 1e-9
    assert abs(float(got[1]) - float(expected[1])) < 1e-9
    other = jwst_euclid.run_starfull_pair_inference(identifier, spec="member:member_2",
                                                    runner=fx.StubRunner())
    assert "member:member_2" in other["model_inference"]
    with pytest.raises(RuntimeError, match="unavailable|unknown"):
        jwst_euclid.run_starfull_pair_inference(identifier, spec="rbf",
                                                runner=fx.StubRunner())
    assert model_catalog.resolve_spec("production").available


def test_archive_clients_and_discovery_helpers_are_module_imports():
    """No by-path script execution and no function-scoped astroquery import:
    the ESA JWST client and the discovery script are ordinary module-top
    imports (user rule), shared with every other importer."""
    assert jwst_euclid.Jwst is esa_jwst.Jwst
    assert jwst_euclid.overlap_discovery is sys.modules["scripts.find_jwst_euclid_overlap"]
    assert "euclid_polish_jwst_overlap_discovery" not in sys.modules


def test_esa_jwst_download_uses_the_module_client(tmp_path, monkeypatch):
    product = tmp_path / "jw_i2d.fits"
    fits.PrimaryHDU(np.ones((4, 4), np.float32)).writeto(product)
    calls: list[tuple] = []

    class FakeJwst:
        def get_product_list(self, **kwargs):
            calls.append(("list", kwargs["observation_id"]))
            return Table({"filename": ["jw_uncal.fits", "jw_i2d.fits"]})

        def get_product(self, *, file_name):
            calls.append(("get", file_name))
            return str(product)

    monkeypatch.setattr(jwst_euclid, "Jwst", FakeJwst())
    destination = tmp_path / "out" / "jwst.fits"
    destination.parent.mkdir()
    assert jwst_euclid._download_jwst_esa("jw01", destination) == "jw_i2d.fits"
    assert calls == [("list", "jw01"), ("get", "jw_i2d.fits")]
    assert jwst_euclid._is_readable_fits(destination)


def test_mast_scope_query_shares_the_scripts_cache(tmp_path, monkeypatch):
    queries: list[dict] = []

    class FakeObservations:
        @staticmethod
        def query_criteria(**kwargs):
            queries.append(kwargs)
            return Table({"obs_id": ["jw1"], "s_ra": [53.1], "s_dec": [-27.8]})

    monkeypatch.setattr(jwst_euclid, "Observations", FakeObservations)
    scope = {"tile_ids": ["102044185"], "query_radius_deg": 0.55,
             "center_ra": 53.1, "center_dec": -27.8}
    rows = jwst_euclid._mast_rows_for_scope(scope, cache_dir=tmp_path, refresh=False)
    assert rows == [{"obs_id": "jw1", "s_ra": 53.1, "s_dec": -27.8}]
    assert queries[0]["obs_collection"] == "JWST" and queries[0]["dataproduct_type"] == "image"
    key = jwst_euclid.overlap_discovery._cache_key("['102044185']:0.550000")
    assert (tmp_path / "mast" / f"scope_{key}.json").is_file()
    assert jwst_euclid._mast_rows_for_scope(scope, cache_dir=tmp_path, refresh=False) == rows
    assert len(queries) == 1                                   # served from the cache
    jwst_euclid._mast_rows_for_scope(scope, cache_dir=tmp_path, refresh=True)
    assert len(queries) == 2


def _pair_world(tmp_path, monkeypatch, *, coverage_by_tile):
    """A pairing run whose Euclid VIS comes from the committed polygons:
    ``coverage_by_tile`` = fraction of the requested box each tile observes."""
    monkeypatch.setattr(Config, "DATA_DIR", str(tmp_path / "data"))
    ra, dec = 53.16, -27.78
    tiles = [q1_mer_tiles.Q1Tile(tile=tile, ra=ra, dec=dec,
                                 polygon=tuple(q1_mer_tiles.square_polygon(ra, dec, 3600.0 - i)))
             for i, tile in enumerate(coverage_by_tile)]
    monkeypatch.setattr(jwst_euclid.q1_mer_tiles, "tiles_containing", lambda *_a, **_k: tiles)
    fetched: list[str] = []

    def fetch(*, ra, dec, band_name, output_file, cutout_size_vis_pixels, tile=None, **_kw):
        assert band_name == "VIS" and tile is not None
        fetched.append(tile.tile)
        side = cutout_size_vis_pixels
        data = np.ones((side, side), np.float32)
        observed = int(round(coverage_by_tile[tile.tile] * side))
        data[:, observed:] = 0.0                              # unobserved columns
        header = fx.wcs_header(ra, dec, 0.1, (side, side))
        header["MAGZERO"] = 24.6
        fits.PrimaryHDU(data, header=header).writeto(output_file, overwrite=True)
        return True, None

    def jwst(_archive, _obs, destination):
        fits.PrimaryHDU(np.ones((1200, 1200), np.float32),
                        header=fx.wcs_header(ra, dec, 0.03, (1200, 1200), bunit="MJy/sr")
                        ).writeto(destination)
        return "jw_i2d.fits"

    def no_archive(*_a, **_k):
        raise AssertionError("the archive INTERSECTS search is only a fallback outside Q1")

    monkeypatch.setattr(jwst_euclid, "fetch_q1_cutout", fetch)
    monkeypatch.setattr(jwst_euclid, "_download_jwst", jwst)
    monkeypatch.setattr(jwst_euclid, "euclid_tile", no_archive)
    monkeypatch.setattr(jwst_euclid, "euclid_tiles_covering", no_archive)
    row = {"euclid_tile_index": "999", "jwst_archive": "mast", "jwst_observation_id": "jw1",
           "jwst_ra_deg": ra, "jwst_dec_deg": dec, "jwst_filters": "F200W"}
    return row, fetched


def test_pair_vis_comes_from_the_containing_q1_tile_with_full_coverage(tmp_path, monkeypatch):
    # the deepest tile only half-covers the 30" box; the next one covers it fully
    row, fetched = _pair_world(tmp_path, monkeypatch,
                               coverage_by_tile={"1001": 0.5, "1002": 1.0})
    manifest = jwst_euclid.download_and_align_pair(row, size_arcsec=30.0)
    assert fetched == ["1001", "1002"]
    assert manifest["euclid_vis_tile_index"] == "1002"
    assert manifest["euclid_selection"]["method"] == "q1_polygon"
    assert manifest["euclid_selection"]["coverage"] == pytest.approx(1.0)
    vis = fits.getdata(jwst_euclid.pair_root() / manifest["field_id"] / "euclid_vis.fits")
    assert vis.shape == (300, 300) and np.all(vis != 0)


def test_pair_download_refuses_a_partial_euclid_box(tmp_path, monkeypatch):
    row, fetched = _pair_world(tmp_path, monkeypatch,
                               coverage_by_tile={"1001": 0.6, "1002": 0.4})
    with pytest.raises(RuntimeError, match="fully cover"):
        jwst_euclid.download_and_align_pair(row, size_arcsec=30.0)
    assert fetched == ["1001", "1002"]
    assert not any(jwst_euclid.pair_root().glob("mast-*"))       # nothing published


def test_nexus_pair_vis_also_needs_a_fully_covering_q1_tile(tmp_path, monkeypatch):
    row, fetched = _pair_world(tmp_path, monkeypatch,
                               coverage_by_tile={"1001": 0.3, "1002": 1.0})
    mosaic = tmp_path / "nexus.fits"
    fits.PrimaryHDU(np.ones((1200, 1200), np.float32),
                    header=fx.wcs_header(row["jwst_ra_deg"], row["jwst_dec_deg"], 0.03,
                                         (1200, 1200), bunit="MJy/sr")).writeto(mosaic)
    monkeypatch.setattr(jwst_euclid, "_download_nexus_mosaic", lambda *_a, **_k: mosaic)
    manifest = jwst_euclid.download_nexus_pair(ra=row["jwst_ra_deg"], dec=row["jwst_dec_deg"],
                                               filter_name="F200W", size_arcsec=20.0)
    assert fetched == ["1001", "1002"]
    assert manifest["euclid_selection"]["tile"] == "1002"
    assert manifest["euclid_vis_tile_index"] == "1002"


def test_legacy_real_field_bands_come_from_the_containing_q1_tile(tmp_path, monkeypatch):
    """The 256″ legacy real field downloads through the committed polygons too
    (never the nearest-centre ``catalog.downloader.fetch_cutout_at``)."""
    calls: list[str] = []
    monkeypatch.setattr(jwst_euclid, "fetch_q1_cutout",
                        fx.fake_cutout_writer(calls, nisp_scale=0.1, pad=0))
    cube = real_field._load_or_download_lr(fx.NEXUS_RA, fx.NEXUS_DEC, tmp_path,
                                           lambda *_a: None)
    assert calls == list(Config.LR_INPUT_BAND_NAMES)
    assert cube.shape == (real_field.FIELD_SIZE, real_field.FIELD_SIZE, 4)
