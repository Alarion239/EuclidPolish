"""Committed Q1 MER tile footprints (``q1_mer_tiles.json``) and the spherical
point-in-polygon / tile-choice helpers that make Euclid coverage checks work
offline (contract C9, plan WP-B2 T1)."""

from __future__ import annotations

import json

import pytest

from euclid_polish.sky.observation import q1_mer_tiles as q1
from euclid_polish.sky.observation.q1_fields import q1_field_for

NEXUS_CENTRE = (268.4625, 65.19917)
JADES = (53.16, -27.78)
COSMOS = (150.1, 2.2)


def test_committed_table_has_every_q1_vis_tile():
    payload = json.loads(q1.TILES_PATH.read_text(encoding="utf-8"))
    assert payload["kind"] == "q1_mer_tiles"
    tiles = q1.load_tiles()
    assert len(tiles) == payload["count"] == 352
    assert len({tile.tile for tile in tiles}) == 352
    for tile in tiles:
        assert len(tile.polygon) == 4
        assert tile.tile.isdigit()
        # the label is the position-derived Q1 field, never a stored string
        assert tile.field == q1_field_for(tile.ra, tile.dec)
        assert q1.point_in_polygon(tile.ra, tile.dec, tile.polygon)


def test_noise_levels_and_rejections_are_joined():
    tiles = q1.load_tiles()
    with_levels = [tile for tile in tiles if tile.levels_e is not None]
    rejected = [tile for tile in tiles if tile.rejected]
    assert len(with_levels) == 294
    assert len(rejected) == 50
    assert all(len(tile.levels_e) == 4 for tile in with_levels)
    assert all(tile.levels_e is None for tile in rejected)
    ldn = [tile for tile in tiles if tile.region == "LDN1641"]
    assert len(ldn) == 8 and all(tile.field is None for tile in ldn)
    by_field = {name: sum(tile.field == name for tile in tiles)
                for name in ("EDF-N", "EDF-S", "EDF-F")}
    assert by_field == {"EDF-N": 124, "EDF-S": 148, "EDF-F": 72}


@pytest.mark.parametrize("position,tile_id", [
    (NEXUS_CENTRE, "102158584"),
    (JADES, "102044185"),
])
def test_known_positions_fall_in_their_tiles(position, tile_id):
    ids = [tile.tile for tile in q1.tiles_containing(*position)]
    assert tile_id in ids
    assert q1.in_q1(*position)


def test_cosmos_is_outside_q1():
    assert q1.tiles_containing(*COSMOS) == []
    assert not q1.in_q1(*COSMOS)
    assert q1.best_tile(*COSMOS) is None


def test_point_in_polygon_square_and_ra_wrap():
    square = [(10.0, -1.0), (12.0, -1.0), (12.0, 1.0), (10.0, 1.0)]
    assert q1.point_in_polygon(11.0, 0.0, square)
    assert not q1.point_in_polygon(13.0, 0.0, square)
    assert not q1.point_in_polygon(11.0, 1.5, square)
    wrap = [(359.0, -1.0), (1.0, -1.0), (1.0, 1.0), (359.0, 1.0)]
    assert q1.point_in_polygon(0.2, 0.0, wrap)
    assert q1.point_in_polygon(359.8, 0.5, wrap)
    assert not q1.point_in_polygon(180.0, 0.0, wrap)


def test_margin_is_positive_inside_and_measures_the_nearest_edge():
    square = [(10.0, -1.0), (12.0, -1.0), (12.0, 1.0), (10.0, 1.0)]
    assert q1.polygon_margin_deg(11.0, 0.0, square) == pytest.approx(1.0, abs=1e-3)
    assert q1.polygon_margin_deg(11.9, 0.0, square) == pytest.approx(0.1, abs=1e-3)
    assert q1.polygon_margin_deg(12.5, 0.0, square) < 0


def test_best_tile_prefers_the_tile_the_point_is_deepest_inside():
    left = q1.Q1Tile(tile="1", ra=10.5, dec=0.0,
                     polygon=((10.0, -1.0), (11.0, -1.0), (11.0, 1.0), (10.0, 1.0)))
    right = q1.Q1Tile(tile="2", ra=11.4, dec=0.0,
                      polygon=((10.8, -1.0), (12.0, -1.0), (12.0, 1.0), (10.8, 1.0)))
    # 10.95 is 0.05 deg inside `left` but 0.15 deg inside `right`.
    assert q1.best_tile(10.95, 0.0, tiles=(left, right)).tile == "2"
    assert [t.tile for t in q1.tiles_containing(10.95, 0.0, tiles=(left, right))] == ["2", "1"]
    # A cutout must fit: with a 0.1 deg half-size only `right` contains it.
    assert q1.best_tile(10.95, 0.0, tiles=(left, right), half_size_deg=0.1).tile == "2"
    assert q1.best_tile(10.5, 0.0, tiles=(left,), half_size_deg=0.6) is None


def test_observed_tiles_rank_before_rejected_ones():
    """A tile the noise campaign measured as unobserved (``rejected``) only
    wins when no observed tile contains the point."""
    empty = q1.Q1Tile(tile="1", ra=10.5, dec=0.0,
                      polygon=((10.0, -1.0), (11.0, -1.0), (11.0, 1.0), (10.0, 1.0)),
                      rejected="no coverage: only 0% of the cutout is observed")
    observed = q1.Q1Tile(tile="2", ra=11.4, dec=0.0,
                         polygon=((10.8, -1.0), (12.0, -1.0), (12.0, 1.0), (10.8, 1.0)))
    # 10.85 is 0.15 deg inside the rejected tile but only 0.05 deg inside the observed one.
    assert [t.tile for t in q1.tiles_containing(10.85, 0.0, tiles=(empty, observed))] == ["2", "1"]
    assert q1.best_tile(10.85, 0.0, tiles=(empty, observed)).tile == "2"
    # the observed tile cannot hold a 0.1 deg half-size cutout: no silent switch to the empty one
    assert q1.best_tile(10.85, 0.0, tiles=(empty, observed), half_size_deg=0.1) is None
    assert q1.best_tile(10.5, 0.0, tiles=(empty, observed)).tile == "1"   # only the empty one
    assert q1.in_q1(10.5, 0.0, tiles=(empty, observed))
    assert not q1.observed(10.5, 0.0, tiles=(empty, observed))
    assert q1.observed(10.85, 0.0, tiles=(empty, observed))


def test_every_committed_rejected_tile_centre_is_unobserved():
    rejected = [tile for tile in q1.load_tiles() if tile.rejected]
    assert len(rejected) == 50
    tile = rejected[0]
    assert q1.in_q1(tile.ra, tile.dec) and not q1.observed(tile.ra, tile.dec)


def test_polygons_intersect():
    a = [(10.0, 0.0), (11.0, 0.0), (11.0, 1.0), (10.0, 1.0)]
    b = [(10.5, 0.5), (11.5, 0.5), (11.5, 1.5), (10.5, 1.5)]
    inner = [(10.2, 0.2), (10.4, 0.2), (10.4, 0.4), (10.2, 0.4)]
    far = [(20.0, 0.0), (21.0, 0.0), (21.0, 1.0), (20.0, 1.0)]
    assert q1.polygons_intersect(a, b)
    assert q1.polygons_intersect(a, inner) and q1.polygons_intersect(inner, a)
    assert not q1.polygons_intersect(a, far)


def test_polygon_centre_and_bbox():
    square = [(359.0, -1.0), (1.0, -1.0), (1.0, 1.0), (359.0, 1.0)]
    ra, dec = q1.polygon_centre(square)
    assert min(abs(ra), abs(ra - 360.0)) < 1e-6 and abs(dec) < 1e-6
    tiles = q1.load_tiles()
    bbox = q1.tiles_bbox(tiles)
    assert bbox["dec_min"] < -51 and bbox["dec_max"] > 69
