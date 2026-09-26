"""Euclid Q1 MER tile footprints, committed so coverage checks work offline.

``q1_mer_tiles.json`` (next to this module) holds the 352 VIS MER mosaic
polygons of Quick Release 1 (IRSA ``s_region``, built by
``scripts/build_q1_mer_tiles.py`` from the cached obscore table), each with
its position-derived Q1 field label (:func:`q1_field_for`) and, where the
noise campaign measured it, the scene-sized 4-band sky level
(``mer_noise_levels.json``) or the reason the tile was rejected.

The geometry helpers work on small spherical polygons (well under a
hemisphere): a polygon is projected gnomonically about its own centre, so RA
wrap-around and convergence of meridians are handled exactly, and the planar
tests (ray casting, edge distances, segment intersection) then apply.

A cutout at ``(ra, dec)`` must come from the tile whose polygon *contains*
the point — ideally deepest inside it (MER tiles overlap by a margin, so a
point near one tile's edge sits comfortably inside its neighbour). Choosing
the tile with the nearest centre instead (the old archive behaviour) returned
empty or partial cutouts at tile edges.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

TILES_PATH = Path(__file__).with_name("q1_mer_tiles.json")

Point = tuple[float, float]


@dataclass(frozen=True)
class Q1Tile:
    """One Q1 MER tile: id, centre, VIS footprint polygon and annotations."""

    tile: str
    ra: float
    dec: float
    polygon: tuple[Point, ...]
    field: str | None = None
    region: str | None = None
    levels_e: tuple[float, ...] | None = None
    level_position: Point | None = None
    rejected: str | None = None

    def to_dict(self) -> dict:
        return {
            "tile": self.tile, "ra": self.ra, "dec": self.dec,
            "polygon": [list(p) for p in self.polygon],
            "field": self.field, "region": self.region,
            "levels_e": None if self.levels_e is None else list(self.levels_e),
            "level_position": (None if self.level_position is None
                               else list(self.level_position)),
            "rejected": self.rejected,
        }


# ---------------------------------------------------------------------------
# spherical geometry on small polygons
# ---------------------------------------------------------------------------

def _unit(ra: float, dec: float) -> tuple[float, float, float]:
    r, d = math.radians(ra), math.radians(dec)
    return (math.cos(d) * math.cos(r), math.cos(d) * math.sin(r), math.sin(d))


def polygon_centre(polygon: Sequence[Point]) -> Point:
    """Centre of a small spherical polygon (normalised mean unit vector)."""
    xs = ys = zs = 0.0
    for ra, dec in polygon:
        x, y, z = _unit(ra, dec)
        xs, ys, zs = xs + x, ys + y, zs + z
    ra = math.degrees(math.atan2(ys, xs)) % 360.0
    dec = math.degrees(math.atan2(zs, math.hypot(xs, ys)))
    return ra, dec


def tangent_plane(ra: float, dec: float, ra0: float, dec0: float) -> Point | None:
    """Gnomonic (TAN) projection of ``(ra, dec)`` about ``(ra0, dec0)``, in
    degrees on the tangent plane; ``None`` on the far hemisphere."""
    r, d = math.radians(ra), math.radians(dec)
    r0, d0 = math.radians(ra0), math.radians(dec0)
    cos_c = math.sin(d0) * math.sin(d) + math.cos(d0) * math.cos(d) * math.cos(r - r0)
    if cos_c <= 1e-9:
        return None
    xi = math.cos(d) * math.sin(r - r0) / cos_c
    eta = (math.cos(d0) * math.sin(d) - math.sin(d0) * math.cos(d) * math.cos(r - r0)) / cos_c
    return math.degrees(xi), math.degrees(eta)


def _projected(polygon: Sequence[Point], ra0: float, dec0: float) -> list[Point] | None:
    points = [tangent_plane(ra, dec, ra0, dec0) for ra, dec in polygon]
    if any(p is None for p in points):
        return None
    return [p for p in points if p is not None]


def _inside_planar(x: float, y: float, vertices: Sequence[Point]) -> bool:
    inside = False
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < cross:
                inside = not inside
    return inside


def _segment_distance(px: float, py: float, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def point_in_polygon(ra: float, dec: float, polygon: Sequence[Point]) -> bool:
    """Whether ``(ra, dec)`` lies inside a small spherical polygon."""
    return polygon_margin_deg(ra, dec, polygon) > 0.0


def polygon_margin_deg(ra: float, dec: float, polygon: Sequence[Point]) -> float:
    """Signed distance (deg, tangent plane) from the point to the nearest
    polygon edge: positive inside, negative outside."""
    ra0, dec0 = polygon_centre(polygon)
    vertices = _projected(polygon, ra0, dec0)
    point = tangent_plane(ra, dec, ra0, dec0)
    if vertices is None or point is None or len(vertices) < 3:
        return -math.inf
    x, y = point
    distance = min(
        _segment_distance(x, y, vertices[i], vertices[(i + 1) % len(vertices)])
        for i in range(len(vertices))
    )
    return distance if _inside_planar(x, y, vertices) else -distance


def _segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    def orient(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    return (o1 * o2 <= 0) and (o3 * o4 <= 0) and not (o1 == o2 == 0)


def polygons_intersect(a: Sequence[Point], b: Sequence[Point]) -> bool:
    """Whether two small spherical polygons overlap (edge crossing or
    containment), tested on the tangent plane about ``a``'s centre."""
    ra0, dec0 = polygon_centre(a)
    pa, pb = _projected(a, ra0, dec0), _projected(b, ra0, dec0)
    if pa is None or pb is None or len(pa) < 3 or len(pb) < 3:
        return False
    for i in range(len(pa)):
        for j in range(len(pb)):
            if _segments_cross(pa[i], pa[(i + 1) % len(pa)],
                               pb[j], pb[(j + 1) % len(pb)]):
                return True
    return _inside_planar(*pb[0], pa) or _inside_planar(*pa[0], pb)


def square_polygon(ra: float, dec: float, side_arcsec: float) -> list[Point]:
    """North-up square of ``side_arcsec`` centred on ``(ra, dec)``
    (corners in the order SE, SW, NW, NE on the sky)."""
    half = side_arcsec / 7200.0
    cos_dec = max(math.cos(math.radians(dec)), 1e-6)
    dra = half / cos_dec
    return [((ra + dra) % 360.0, dec - half), ((ra - dra) % 360.0, dec - half),
            ((ra - dra) % 360.0, dec + half), ((ra + dra) % 360.0, dec + half)]


# ---------------------------------------------------------------------------
# the committed table
# ---------------------------------------------------------------------------

def _tile_from_row(row: dict) -> Q1Tile:
    levels = row.get("levels_e")
    position = row.get("level_position")
    return Q1Tile(
        tile=str(row["tile"]), ra=float(row["ra"]), dec=float(row["dec"]),
        polygon=tuple((float(p[0]), float(p[1])) for p in row["polygon"]),
        field=row.get("field"), region=row.get("region"),
        levels_e=None if levels is None else tuple(float(v) for v in levels),
        level_position=None if position is None else (float(position[0]), float(position[1])),
        rejected=row.get("rejected") or None,
    )


@lru_cache(maxsize=4)
def _load(path: str) -> tuple[Q1Tile, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(_tile_from_row(row) for row in payload["tiles"])


def load_tiles(path: str | Path | None = None) -> tuple[Q1Tile, ...]:
    """Every committed Q1 MER tile (cached per path)."""
    return _load(str(path or TILES_PATH))


def tile_by_id(tile_id: str) -> Q1Tile | None:
    return next((tile for tile in load_tiles() if tile.tile == str(tile_id)), None)


def _near(tile: Q1Tile, ra: float, dec: float) -> bool:
    # MER tiles span ~0.53 deg; skip the exact test for far-away centres.
    cos_dec = max(math.cos(math.radians(dec)), 1e-3)
    dra = abs((ra - tile.ra + 180.0) % 360.0 - 180.0) * cos_dec
    return dra < 1.5 and abs(dec - tile.dec) < 1.5


def _ranked(ra: float, dec: float, tiles: Iterable[Q1Tile] | None) -> list[tuple[float, Q1Tile]]:
    """``(margin, tile)`` of every tile containing the point: tiles with
    observed data first (a ``rejected`` tile was measured unobserved by the
    noise campaign), each group deepest-inside first."""
    scored = []
    for tile in (load_tiles() if tiles is None else tiles):
        if not _near(tile, ra, dec):
            continue
        margin = polygon_margin_deg(ra, dec, tile.polygon)
        if margin > 0:
            scored.append((margin, tile))
    scored.sort(key=lambda item: (item[1].rejected is not None, -item[0]))
    return scored


def tiles_containing(ra: float, dec: float, *,
                     tiles: Iterable[Q1Tile] | None = None) -> list[Q1Tile]:
    """Tiles whose polygon contains the point: observed tiles before
    ``rejected`` (unobserved) ones, each deepest-inside first."""
    return [tile for _margin, tile in _ranked(ra, dec, tiles)]


def best_tile(ra: float, dec: float, *, half_size_deg: float = 0.0,
              tiles: Iterable[Q1Tile] | None = None) -> Q1Tile | None:
    """The tile a cutout at the point should come from: the observed tile
    the point is deepest inside (a ``rejected`` one only when no observed
    tile contains the point), when that margin also holds a cutout of
    half-size ``half_size_deg`` (``None`` otherwise)."""
    scored = _ranked(ra, dec, tiles)
    if not scored:
        return None
    margin, tile = scored[0]
    # A square cutout of half-size h fits when the centre is >= h·√2 from
    # every edge in the worst case; requiring >= h is exact for the
    # north-aligned MER tiles and never rejects a cutout that fits.
    return tile if margin >= half_size_deg else None


def in_q1(ra: float, dec: float, *, tiles: Iterable[Q1Tile] | None = None) -> bool:
    """Whether a position is inside any Q1 MER tile footprint."""
    return bool(_ranked(ra, dec, tiles))


def observed(ra: float, dec: float, *, tiles: Iterable[Q1Tile] | None = None) -> bool:
    """Whether a position is inside a Q1 MER tile that is not ``rejected``
    (i.e. one the noise campaign did not measure as unobserved)."""
    return any(tile.rejected is None for _margin, tile in _ranked(ra, dec, tiles))


def tiles_bbox(tiles: Iterable[Q1Tile]) -> dict[str, float]:
    """RA/Dec bounding box of tile polygons (RA not unwrapped)."""
    points = [p for tile in tiles for p in tile.polygon]
    ras = [p[0] for p in points]
    decs = [p[1] for p in points]
    if not points:
        return {}
    return {"ra_min": min(ras), "ra_max": max(ras),
            "dec_min": min(decs), "dec_max": max(decs)}


__all__ = [
    "Q1Tile",
    "TILES_PATH",
    "best_tile",
    "in_q1",
    "load_tiles",
    "observed",
    "point_in_polygon",
    "polygon_centre",
    "polygon_margin_deg",
    "polygons_intersect",
    "square_polygon",
    "tangent_plane",
    "tile_by_id",
    "tiles_bbox",
    "tiles_containing",
]
