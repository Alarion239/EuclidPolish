"""Sky-atlas layers: every local sky dataset as compact, inspectable features.

``GET /api/sky/layers`` lists the layer catalogue; ``GET /api/sky/layer/<id>``
returns one layer's features in one of three compact shapes:

* ``points``   — ``{"kind": "points", "columns": [...], "rows": [[ra, dec, …]],
  "inspect": {"kind", "prefix", "id_column"}}`` (the inspector entity of row
  ``r`` is ``{kind}:{prefix}{r[id_column]}``, or the row index when
  ``id_column`` is null);
* ``polygons`` — ``{"kind": "polygons", "features": [{"id", "polygon":
  [[ra, dec], …], "props", "inspect": {"kind", "id"}}]}``;
* ``circles``  — ``{"kind": "circles", "features": [{"id", "ra", "dec",
  "radius_deg", "props", "inspect"}]}``.

Every builder reads local files only (works offline) and is memoised on the
mtimes of what it reads. Groups follow the atlas Layers panel (spec §7.1):
``coverage``, ``results``, ``catalogues``.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy.io import fits

from euclid_polish.config import Config
from euclid_polish.sky.observation import q1_mer_tiles
from euclid_polish.sky.observation.q1_fields import Q1_FIELDS, q1_field_for
from euclid_polish.web.helpers import experiments, jwst_euclid, model_catalog, real_tiles
from euclid_polish.web.helpers.status import _cached_fasrc_catalog_dir, _cached_psf_clusters_json

BAND_NAMES = tuple(Config.LR_INPUT_BAND_NAMES)
#: Star-row validity bits (valid cutout in the band at any size).
STAR_FLAG_BITS = {band: 1 << index for index, band in enumerate(BAND_NAMES)}


@dataclass(frozen=True)
class LayerSpec:
    id: str
    label: str
    group: str                      # coverage | results | catalogues
    kind: str                       # points | polygons | circles
    build: Callable[[], dict[str, Any]]
    stamp: Callable[[], tuple]
    style: Mapping[str, Any] = field(default_factory=dict)
    fill_action: Mapping[str, Any] | None = None
    description: str = ""


def _mtime(path: Path | str | None) -> float:
    if path is None:
        return 0.0
    try:
        return os.stat(path).st_mtime
    except OSError:
        return 0.0


def _dir_stamp(path: Path) -> tuple:
    if not path.is_dir():
        return (0.0,)
    return tuple(sorted((child.name, _mtime(child)) for child in path.iterdir()))


def _round(value: float | None, digits: int = 7) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _points(columns: list[str], rows: list[list[Any]], *, inspect_kind: str = "source",
            prefix: str = "", id_column: str | None = None) -> dict[str, Any]:
    return {"kind": "points", "columns": columns, "rows": rows,
            "inspect": {"kind": inspect_kind, "prefix": prefix, "id_column": id_column}}


def _polygons(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kind": "polygons", "features": features}


def _circles(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kind": "circles", "features": features}


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def _q1_tiles() -> dict[str, Any]:
    features = []
    for tile in q1_mer_tiles.load_tiles():
        features.append({
            "id": tile.tile, "polygon": [list(p) for p in tile.polygon],
            "props": {"tile": tile.tile, "field": tile.field, "region": tile.region,
                      "levels_e": list(tile.levels_e) if tile.levels_e else None,
                      "vis_level_e": tile.levels_e[0] if tile.levels_e else None,
                      "rejected": tile.rejected,
                      "state": "rejected" if tile.rejected else (
                          "measured" if tile.levels_e else "unmeasured")},
            "inspect": {"kind": "source", "id": f"q1-tiles/{tile.tile}"},
        })
    return _polygons(features)


def _q1_fields() -> dict[str, Any]:
    return _circles([{
        "id": item.name, "ra": item.ra, "dec": item.dec, "radius_deg": item.radius_deg,
        "props": {"name": item.name},
        "inspect": {"kind": "source", "id": f"q1-fields/{item.name}"},
    } for item in Q1_FIELDS])


def convex_hull(points: Iterable[tuple[float, float]]) -> list[list[float]]:
    """Convex hull of sky points (tangent plane about their centre), as a
    polygon ``[[ra, dec], …]``; ``[]`` for fewer than three points."""
    pts = [(float(ra), float(dec)) for ra, dec in points]
    if len(pts) < 3:
        return []
    ra0, dec0 = q1_mer_tiles.polygon_centre(pts)
    projected = sorted({(round(xy[0], 9), round(xy[1], 9)): p for p in pts
                        if (xy := q1_mer_tiles.tangent_plane(p[0], p[1], ra0, dec0))}.items())

    def cross(o, a, b) -> float:
        return (a[0][0] - o[0][0]) * (b[0][1] - o[0][1]) - (a[0][1] - o[0][1]) * (b[0][0] - o[0][0])

    lower: list = []
    for item in projected:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], item) <= 0:
            lower.pop()
        lower.append(item)
    upper: list = []
    for item in reversed(projected):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], item) <= 0:
            upper.pop()
        upper.append(item)
    hull = lower[:-1] + upper[:-1]
    return [[round(p[1][0], 7), round(p[1][1], 7)] for p in hull]


def _nexus_footprint() -> dict[str, Any]:
    features = []
    for field_id, manifest in real_tiles._nexus_manifests():
        polygons = jwst_euclid.nexus_tile_polygons(field_id)
        hull = convex_hull(p for polygon in polygons.values() for p in polygon)
        props = {"field_id": field_id, "filter": manifest.get("filter"),
                 "target": manifest.get("target_name"),
                 "tiles": len(manifest.get("tiles") or [])}
        if hull:
            features.append({
                "id": field_id, "polygon": hull,
                "props": {**props, "outline": "convex hull of the Euclid tile cells"},
                "inspect": {"kind": "source", "id": f"nexus-footprint/{field_id}"},
            })
        grid = manifest.get("footprint")
        if isinstance(grid, list) and len(grid) >= 3:
            features.append({
                "id": f"{field_id}:mosaic", "polygon": grid,
                "props": {**props, "outline": "JWST mosaic pixel grid (data covers ~30 %)"},
                "inspect": {"kind": "source", "id": f"nexus-footprint/{field_id}"},
            })
    return _polygons(features)


# ---------------------------------------------------------------------------
# real results
# ---------------------------------------------------------------------------

def production_fingerprint() -> str | None:
    """The production spec's fingerprint now (``None`` when it cannot run)."""
    try:
        spec = model_catalog.resolve_spec(model_catalog.SPEC_PRODUCTION)
    except KeyError:
        return None
    return spec.fingerprint if spec.available else None


def production_state(entry: real_tiles.TileEntry, fingerprint: str | None,
                     outputs: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    """``current`` | ``stale`` | ``missing`` production SR of a real tile.

    Decided on the tile's merged outputs (:func:`real_tiles.tile_outputs`:
    the C9 store and the legacy NEXUS / pair SRs alike): ``current`` when its
    ``production`` output carries the production fingerprint now; ``stale``
    when that output is older — or when only a legacy SR of the production
    pipeline exists (e.g. an RBF-era NEXUS SR, served as ``m:rbf``);
    ``missing`` otherwise.
    """
    if outputs is None:
        outputs = real_tiles.tile_outputs(entry, {model_catalog.SPEC_PRODUCTION: fingerprint})
    stored = outputs.get(model_catalog.SPEC_PRODUCTION)
    if stored is not None:
        return "current" if fingerprint and stored.get("fingerprint") == fingerprint else "stale"
    return "stale" if entry.extras.get("legacy_sr") else "missing"


def _entry_polygons(source: str) -> dict[str, Any]:
    fingerprint = production_fingerprint()
    current = {model_catalog.SPEC_PRODUCTION: fingerprint}
    features = []
    for entry in real_tiles.list_entries(source):
        if not entry.polygon:
            continue
        outputs = real_tiles.tile_outputs(entry, current)
        features.append({
            "id": entry.id, "polygon": entry.polygon,
            "props": {"label": entry.label, "ref": entry.ref, "field": entry.field,
                      "ra": _round(entry.ra), "dec": _round(entry.dec),
                      "has_jwst": entry.has_jwst, "model_ready": entry.model_ready,
                      "state": production_state(entry, fingerprint, outputs),
                      "models": sorted(outputs),
                      **{key: entry.extras.get(key) for key in (
                          "position_name", "stored_field", "grade", "field_id")
                         if entry.extras.get(key) is not None}},
            "inspect": {"kind": "realtile", "id": entry.ref},
        })
    return _polygons(features)


def _eval_objects() -> dict[str, Any]:
    rows = []
    for entry in real_tiles.list_entries("eval"):
        rows.append([_round(entry.ra), _round(entry.dec), entry.extras.get("grade") or "",
                     entry.extras.get("kind"), _round(entry.extras.get("flux_ratio_sr_over_lr"), 4),
                     entry.id])
    return _points(["ra", "dec", "grade", "kind", "flux_ratio_sr_over_lr", "id"], rows,
                   inspect_kind="realtile", prefix="eval/", id_column="id")


def _experiment_points() -> dict[str, Any]:
    by_ref: dict[str, list[str]] = {}
    for record in experiments.list_experiments():
        for ref in record.get("tiles") or []:
            by_ref.setdefault(str(ref), []).append(str(record.get("id")))
    rows = []
    for ref, ids in sorted(by_ref.items()):
        source, _, identifier = ref.partition("/")
        with contextlib.suppress(real_tiles.RealTileError, KeyError):
            entry = real_tiles.get_entry(source, identifier)
            rows.append([_round(entry.ra), _round(entry.dec), ref, len(ids), ids[0]])
    return _points(["ra", "dec", "ref", "experiments", "latest"], rows,
                   inspect_kind="realtile", id_column="ref")


# ---------------------------------------------------------------------------
# catalogues
# ---------------------------------------------------------------------------

def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def lens_catalog_path() -> Path:
    return Path(Config.EVAL_CATALOG_DIR) / "lens_catalog" / "lenses.csv"


def galaxy_catalog_path() -> Path:
    return Path(Config.EVAL_CATALOG_DIR) / "galaxy_catalog" / "galaxies.csv"


def _lenses() -> dict[str, Any]:
    rows = []
    for row in _csv_rows(lens_catalog_path()):
        ra, dec = _float(row.get("ra")), _float(row.get("dec"))
        if ra is None or dec is None:
            continue
        rows.append([_round(ra), _round(dec), row.get("grade") or "",
                     row.get("subset") or "", row.get("id") or "", q1_field_for(ra, dec)])
    return _points(["ra", "dec", "grade", "subset", "id", "field"], rows,
                   prefix="lens-candidates/", id_column="id")


def _galaxies() -> dict[str, Any]:
    rows = []
    for row in _csv_rows(galaxy_catalog_path()):
        ra, dec = _float(row.get("ra")), _float(row.get("dec"))
        if ra is not None and dec is not None:
            rows.append([_round(ra), _round(dec), row.get("id") or "", q1_field_for(ra, dec)])
    return _points(["ra", "dec", "id", "field"], rows, prefix="galaxies/", id_column="id")


def stars_path() -> Path | None:
    """The FASRC-mirror ``stars.csv`` (the 43k-star catalogue) — never the
    stale local ``data/euclid_stars/stars.csv`` copy."""
    directory = _cached_fasrc_catalog_dir()
    return Path(directory) / Config.CATALOG_FILE if directory else None


def star_flags(row: Mapping[str, Any]) -> int:
    """Bit ``2**b`` set when the star has a valid cutout in band ``b`` (any size)."""
    flags = 0
    for key, value in row.items():
        if not str(key).startswith("valid:") or str(value).strip().lower() != "true":
            continue
        band = str(key).split(":")[1]
        flags |= STAR_FLAG_BITS.get(band, 0)
    return flags


def _stars() -> dict[str, Any]:
    path = stars_path()
    rows = []
    if path is not None:
        for row in _csv_rows(path):
            ra, dec = _float(row.get("ra")), _float(row.get("dec"))
            if ra is None or dec is None:
                continue
            rows.append([round(ra, 6), round(dec, 6), _round(_float(row.get("magnitude")), 3),
                         star_flags(row)])
    payload = _points(["ra", "dec", "mag", "flags"], rows, prefix="stars/")
    payload["flag_bits"] = dict(STAR_FLAG_BITS)
    payload["path"] = str(path) if path else None
    return payload


def _psf_clusters() -> dict[str, Any]:
    rows = []
    path = _cached_psf_clusters_json()
    source = "fasrc-cache"
    if path:
        with contextlib.suppress(OSError, ValueError, AttributeError):
            clusters = json.loads(Path(path).read_text(encoding="utf-8")).get("clusters", [])
            for index, cluster in enumerate(clusters):
                if not isinstance(cluster, Mapping):
                    continue
                ra, dec = _float(cluster.get("ra")), _float(cluster.get("dec"))
                if ra is not None and dec is not None:
                    fwhm = _float(cluster.get("fwhm_arcsec") or cluster.get("fwhm"))
                    rows.append([_round(ra), _round(dec), fwhm, index + 1, f"cluster-{index + 1:03d}"])
    if not rows:
        source = "local"
        local = Path(Config.EUCLID_PSF_DIR) / "euclid_psf_VIS.fits"
        with contextlib.suppress(OSError), fits.open(local, memmap=False) as hdul:
            for index, hdu in enumerate(hdul[1:]):
                ra, dec = _float(hdu.header.get("RA")), _float(hdu.header.get("DEC"))
                if ra is not None and dec is not None:
                    rows.append([_round(ra), _round(dec), _float(hdu.header.get("FWHM")),
                                 index + 1, f"cluster-{index + 1:03d}"])
    payload = _points(["ra", "dec", "fwhm_arcsec", "cluster", "id"], rows,
                      prefix="psf-clusters/", id_column="id")
    payload["catalog_source"] = source if rows else None
    return payload


def _psf_stamp() -> tuple:
    return (_mtime(_cached_psf_clusters_json()),
            _mtime(Path(Config.EUCLID_PSF_DIR) / "euclid_psf_VIS.fits"))


def _noise_positions() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "sky" / "observation" / "mer_noise_levels.json"
    rows = []
    with contextlib.suppress(OSError, ValueError):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            levels = list(row.get("levels_e") or [None] * 4)
            rows.append([_round(row.get("ra")), _round(row.get("dec")), *levels[:4],
                         str(row.get("tile")), q1_field_for(float(row["ra"]), float(row["dec"]))])
    return _points(["ra", "dec", "VIS", "Y_E", "J_E", "H_E", "tile", "field"], rows,
                   prefix="noise-positions/", id_column="tile")


def population_meta_path() -> Path:
    return Path(Config.DATA_DIR) / "population_comparison" / "euclid_population_meta.json"


def gaia_meta_path() -> Path:
    return Path(Config.DATA_DIR) / "population_comparison" / "gaia_population.meta.json"


def _population_cones() -> dict[str, Any]:
    features = []
    with contextlib.suppress(OSError, ValueError):
        meta = json.loads(population_meta_path().read_text(encoding="utf-8"))
        radius = float(meta.get("radius_arcmin") or 5.0) / 60.0
        for index, cone in enumerate(meta.get("cones") or []):
            features.append({
                "id": f"cone-{index:02d}", "ra": _round(cone.get("ra")),
                "dec": _round(cone.get("dec")), "radius_deg": radius,
                "props": {"rows": cone.get("rows"), "star_id": cone.get("star_id"),
                          "field": q1_field_for(float(cone["ra"]), float(cone["dec"]))},
                "inspect": {"kind": "source", "id": f"population-cones/cone-{index:02d}"},
            })
    return _circles(features)


def _gaia_fields() -> dict[str, Any]:
    features = []
    with contextlib.suppress(OSError, ValueError):
        meta = json.loads(gaia_meta_path().read_text(encoding="utf-8"))
        radius = float(meta.get("radius_arcmin") or 21.0) / 60.0
        for item in meta.get("fields") or []:
            name = str(item.get("name") or "")
            features.append({
                "id": name, "ra": _round(item.get("ra")), "dec": _round(item.get("dec")),
                "radius_deg": radius, "props": {"name": name, "rows": item.get("rows")},
                "inspect": {"kind": "source", "id": f"gaia-fields/{name}"},
            })
    return _circles(features)


def _jwst_mast() -> dict[str, Any]:
    payload = jwst_euclid.load_footprints()
    rows = []
    for obs_id, item in sorted(payload["footprints"].items()):
        if item.get("ra") is None or item.get("dec") is None:
            continue
        rows.append([_round(item["ra"]), _round(item["dec"]), obs_id,
                     item.get("instrument") or "", item.get("filters") or "",
                     item.get("target") or "", len(item.get("polygons") or []),
                     item.get("status") or ""])
    out = _points(["ra", "dec", "obs_id", "instrument", "filters", "target",
                   "polygons", "status"], rows, prefix="jwst-mast/", id_column="obs_id")
    out["polygons_url"] = "/api/sky/jwst/footprints"
    out["updated_utc"] = payload.get("updated_utc")
    return out


# ---------------------------------------------------------------------------
# the catalogue
# ---------------------------------------------------------------------------

_STATE_COLORS = {"current": "#2e9d57", "stale": "#d99a06", "missing": "#8a8f98"}

LAYERS: tuple[LayerSpec, ...] = (
    LayerSpec("q1-tiles", "Q1 MER tiles", "coverage", "polygons", _q1_tiles,
              lambda: (_mtime(q1_mer_tiles.TILES_PATH),),
              style={"color": "#4f7cff", "opacity": 0.35, "color_by": "vis_level_e",
                     "rejected_color": "#c0392b"},
              description="352 Q1 VIS MER footprints; colour = measured VIS sky level."),
    LayerSpec("q1-fields", "Q1 deep fields", "coverage", "circles", _q1_fields,
              lambda: (1,), style={"color": "#7b61ff", "opacity": 0.2},
              description="EDF-N / EDF-S / EDF-F query cones (6°)."),
    LayerSpec("nexus-footprint", "NEXUS F200W mosaic", "coverage", "polygons",
              _nexus_footprint, lambda: _dir_stamp(jwst_euclid.nexus_field_root()),
              style={"color": "#f39c12", "opacity": 0.2},
              fill_action={"method": "POST", "url": "/api/jwst-euclid/nexus/download-field",
                           "label": "Cache the NEXUS mosaic + Euclid tiles"}),
    LayerSpec("nexus-tiles", "NEXUS × Euclid tiles", "results", "polygons",
              lambda: _entry_polygons("nexus"),
              lambda: _results_stamp(jwst_euclid.nexus_field_root()),
              style={"color_by": "state", "colors": _STATE_COLORS, "opacity": 0.5},
              fill_action={"method": "POST", "url": "/api/experiments",
                           "label": "Run models on selected tiles"}),
    LayerSpec("real-tiles", "Cached 25.6″ tiles", "results", "polygons",
              lambda: _entry_polygons("tile"),
              lambda: _results_stamp(real_tiles.tiles_root()),
              style={"color_by": "state", "colors": _STATE_COLORS, "opacity": 0.5},
              fill_action={"method": "POST", "url": "/api/real/tiles",
                           "label": "Cache a 25.6″ tile here"}),
    LayerSpec("real-fields", "Legacy real fields", "results", "polygons",
              lambda: _entry_polygons("field"),
              lambda: _results_stamp(real_field_root()),
              style={"color_by": "state", "colors": _STATE_COLORS, "opacity": 0.35}),
    LayerSpec("poster", "Poster target", "results", "polygons",
              lambda: _entry_polygons("poster"),
              lambda: _results_stamp(real_tiles.poster_root()),
              style={"color": "#e84393", "opacity": 0.5}),
    LayerSpec("pairs", "JWST × Euclid pairs", "results", "polygons",
              lambda: _entry_polygons("pair"),
              lambda: _results_stamp(jwst_euclid.pair_root()),
              style={"color": "#00a8a8", "opacity": 0.5},
              fill_action={"method": "POST", "url": "/api/sky/jwst/pair",
                           "label": "Download a JWST × Euclid pair"}),
    LayerSpec("archive-fields", "Archive fields", "results", "polygons",
              lambda: _entry_polygons("archive"),
              lambda: _results_stamp(Path(Config.EUCLID_SKY_DIR) / "archive_fields"),
              style={"color": "#6c5ce7", "opacity": 0.45, "color_by": "field"},
              fill_action={"method": "POST", "url": "/api/archive-fields/sync",
                           "label": "Sync the archive fields from FASRC"}),
    LayerSpec("eval-objects", "Evaluation objects", "results", "points", _eval_objects,
              lambda: (_mtime(Path(Config.EVAL_RESULTS_DIR) / "manifest.csv"),),
              style={"color_by": "flux_ratio_sr_over_lr", "shape": "circle", "size": 6}),
    LayerSpec("experiments", "Experiment tiles", "results", "points", _experiment_points,
              lambda: _dir_stamp(experiments.records_root()),
              style={"color": "#ff7a45", "shape": "diamond", "size": 8}),
    LayerSpec("lens-candidates", "Q1 lens candidates", "catalogues", "points", _lenses,
              lambda: (_mtime(lens_catalog_path()),),
              style={"color_by": "grade",
                     "colors": {"A": "#d63031", "B": "#e17055", "C": "#fdcb6e"},
                     "shape": "circle", "size": 5},
              fill_action={"method": "POST", "url": "/api/evaluation/fetch-catalog",
                           "label": "Download the Q1 lens catalogue"}),
    LayerSpec("galaxies", "Evaluation galaxies", "catalogues", "points", _galaxies,
              lambda: (_mtime(galaxy_catalog_path()),),
              style={"color": "#00b894", "shape": "circle", "size": 4}),
    LayerSpec("stars", "Stars (FASRC catalogue)", "catalogues", "points", _stars,
              lambda: (_mtime(stars_path()),),
              style={"color_by": "mag", "shape": "square", "size": 2},
              fill_action={"method": "POST", "url": "/api/status/refresh-catalog",
                           "label": "Pull stars.csv from FASRC", "requires_fasrc": True}),
    LayerSpec("psf-clusters", "PSF clusters", "catalogues", "points", _psf_clusters,
              _psf_stamp, style={"color_by": "fwhm_arcsec", "shape": "cross", "size": 10},
              fill_action={"method": "POST", "url": "/api/euclid-psf/sync-meta",
                           "label": "Sync PSF cluster metadata", "requires_fasrc": True}),
    LayerSpec("noise-positions", "MER noise samples", "catalogues", "points",
              _noise_positions, lambda: (1,),
              style={"color_by": "VIS", "shape": "circle", "size": 5}),
    LayerSpec("population-cones", "Population cones", "catalogues", "circles",
              _population_cones, lambda: (_mtime(population_meta_path()),),
              style={"color": "#0984e3", "opacity": 0.25}),
    LayerSpec("gaia-fields", "Gaia fields", "catalogues", "circles", _gaia_fields,
              lambda: (_mtime(gaia_meta_path()),),
              style={"color": "#fab1a0", "opacity": 0.25}),
    LayerSpec("jwst-mast", "JWST MAST footprints", "catalogues", "points", _jwst_mast,
              lambda: (_mtime(jwst_euclid.footprints_path()),),
              style={"color": "#e17055", "shape": "plus", "size": 6},
              fill_action={"method": "POST", "url": "/api/sky/jwst/discover",
                           "label": "Discover JWST observations (MAST)"},
              description="Centres of discovered JWST imaging; polygons per view via "
                          "/api/sky/jwst/footprints."),
)
_BY_ID = {layer.id: layer for layer in LAYERS}


def real_field_root() -> Path:
    return Path(Config.EUCLID_INFERENCE_DIR) / "real_fields"


def _results_stamp(path: Path) -> tuple:
    return (_dir_stamp(path), _dir_stamp(model_catalog.outputs_root()),
            _mtime(model_catalog.regime_dir() / model_catalog.PRODUCTION_ARTIFACT_DIR
                   / "combiner.npz"))


_CACHE: dict[str, tuple[tuple, float, dict[str, Any]]] = {}
#: A memoised layer is rebuilt when its sources' mtimes change or after this
#: many seconds (nested writes do not always touch the stamped directories).
CACHE_TTL_S = 30.0


def layer_features(layer_id: str) -> dict[str, Any]:
    """One layer's features (memoised on its sources' mtimes); :class:`KeyError`."""
    layer = _BY_ID[layer_id]
    stamp = layer.stamp()
    cached = _CACHE.get(layer_id)
    if (cached is not None and cached[0] == stamp
            and time.monotonic() - cached[1] < CACHE_TTL_S):
        return cached[2]
    payload = {"id": layer.id, "label": layer.label, "group": layer.group, **layer.build()}
    payload["count"] = (len(payload.get("rows") or [])
                        if payload["kind"] == "points" else len(payload.get("features") or []))
    _CACHE[layer_id] = (stamp, time.monotonic(), payload)
    return payload


def invalidate() -> None:
    """Forget every memoised layer (tests, writers)."""
    _CACHE.clear()


def _coordinates(payload: Mapping[str, Any]) -> Iterable[tuple[float, float]]:
    if payload["kind"] == "points":
        for row in payload.get("rows") or []:
            if row[0] is not None and row[1] is not None:
                yield float(row[0]), float(row[1])
    elif payload["kind"] == "polygons":
        for feature in payload.get("features") or []:
            for ra, dec in feature.get("polygon") or []:
                yield float(ra), float(dec)
    else:
        for feature in payload.get("features") or []:
            if feature.get("ra") is not None:
                yield float(feature["ra"]), float(feature["dec"])


def _bbox(payload: Mapping[str, Any]) -> dict[str, float] | None:
    points = list(_coordinates(payload))
    if not points:
        return None
    ras, decs = [p[0] for p in points], [p[1] for p in points]
    return {"ra_min": min(ras), "ra_max": max(ras), "dec_min": min(decs), "dec_max": max(decs)}


def layers_payload() -> dict[str, Any]:
    """``GET /api/sky/layers``: the catalogue (no features)."""
    out = []
    for layer in LAYERS:
        try:
            payload = layer_features(layer.id)
            count, bbox, reason = payload["count"], _bbox(payload), None
        except Exception as exc:  # noqa: BLE001 - one broken dataset must not hide the rest
            count, bbox, reason = 0, None, f"{type(exc).__name__}: {exc}"
        out.append({
            "id": layer.id, "label": layer.label, "group": layer.group, "kind": layer.kind,
            "count": count, "bbox": bbox, "style": dict(layer.style),
            "ready": count > 0, "reason": reason or (None if count else "no local data yet"),
            "fill_action": dict(layer.fill_action) if layer.fill_action else None,
            "description": layer.description,
            "url": f"/api/sky/layer/{layer.id}",
        })
    return {"layers": out, "groups": ["coverage", "results", "catalogues"]}


def at(ra: float, dec: float) -> dict[str, Any]:
    """``GET /api/sky/at``: everything that covers a sky position."""
    ra = float(ra) % 360.0
    containing = q1_mer_tiles.tiles_containing(ra, dec)
    q1 = []
    for tile in containing:
        q1.append({**tile.to_dict(),
                   "margin_arcsec": round(q1_mer_tiles.polygon_margin_deg(
                       ra, dec, tile.polygon) * 3600.0, 2)})
    fingerprint = production_fingerprint()
    real = []
    for entry in real_tiles.entries_containing(ra, dec):
        real.append({"source": entry.source, "id": entry.id, "ref": entry.ref,
                     "label": entry.label, "has_jwst": entry.has_jwst,
                     "state": production_state(entry, fingerprint),
                     "inspect": {"kind": "realtile", "id": entry.ref}})
    jwst = []
    for obs_id, item in jwst_euclid.load_footprints()["footprints"].items():
        if any(q1_mer_tiles.point_in_polygon(ra, dec, polygon)
               for polygon in item.get("polygons") or []):
            jwst.append({key: item.get(key) for key in (
                "obs_id", "instrument", "filters", "target", "exptime_s", "status")}
                | {"obs_id": obs_id})
    observed = any(tile.rejected is None for tile in containing)
    return {
        "ra": ra, "dec": dec, "field": q1_field_for(ra, dec),
        "in_q1": bool(containing),
        "q1_observed": observed,
        "q1_verdict": ("observed" if observed else "unobserved") if containing else "outside",
        "q1_tiles": q1,
        "best_tile": q1[0]["tile"] if q1 else None,
        "real_tiles": real,
        "nexus": [item for item in real if item["source"] == "nexus"],
        "pairs": [item for item in real if item["source"] == "pair"],
        "jwst": jwst,
        "jwst_discovered": jwst_euclid.footprints_path().is_file(),
    }


__all__ = [
    "LAYERS",
    "convex_hull",
    "invalidate",
    "STAR_FLAG_BITS",
    "at",
    "layer_features",
    "layers_payload",
    "production_fingerprint",
    "production_state",
    "star_flags",
    "stars_path",
]
