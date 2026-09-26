"""Sky atlas backend (contract C9, spec §9.3): layers, point lookup, JWST
discovery / footprints / pairing.

Layers and lookups read local files only; discovery and pairing talk to the
public MAST / Euclid archives from local jobs — none of it needs FASRC.
Errors under ``/api/sky/`` are JSON (:func:`errors.json_errors_for`).
"""

from __future__ import annotations

import math

from flask import jsonify, request

from euclid_polish.sky.observation import q1_mer_tiles
from euclid_polish.sky.observation.q1_fields import angular_separation_deg
from euclid_polish.web import errors
from euclid_polish.web.helpers import (
    experiments,
    jwst_euclid,
    model_catalog,
    real_tiles,
    sky_atlas,
)
from euclid_polish.web.jobs import REGISTRY

MAX_FOOTPRINT_RADIUS_DEG = 5.0
NEXUS_FILTERS = ("F200W", "F444W")


def _fail(message: str, status: int = 400, **extra):
    return jsonify({"ok": False, "error": message, **extra}), status


def _coordinate(source, name: str, *, required: bool = True, default: float | None = None) -> float:
    raw = source.get(name)
    if raw in (None, ""):
        if required:
            raise ValueError(f"{name} is required (degrees)")
        return float(default if default is not None else 0.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number (degrees)") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _check_position(ra: float, dec: float) -> None:
    if not (0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0):
        raise ValueError("ra must be in [0, 360) and dec in [-90, 90]")


def _group_for_obs(obs_id: str) -> dict | None:
    groups, _status = jwst_euclid.location_groups()
    for group in groups:
        products = group.get("jwst_products") or [group]
        if any(str(product.get("jwst_observation_id")) == obs_id for product in products):
            return group
    return None


def _group_near(ra: float, dec: float, radius_arcsec: float) -> dict | None:
    groups, _status = jwst_euclid.location_groups()
    best = None
    for group in groups:
        gra, gdec = jwst_euclid.field_coordinates(group)
        if gra is None or gdec is None:
            continue
        separation = angular_separation_deg(ra, dec, gra, gdec) * 3600.0
        if separation <= radius_arcsec and (best is None or separation < best[0]):
            best = (separation, group)
    return best[1] if best else None


def _in_nexus(ra: float, dec: float) -> bool:
    """Whether a point lies in one of the NEXUS × Euclid tile cells (the
    cells' own polygons — their convex hull is ~12 % larger than the real
    coverage and would send gap points to a blank NEXUS cutout)."""
    return bool(real_tiles.entries_containing(ra, dec, sources=("nexus",)))


def register(app):
    errors.json_errors_for(app, "/api/sky/")

    @app.get("/api/sky/layers")
    def api_sky_layers():
        return jsonify(sky_atlas.layers_payload())

    @app.get("/api/sky/layer/<layer_id>")
    def api_sky_layer(layer_id: str):
        try:
            return jsonify(sky_atlas.layer_features(layer_id))
        except KeyError:
            return _fail(f"unknown sky layer {layer_id!r}", 404)

    @app.get("/api/sky/at")
    def api_sky_at():
        try:
            ra = _coordinate(request.args, "ra")
            dec = _coordinate(request.args, "dec")
            _check_position(ra, dec)
        except ValueError as exc:
            return _fail(str(exc))
        return jsonify(sky_atlas.at(ra, dec))

    @app.post("/api/sky/jwst/discover")
    def api_sky_jwst_discover():
        fields = [f.strip() for f in (request.form.get("fields") or "").split(",") if f.strip()]
        region = None
        raw_region = (request.form.get("region") or "").strip()
        try:
            if raw_region:
                parts = [float(v) for v in raw_region.split(",")]
                if len(parts) != 3 or not all(math.isfinite(v) for v in parts):
                    raise ValueError("region must be 'ra,dec,radius_deg'")
                _check_position(parts[0], parts[1])
                if not 0 < parts[2] <= 10:
                    raise ValueError("region radius must be in (0, 10] degrees")
                region = (parts[0], parts[1], parts[2])
            known = {str(t.region).upper() for t in q1_mer_tiles.load_tiles() if t.region}
            unknown = [f for f in fields if f.upper() not in known]
            if unknown:
                raise ValueError(f"unknown Q1 field(s): {', '.join(unknown)} "
                                 f"(use {', '.join(sorted(known))})")
            tiles = jwst_euclid.discovery_tiles(fields=fields, region=region)
            if not tiles:
                raise ValueError("no Q1 MER tile lies in the requested discovery scope")
        except ValueError as exc:
            return _fail(str(exc))
        refresh = (request.form.get("refresh") or "").lower() in {"1", "true", "yes"}
        scope = ", ".join(fields) or (f"cone {region}" if region else "all of Q1")

        def target(cap):
            result = jwst_euclid.discover_jwst_overlap(
                fields=fields, region=region, refresh=refresh, progress=cap.tick)
            sky_atlas.invalidate()
            return result

        job_id = REGISTRY.spawn(f"discover JWST × Euclid overlap ({scope})", target,
                                kind="jwst-discover")
        return jsonify({"ok": True, "job_id": job_id, "tile_count": len(tiles),
                        "fields": fields, "region": list(region) if region else None})

    @app.get("/api/sky/jwst/footprints")
    def api_sky_jwst_footprints():
        try:
            ra = _coordinate(request.args, "ra")
            dec = _coordinate(request.args, "dec")
            radius = _coordinate(request.args, "r", required=False, default=0.5)
            _check_position(ra, dec)
            if not 0 < radius <= MAX_FOOTPRINT_RADIUS_DEG:
                raise ValueError(f"r must be in (0, {MAX_FOOTPRINT_RADIUS_DEG:g}] degrees")
        except ValueError as exc:
            return _fail(str(exc))
        return jsonify({"ra": ra, "dec": dec, "r": radius,
                        **jwst_euclid.footprints_in_cone(ra, dec, radius)})

    @app.post("/api/sky/jwst/pair")
    def api_sky_jwst_pair():
        form = request.form
        obs_id = (form.get("obs_id") or "").strip()
        try:
            size = _coordinate(form, "size_arcsec", required=False, default=30.0)
            if not 1.0 <= size <= 120.0:
                raise ValueError("size_arcsec must be between 1 and 120")
            specs = model_catalog.parse_specs(form.get("run", ""))
            filter_name = (form.get("filter") or "F200W").strip().upper()
            if filter_name not in NEXUS_FILTERS:
                raise ValueError(f"filter must be one of {', '.join(NEXUS_FILTERS)}")
            if obs_id:
                group = _group_for_obs(obs_id)
                if group is None:
                    return _fail(f"JWST observation {obs_id!r} is not in the discovered "
                                 "overlap; run POST /api/sky/jwst/discover first", 404,
                                 code="not_discovered")
                mode, ra, dec = "archive", None, None
            else:
                ra = _coordinate(form, "ra")
                dec = _coordinate(form, "dec")
                _check_position(ra, dec)
                if _in_nexus(ra, dec):
                    mode, group = "nexus", None
                else:
                    group = _group_near(ra, dec, max(size / 2.0, 15.0))
                    if group is None:
                        return _fail("no discovered JWST observation covers this point "
                                     "(NEXUS or the discovery cache); run discovery first",
                                     404, code="not_discovered")
                    mode = "archive"
        except ValueError as exc:
            return _fail(str(exc))
        if mode == "nexus":
            assert ra is not None and dec is not None
            pair_id = jwst_euclid.nexus_pair_id(ra, dec, filter_name, size)
        else:
            assert group is not None
            pair_id = str(group.get("field_id"))

        def target(cap):
            if mode == "nexus":
                manifest = jwst_euclid.download_nexus_pair(
                    ra=ra, dec=dec, filter_name=filter_name, size_arcsec=size,
                    progress=cap.tick)
            else:
                manifest = jwst_euclid.download_and_align_pair(
                    group, size_arcsec=size, progress=cap.tick)
            identifier = str(manifest.get("field_id") or pair_id)
            jwst_euclid.pair_lr_input(identifier, progress=cap.tick)
            real_tiles.invalidate("pair")
            sky_atlas.invalidate()
            result = {"pair_id": identifier, "ref": f"pair/{identifier}"}
            if specs:
                record = experiments.run_experiment(
                    [("pair", identifier)], specs, label=f"pair {identifier}",
                    job_id=cap.job.job_id, progress=cap.tick,
                    check_cancelled=cap.check_cancelled)
                result["experiment"] = experiments.job_result(record)
            return result

        job_id = REGISTRY.spawn(f"JWST × Euclid pair ({pair_id})", target, kind="jwst-pair")
        return jsonify({"ok": True, "job_id": job_id, "pair_id": pair_id,
                        "ref": f"pair/{pair_id}", "mode": mode})
