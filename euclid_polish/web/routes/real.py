"""Real results, model catalogue and experiments (contract C9, spec §9.1–9.2).

Everything here is local (or talks to the public Euclid archive from a local
job) and works with FASRC disconnected — nothing is gated. Errors under
``/api/real/``, ``/api/models`` and ``/api/experiments`` are JSON
(:func:`errors.json_errors_for`).
"""

from __future__ import annotations

import io
import math

import numpy as np
from astropy.io import fits
from flask import Response, jsonify, request

from euclid_polish.web import errors
from euclid_polish.web.helpers import (
    experiments,
    jwst_euclid,
    model_catalog,
    real_tiles,
    sky_atlas,
)
from euclid_polish.web.jobs import REGISTRY


def _fail(message: str, status: int = 400, **extra):
    return jsonify({"ok": False, "error": message, **extra}), status


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _float_arg(source, name: str) -> float:
    try:
        value = float(source.get(name, ""))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number (degrees)") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


_OUTPUT_KEYS = ("label", "kind", "fingerprint", "created", "experiment_id", "file",
                "member_labels", "combiner_kind", "lr_sha", "shape", "legacy", "origin")


def _model_rows(entry: real_tiles.TileEntry, current: dict[str, str | None], *,
                full: bool) -> tuple[dict, dict]:
    """``(models, outputs)`` of a tile: one row per spec with an output (the
    C9 store or a legacy SR read in place, :func:`real_tiles.tile_outputs`)."""
    outputs = real_tiles.tile_outputs(entry, current)
    rows = {}
    for spec, meta in outputs.items():
        metrics = meta.get("metrics") or {}
        row = {"state": model_catalog.output_state(meta, current),
               "legacy": bool(meta.get("legacy"))}
        if full:
            row.update({key: meta.get(key) for key in _OUTPUT_KEYS if key != "legacy"})
            row["metrics"] = {key: metrics.get(key) for key in (
                "per_band", "summary", "gate_core_weights") if key in metrics}
            row["image_url"] = (f"/api/real/{entry.source}/{entry.id}/image.fits"
                                f"?tier=m:{spec}&band=VIS")
        else:
            row.update({key: meta.get(key) for key in (
                "label", "fingerprint", "created", "experiment_id", "file", "origin")})
            row["summary"] = metrics.get("summary")
        rows[spec] = row
    return rows, outputs


def _card(entry: real_tiles.TileEntry) -> dict:
    specs = model_catalog.list_specs()
    current = model_catalog.current_fingerprints(specs)
    models, outputs = _model_rows(entry, current, full=True)
    q1_tile = (jwst_euclid.choose_q1_tile(entry.ra, entry.dec, 0.0)
               if entry.ra is not None and entry.dec is not None else None)
    tiers = ["lr"] + (["jwst"] if entry.has_jwst else []) + [f"m:{spec}" for spec in models]
    return {
        **entry.to_dict(),
        "models": models,
        "production_state": sky_atlas.production_state(
            entry, current.get(model_catalog.SPEC_PRODUCTION), outputs),
        "legacy": entry.extras.get("legacy_sr"),
        "runnable_models": [item.spec for item in specs if item.available],
        "experiments": experiments.experiments_for_tile(entry.source, entry.id),
        "disk": real_tiles.disk_usage(entry),
        "q1_tile": q1_tile.to_dict() if q1_tile is not None else None,
        "image_urls": {tier: (f"/api/real/{entry.source}/{entry.id}/image.fits"
                              f"?tier={tier}&band=VIS") for tier in tiers},
        "viewer": {"collection": "real", "params": {"source": entry.source}, "id": entry.id},
    }


def _image(entry: real_tiles.TileEntry, tier: str, band: str) -> tuple[np.ndarray, fits.Header, str]:
    """``(2-D plane, WCS header, unit)`` of one tier/band of a real tile."""
    if tier == "lr":
        tile = real_tiles.get_tile(entry.source, entry.id, entry=entry)
        bands = [b.upper() for b in entry.bands]
        if band.upper() not in bands:
            raise ValueError(f"band must be one of {', '.join(entry.bands)}")
        header = tile.wcs_header or fits.Header()
        return tile.lr_e[..., bands.index(band.upper())], header, "electron"
    if tier == "jwst":
        planes = real_tiles.jwst_planes(entry)
        if not planes:
            raise real_tiles.RealTileError(404, f"{entry.ref} has no JWST image")
        plane = next((p for p in planes if p["band"].upper() == band.upper()), None)
        if plane is None and band.upper() in ("", "VIS", "JWST"):
            plane = planes[0]
        if plane is None:
            raise ValueError(f"JWST band must be one of {', '.join(p['band'] for p in planes)}")
        header = real_tiles.wcs_only_header(plane["header"]) or fits.Header()
        return np.asarray(plane["data"], np.float32), header, "MJy/sr"
    if tier.startswith("m:"):
        spec = model_catalog.canonical_spec(tier[2:])
        current = model_catalog.current_fingerprints().get(spec)
        try:
            cube, header, _meta = real_tiles.load_output(entry, spec, current=current)
        except FileNotFoundError:
            raise real_tiles.RealTileError(
                404, f"{spec} has not been run on {entry.ref} yet") from None
        bands = [b.upper() for b in real_tiles.BAND_NAMES]
        if band.upper() not in bands:
            raise ValueError(f"band must be one of {', '.join(real_tiles.BAND_NAMES)}")
        return cube[..., bands.index(band.upper())], real_tiles.wcs_only_header(header) \
            or fits.Header(), "electron"
    raise ValueError("tier must be lr, jwst or m:<spec>")


def _q1_refusal(ra: float, dec: float, force: bool):
    """The 400 answer when a 25.6″ tile cannot be cached at ``(ra, dec)``
    (``None`` when it can): outside every committed Q1 MER polygon, or —
    unless ``force`` — inside only tiles the noise campaign measured as
    unobserved (``rejected``)."""
    tile = jwst_euclid.choose_q1_tile(ra, dec, real_tiles.TILE_SIZE_ARCSEC)
    if tile is None:
        return _fail(f"({ra:.5f}, {dec:+.5f}) is outside the Euclid Q1 MER footprints",
                     code="outside_q1")
    if tile.rejected and not force:
        return _fail(f"every Q1 MER tile containing ({ra:.5f}, {dec:+.5f}) was measured "
                     f"unobserved (tile {tile.tile}: {tile.rejected}); pass force=1 to try "
                     "anyway", code="unobserved_q1", tile=tile.tile)
    return None


def register(app):
    errors.json_errors_for(app, "/api/real/", "/api/models", "/api/experiments")

    @app.errorhandler(real_tiles.RealTileError)
    def _real_tile_error(exc: real_tiles.RealTileError):
        return jsonify({"ok": False, "error": str(exc)}), exc.code

    @app.get("/api/real/sources")
    def api_real_sources():
        return jsonify(real_tiles.sources_payload())

    @app.get("/api/real/<source>")
    def api_real_list(source: str):
        entries = real_tiles.list_entries(source)
        current = model_catalog.current_fingerprints()
        tiles = []
        for entry in entries:
            row = entry.to_dict()
            row["models"], outputs = _model_rows(entry, current, full=False)
            row["production_state"] = sky_atlas.production_state(
                entry, current.get(model_catalog.SPEC_PRODUCTION), outputs)
            tiles.append(row)
        return jsonify({"source": source, **real_tiles.SOURCE_INFO[source],
                        "count": len(tiles), "tiles": tiles})

    @app.get("/api/real/<source>/<identifier>")
    def api_real_card(source: str, identifier: str):
        return jsonify(_card(real_tiles.get_entry(source, identifier)))

    @app.get("/api/real/<source>/<identifier>/image.fits")
    def api_real_image(source: str, identifier: str):
        entry = real_tiles.get_entry(source, identifier)
        tier = (request.args.get("tier") or "lr").strip()
        band = (request.args.get("band") or "VIS").strip()
        try:
            plane, header, unit = _image(entry, tier, band)
        except ValueError as exc:
            return _fail(str(exc))
        out = fits.Header()
        out.update(header)
        out["BUNIT"] = unit
        out["TIER"] = tier[:68]
        out["BAND"] = band[:68]
        out["REALTILE"] = entry.ref[:68]
        buffer = io.BytesIO()
        fits.PrimaryHDU(np.asarray(plane, np.float32), header=out).writeto(
            buffer, output_verify="silentfix")
        slug = tier.replace(":", "-")
        response = Response(buffer.getvalue(), mimetype="application/fits")
        response.headers["Content-Disposition"] = (
            f'inline; filename="{entry.source}-{entry.id}-{slug}-{band}.fits"')
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.post("/api/real/tiles")
    def api_real_cache_tile():
        try:
            ra = _float_arg(request.form, "ra")
            dec = _float_arg(request.form, "dec")
            specs = model_catalog.parse_specs(request.form.get("run", ""))
        except ValueError as exc:
            return _fail(str(exc))
        if not (0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0):
            return _fail("ra must be in [0, 360) and dec in [-90, 90]")
        refusal = _q1_refusal(ra, dec, _truthy(request.form.get("force")))
        if refusal is not None:
            return refusal
        identifier = real_tiles.real_tile_id(ra, dec)
        experiment_id = experiments.new_experiment_id() if specs else None

        def target(cap):
            manifest = real_tiles.cache_tile(ra, dec, progress=cap.tick)
            result = {"tile": manifest["id"], "ref": f"tile/{manifest['id']}"}
            if specs:
                record = experiments.run_experiment(
                    [("tile", manifest["id"])], specs, experiment_id=experiment_id,
                    label=f"cached tile {manifest['id']}", job_id=cap.job.job_id,
                    progress=cap.tick, check_cancelled=cap.check_cancelled)
                result["experiment"] = experiments.job_result(record)
            return result

        job_id = REGISTRY.spawn(
            f"cache real tile ({ra:.5f}, {dec:+.5f})" + (f" + {', '.join(specs)}" if specs else ""),
            target, kind="real-tile")
        return jsonify({"ok": True, "job_id": job_id, "id": identifier,
                        "ref": f"tile/{identifier}", "experiment_id": experiment_id})

    @app.post("/api/real/<source>/<identifier>/delete-outputs")
    def api_real_delete_outputs(source: str, identifier: str):
        entry = real_tiles.get_entry(source, identifier)
        result = experiments.delete_tile_outputs(entry.source, entry.id)
        sky_atlas.invalidate()
        return jsonify({"ok": True, "ref": entry.ref, **result})

    @app.get("/api/models")
    def api_models():
        return jsonify(model_catalog.catalog_payload())

    @app.post("/api/experiments")
    def api_experiments_start():
        try:
            refs = real_tiles.parse_refs(request.form.get("tiles", ""))
            specs = model_catalog.parse_specs(request.form.get("models", ""))
            entries, runnable, skipped = experiments.plan(refs, specs)
        except experiments.DiskSpaceError as exc:
            return _fail(str(exc), 507, code="insufficient_storage",
                         needed_bytes=exc.needed, free_bytes=exc.free)
        except ValueError as exc:
            return _fail(str(exc))
        label = (request.form.get("label") or "").strip()[:200]
        experiment_id = experiments.new_experiment_id()

        def target(cap):
            record = experiments.run_experiment(
                refs, specs, experiment_id=experiment_id, label=label,
                job_id=cap.job.job_id, progress=cap.tick,
                check_cancelled=cap.check_cancelled)
            sky_atlas.invalidate()
            return experiments.job_result(record)

        job_id = REGISTRY.spawn(
            f"experiment · {len(entries)} tile(s) × {len(runnable)} model(s)", target,
            kind="real-experiment")
        return jsonify({"ok": True, "job_id": job_id, "experiment_id": experiment_id,
                        "tiles": [entry.ref for entry in entries],
                        "models": [item.spec for item in runnable], "skipped": skipped})

    @app.get("/api/experiments")
    def api_experiments_list():
        return jsonify({"experiments": experiments.list_experiments()})

    @app.get("/api/experiments/<experiment_id>")
    def api_experiment_get(experiment_id: str):
        try:
            return jsonify(experiments.get_experiment(experiment_id))
        except KeyError:
            return _fail(f"unknown experiment {experiment_id!r}", 404)
