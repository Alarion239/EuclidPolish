"""JWST × Euclid paired-field archive and viewer routes."""

from __future__ import annotations

import json
import re
from pathlib import Path

from flask import jsonify, request, send_file

from euclid_polish.web.helpers.jwst_euclid import (
    _cached_pair_is_usable,
    download_and_align_pair,
    download_nexus_field,
    download_nexus_pair,
    download_remaining_locations,
    enrich_manifest_metadata,
    find_location_group,
    location_groups,
    nexus_fields,
    nexus_pair_id,
    nexus_product_options,
    overlap_rows,
    pair_root,
    run_starfull_nexus_field_inference,
    run_starfull_pair_inference,
    saved_pairs,
    scan_euclid_coverage,
)
from euclid_polish.web.jobs import REGISTRY

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,220}$")
_SIZES = {"euclid_png": "euclid_vis.png", "jwst_png": "jwst_native.png"}


def _manifest(identifier: str) -> dict | None:
    if not _SAFE_ID.fullmatch(identifier):
        return None
    path = pair_root() / identifier / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    directory = pair_root() / identifier
    if not _cached_pair_is_usable(directory, payload):
        return None
    return enrich_manifest_metadata(directory, payload)


def _asset_path(identifier: str, filename: str) -> Path | None:
    """Resolve a cached asset below exactly one paired-field directory."""
    root = pair_root().resolve()
    field_dir = (root / identifier).resolve()
    path = (field_dir / filename).resolve()
    if field_dir.parent != root:
        return None
    try:
        path.relative_to(field_dir)
    except ValueError:
        return None
    return path


def register(app):
    @app.get("/api/jwst-euclid/fields")
    def api_jwst_euclid_fields():
        rows, status = location_groups()
        return jsonify({"fields": rows, "status": status})

    @app.get("/api/jwst-euclid/nexus/options")
    def api_nexus_options():
        return jsonify({"products": nexus_product_options()})

    @app.get("/api/jwst-euclid/nexus/fields")
    def api_nexus_fields():
        fields = [{
            "field_id": field.get("field_id"), "filter": field.get("filter"),
            "target_name": field.get("target_name"), "count": field.get("count"),
            "tile_size_euclid_pixels": field.get("tile_size_euclid_pixels"),
            "tile_size_jwst_pixels": field.get("tile_size_jwst_pixels"),
            "four_band_count": field.get("four_band_count", 0),
            "sr_count": field.get("sr_count", 0),
            "current_sr_count": field.get("current_sr_count", 0),
            "stale_sr_count": field.get("stale_sr_count", 0),
            "active_combiner_kind": field.get("active_combiner_kind"),
        } for field in nexus_fields()]
        return jsonify({"fields": fields})

    @app.get("/api/jwst-euclid/saved")
    def api_jwst_euclid_saved():
        source = request.args.get("source") or None
        fields = [{
            "field_id": field.get("field_id"),
            "source": field.get("source"),
            "target_name": field.get("target_name"),
            "jwst_filters": field.get("jwst_filters"),
            "ra_deg": field.get("ra_deg"),
            "dec_deg": field.get("dec_deg"),
            "size_arcsec": field.get("size_arcsec"),
        } for field in saved_pairs(source=source)]
        return jsonify({"fields": fields})

    @app.get("/api/jwst-euclid/field.json")
    def api_jwst_euclid_field():
        identifier = request.args.get("id", "")
        payload = _manifest(identifier)
        if payload is None:
            return jsonify({"error": "paired field not found"}), 404
        return jsonify(payload)

    @app.post("/api/jwst-euclid/download")
    def api_jwst_euclid_download():
        identifier = request.form.get("field_id", "").strip()
        try:
            size_arcsec = float(request.form.get("size_arcsec", "30"))
        except ValueError:
            return jsonify({"error": "size_arcsec must be a number"}), 400
        if not _SAFE_ID.fullmatch(identifier):
            return jsonify({"error": "select a sky location from the field list"}), 400
        if not 1.0 <= size_arcsec <= 120.0:
            return jsonify({"error": "size_arcsec must be between 1 and 120 arcsec"}), 400

        row = find_location_group(identifier)
        if row is None:
            return jsonify({
                "error": "select a sky location from the cached field list before downloading",
            }), 400
        job_id = REGISTRY.spawn(
            label=f"download JWST bands + Euclid ({row.get('jwst_target_name') or identifier})",
            target=lambda cap: download_and_align_pair(
                row,
                size_arcsec=size_arcsec,
                progress=lambda done, total, label: cap.tick(done, total, label),
            ),
        )
        return jsonify({"job_id": job_id, "field_id": identifier})

    @app.post("/api/jwst-euclid/nexus/download")
    def api_nexus_download():
        try:
            ra = float(request.form.get("ra", ""))
            dec = float(request.form.get("dec", ""))
            size_arcsec = float(request.form.get("size_arcsec", "30"))
        except ValueError:
            return jsonify({"error": "NEXUS RA, Dec, and cutout size must be numbers"}), 400
        filter_name = request.form.get("filter", "F200W").strip().upper()
        if filter_name not in {str(product["filter"]) for product in nexus_product_options()}:
            return jsonify({"error": "NEXUS filter must be F200W or F444W"}), 400
        if not (0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0 and 1.0 <= size_arcsec <= 120.0):
            return jsonify({"error": "NEXUS coordinates or cutout size are outside the supported range"}), 400
        identifier = nexus_pair_id(ra, dec, filter_name, size_arcsec)
        job_id = REGISTRY.spawn(
            label=f"download NEXUS {filter_name} + matching Euclid VIS",
            target=lambda cap: download_nexus_pair(
                ra=ra, dec=dec, filter_name=filter_name, size_arcsec=size_arcsec,
                progress=lambda done, total, label: cap.tick(done, total, label),
            ),
        )
        return jsonify({"job_id": job_id, "field_id": identifier})

    @app.post("/api/jwst-euclid/nexus/download-field")
    def api_nexus_download_field():
        filter_name = request.form.get("filter", "F200W").strip().upper()
        if filter_name not in {str(product["filter"]) for product in nexus_product_options()}:
            return jsonify({"error": "NEXUS filter must be F200W or F444W"}), 400
        job_id = REGISTRY.spawn(
            label=f"cache NEXUS {filter_name} mosaic + four-band Euclid coverage",
            target=lambda cap: download_nexus_field(
                filter_name=filter_name,
                progress=lambda done, total, label: cap.tick(done, total, label),
            ),
        )
        return jsonify({"job_id": job_id})

    @app.post("/api/jwst-euclid/nexus/infer")
    def api_nexus_infer():
        identifier = request.form.get("field_id", "").strip()
        if not _SAFE_ID.fullmatch(identifier) or not any(
            field.get("field_id") == identifier for field in nexus_fields()
        ):
            return jsonify({"error": "save a valid NEXUS Euclid field before inference"}), 404
        job_id = REGISTRY.spawn(
            label=f"run STARFULL combiner on NEXUS tiles ({identifier})",
            target=lambda cap: run_starfull_nexus_field_inference(
                identifier,
                progress=lambda done, total, label: cap.tick(done, total, label),
            ),
        )
        return jsonify({"job_id": job_id, "field_id": identifier})

    @app.post("/api/jwst-euclid/download-all")
    def api_jwst_euclid_download_all():
        try:
            size_arcsec = float(request.form.get("size_arcsec", "30"))
        except ValueError:
            return jsonify({"error": "size_arcsec must be a number"}), 400
        if not 1.0 <= size_arcsec <= 120.0:
            return jsonify({"error": "size_arcsec must be between 1 and 120 arcsec"}), 400
        rows, _ = location_groups()
        remaining = sum(
            not row.get("available") and row.get("euclid_coverage_status") != "not_covered"
            for row in rows
        )
        job_id = REGISTRY.spawn(
            label=f"download remaining JWST bands + Euclid ({remaining} locations)",
            target=lambda cap: download_remaining_locations(
                size_arcsec=size_arcsec,
                progress=lambda done, total, label: cap.tick(done, total, label),
            ),
        )
        return jsonify({"job_id": job_id, "remaining_count": remaining})

    @app.post("/api/jwst-euclid/scan-coverage")
    def api_jwst_euclid_scan_coverage():
        rows, status = overlap_rows()
        if not rows:
            return jsonify({"error": "no cached JWST fields are available to scan"}), 400
        unique_count = status.get("coverage_scan", {}).get("unique_count", len(rows))
        job_id = REGISTRY.spawn(
            label=f"scan Euclid VIS coverage ({len(rows)} JWST rows; {unique_count} unique centers)",
            target=lambda cap: scan_euclid_coverage(
                progress=lambda done, total, label: cap.tick(done, total, label),
            ),
        )
        return jsonify({"job_id": job_id})

    @app.post("/api/jwst-euclid/infer")
    def api_jwst_euclid_infer():
        identifier = request.form.get("field_id", "").strip()
        if _manifest(identifier) is None:
            return jsonify({"error": "save a valid JWST × Euclid field before inference"}), 404
        job_id = REGISTRY.spawn(
            label=f"run STARFULL combiner ({identifier})",
            target=lambda cap: run_starfull_pair_inference(
                identifier,
                progress=lambda done, total, label: cap.tick(done, total, label),
            ),
        )
        return jsonify({"job_id": job_id, "field_id": identifier})

    @app.get("/api/jwst-euclid/field/<identifier>/<kind>")
    def api_jwst_euclid_image(identifier: str, kind: str):
        filename = _SIZES.get(kind)
        payload = _manifest(identifier)
        if filename is None or payload is None:
            return jsonify({"error": "paired field asset not found"}), 404
        path = _asset_path(identifier, filename)
        if path is None or not path.is_file():
            return jsonify({"error": "paired field asset not found"}), 404
        return send_file(path, mimetype="image/png", max_age=0)

    @app.get("/api/jwst-euclid/field/<identifier>/download/<kind>")
    def api_jwst_euclid_download_asset(identifier: str, kind: str):
        payload = _manifest(identifier)
        filename = payload.get("files", {}).get(kind) if payload else None
        if filename is None and payload and kind in {"lr", "starfull"}:
            filename = (payload.get("inference", {}).get("files", {}) or {}).get(kind)
        if payload is None or not isinstance(filename, str) or not _SAFE_ID.fullmatch(identifier):
            return jsonify({"error": "paired field asset not found"}), 404
        path = _asset_path(identifier, filename)
        if path is None or not path.is_file():
            return jsonify({"error": "paired field asset not found"}), 404
        return send_file(path, as_attachment=True, download_name=path.name, max_age=0)
