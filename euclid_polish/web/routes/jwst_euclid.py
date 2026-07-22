"""JWST × Euclid paired-field archive and viewer routes."""

from __future__ import annotations

import json
import re
from pathlib import Path

from flask import jsonify, request, send_file

from euclid_polish.web.helpers.jwst_euclid import (
    _cached_pair_is_usable,
    download_and_align_pair,
    field_id,
    find_overlap_row,
    overlap_rows,
    pair_root,
    scan_euclid_coverage,
)
from euclid_polish.web.jobs import REGISTRY

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,220}$")
_SIZES = {"euclid_png": "euclid_vis.png", "jwst_png": "jwst_aligned.png"}


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
    return payload if _cached_pair_is_usable(pair_root() / identifier, payload) else None


def _asset_path(identifier: str, filename: str) -> Path | None:
    """Resolve a cached asset below exactly one paired-field directory."""
    root = pair_root().resolve()
    field_dir = (root / identifier).resolve()
    path = (field_dir / filename).resolve()
    if field_dir.parent != root or path.parent != field_dir:
        return None
    return path


def register(app):
    @app.get("/api/jwst-euclid/fields")
    def api_jwst_euclid_fields():
        rows, status = overlap_rows()
        return jsonify({"fields": rows, "status": status})

    @app.get("/api/jwst-euclid/field.json")
    def api_jwst_euclid_field():
        identifier = request.args.get("id", "")
        payload = _manifest(identifier)
        if payload is None:
            return jsonify({"error": "paired field not found"}), 404
        return jsonify(payload)

    @app.post("/api/jwst-euclid/download")
    def api_jwst_euclid_download():
        archive = request.form.get("jwst_archive", "esa").strip().lower()
        tile_index = request.form.get("euclid_tile_index", "").strip()
        observation_id = request.form.get("jwst_observation_id", "").strip()
        try:
            size_arcsec = float(request.form.get("size_arcsec", "30"))
        except ValueError:
            return jsonify({"error": "size_arcsec must be a number"}), 400
        if archive not in {"esa", "mast"}:
            return jsonify({"error": "jwst_archive must be esa or mast"}), 400
        if not tile_index or len(tile_index) > 120 or not observation_id or len(observation_id) > 180:
            return jsonify({"error": "a Euclid tile and JWST observation id are required"}), 400
        if not 1.0 <= size_arcsec <= 120.0:
            return jsonify({"error": "size_arcsec must be between 1 and 120 arcsec"}), 400

        row = find_overlap_row(archive, tile_index, observation_id)
        if row is None:
            return jsonify({
                "error": "select a field from the cached overlap table before downloading",
            }), 400
        identifier = field_id(archive, tile_index, observation_id, size_arcsec)
        job_id = REGISTRY.spawn(
            label=f"download + align JWST × Euclid ({tile_index} / {observation_id})",
            target=lambda cap: download_and_align_pair(
                row,
                size_arcsec=size_arcsec,
                progress=lambda done, total, label: cap.tick(done, total, label),
            ),
        )
        return jsonify({"job_id": job_id, "field_id": identifier})

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
        if payload is None or not isinstance(filename, str) or not _SAFE_ID.fullmatch(identifier):
            return jsonify({"error": "paired field asset not found"}), 404
        path = _asset_path(identifier, filename)
        if path is None or not path.is_file():
            return jsonify({"error": "paired field asset not found"}), 404
        return send_file(path, as_attachment=True, download_name=filename, max_age=0)
