"""Figures workspace endpoints (W-Figures): NEXUS × Euclid comparison plates.

* ``GET /api/figures/nexus-plates``                  — every plate run (PNGs + provenance).
* ``POST /api/figures/nexus-plates``                 — render plates (local job
  ``figure-nexus-plates``): ``tiles`` (comma list), ``band``, ``model``, ``tag``.
* ``GET /api/figures/nexus-plates/<tag>/<name>``     — one plate PNG (``?thumb=<px>``: JPEG preview).
* ``POST /api/figures/nexus-plates/<tag>/delete``    — delete one run.

Local only (the plates read the cached NEXUS tiles and model outputs); JSON
``{ok:false, error}`` errors under ``/api/figures/``.
"""
from __future__ import annotations

import threading
from io import BytesIO

from flask import jsonify, request, send_file

from euclid_polish.web import errors
from euclid_polish.web.helpers import nexus_plates
from euclid_polish.web.jobs import REGISTRY

#: One plate render at a time (runs may share a tag directory).
_RENDER_LOCK = threading.Lock()


def _error(exc: nexus_plates.PlateError):
    return jsonify({"ok": False, "error": str(exc), **exc.extra}), exc.code


def register(app):
    errors.json_errors_for(app, "/api/figures/")

    @app.get("/api/figures/nexus-plates")
    def api_figures_nexus_plates():
        response = jsonify(nexus_plates.list_runs())
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.post("/api/figures/nexus-plates")
    def api_figures_nexus_plates_render():
        form = request.get_json(silent=True) if request.is_json else request.form.to_dict(flat=True)
        if not isinstance(form, dict):
            return jsonify({"ok": False, "error": "request body must be an object"}), 400
        tiles = form.get("tiles") or ""
        try:
            band = nexus_plates.check_band(form.get("band") or nexus_plates.DEFAULT_BAND)
            spec, resolved, _meta = nexus_plates.resolve_request(tiles, form.get("model") or "production")
            tag = (nexus_plates.check_tag(form["tag"]) if str(form.get("tag") or "").strip()
                   else nexus_plates.default_tag(spec))
        except nexus_plates.PlateError as exc:
            return _error(exc)
        ids = [tile.id for tile in resolved]

        def target(cap):
            with _RENDER_LOCK:
                record = nexus_plates.render_plates(
                    ids, band=band, model=spec, tag=tag, progress=cap.tick)
            print(f"wrote {len(record['tiles'])} tile plate(s) + {record['sheet']} to {tag}/")
            return {"tag": tag, "band": band, "model": spec, "sheet": record["sheet"],
                    "tiles": [tile["file"] for tile in record["tiles"]]}

        job_id = REGISTRY.spawn(
            f"NEXUS plates · {nexus_plates.model_short(spec)} · {band} · {len(ids)} tile(s)",
            target, kind="figure-nexus-plates")
        return jsonify({"ok": True, "job_id": job_id, "tag": tag, "band": band,
                        "model": spec, "tiles": ids})

    @app.get("/api/figures/nexus-plates/<tag>/<name>")
    def api_figures_nexus_plate_file(tag: str, name: str):
        try:
            path = nexus_plates.run_file(tag, name)
            thumb = request.args.get("thumb")
            if thumb:
                try:
                    side = int(thumb)
                except ValueError:
                    return jsonify({"ok": False, "error": "thumb must be an integer"}), 400
                return send_file(BytesIO(nexus_plates.thumbnail(path, side)), mimetype="image/jpeg",
                                 max_age=0, etag=False, last_modified=path.stat().st_mtime)
        except nexus_plates.PlateError as exc:
            return _error(exc)
        download = request.args.get("download", "").lower() in {"1", "true", "yes"}
        return send_file(path, mimetype="image/png", as_attachment=download,
                         download_name=f"{tag}_{name}", max_age=0)

    @app.post("/api/figures/nexus-plates/<tag>/delete")
    def api_figures_nexus_plates_delete(tag: str):
        try:
            removed = nexus_plates.delete_run(tag)
        except nexus_plates.PlateError as exc:
            return _error(exc)
        return jsonify({"ok": True, "tag": removed})
