"""Figures workspace endpoints (W-Figures): NEXUS × Euclid comparison plates.

* ``GET /api/figures/nexus-plates``                  — every plate run (PNGs + provenance).
* ``POST /api/figures/nexus-plates``                 — render plates (local job
  ``figure-nexus-plates``): ``tiles`` (comma list), ``band``, ``model``, ``tag``.
* ``GET /api/figures/nexus-plates/<tag>/<name>``     — one plate PNG (``?thumb=<px>``: JPEG preview).
* ``POST /api/figures/nexus-plates/<tag>/delete``    — delete one run.
* ``GET /api/figures/real-sr``                        — the newest cached
  production SRs of real tiles (Home's thumbnail strip).
* ``GET /api/figures/real-sr/<source>/<identifier>.jpg`` — a colour JPEG
  preview of one cached production SR (``?size=<px>``).

Local only (the plates read the cached NEXUS tiles and model outputs); JSON
``{ok:false, error}`` errors under ``/api/figures/``. The real-SR previews
only read the C9 output store: nothing here ever runs a model.
"""
from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from flask import jsonify, request, send_file
from PIL import Image as PILImage

from euclid_polish.config import Config
from euclid_polish.web import errors
from euclid_polish.web.helpers import model_catalog, nexus_plates, real_tiles
from euclid_polish.web.jobs import REGISTRY

#: One plate render at a time (runs may share a tag directory).
_RENDER_LOCK = threading.Lock()


def _error(exc: nexus_plates.PlateError):
    return jsonify({"ok": False, "error": str(exc), **exc.extra}), exc.code


# ---------------------------------------------------------------------------
# cached production SRs of real tiles (read-only previews)
# ---------------------------------------------------------------------------

REAL_SR_LIMIT = 6
REAL_SR_MAX_LIMIT = 24
REAL_SR_THUMB = 240
REAL_SR_MAX_THUMB = 640
_REAL_THUMB_CACHE_MAX = 48
_REAL_THUMBS: OrderedDict[tuple[str, int, int], bytes] = OrderedDict()
_REAL_THUMB_LOCK = threading.Lock()


def _production_sidecars() -> list[tuple[Path, dict[str, Any]]]:
    """``(fits path, sidecar)`` of every cached production output whose FITS
    exists, newest first (one glob over the output store; no pixels read)."""
    slug = model_catalog.spec_slug(model_catalog.SPEC_PRODUCTION)
    rows: list[tuple[Path, dict[str, Any]]] = []
    root = model_catalog.outputs_root()
    if not root.is_dir():
        return rows
    for meta_path in root.glob(f"*/*/{slug}.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        fits_path = meta_path.parent / str(meta.get("file") or f"{slug}.fits")
        if fits_path.is_file():
            meta.setdefault("source", meta_path.parent.parent.name)
            meta.setdefault("id", meta_path.parent.name)
            rows.append((fits_path, meta))
    rows.sort(key=lambda row: str(row[1].get("created") or ""), reverse=True)
    return rows


def real_sr_items(limit: int = REAL_SR_LIMIT) -> dict[str, Any]:
    """The newest cached production SRs: ``{total, items:[{ref, source, id,
    source_label, label, created, state, thumb}]}``."""
    current = model_catalog.current_fingerprints()
    rows = _production_sidecars()
    items = []
    for _path, meta in rows[:max(0, limit)]:
        source, identifier = str(meta["source"]), str(meta["id"])
        items.append({
            "ref": f"{source}/{identifier}", "source": source, "id": identifier,
            "source_label": real_tiles.SOURCE_INFO.get(source, {}).get("label", source),
            "label": meta.get("label"), "created": meta.get("created"),
            "state": model_catalog.output_state(meta, current),
            "thumb": f"/api/figures/real-sr/{source}/{identifier}.jpg",
        })
    return {"total": len(rows), "items": items}


def _block_mean(cube: np.ndarray, side: int) -> np.ndarray:
    """Shrink ``(H, W, C)`` by an integer block mean to roughly 2× ``side``."""
    factor = max(1, min(cube.shape[0], cube.shape[1]) // max(1, 2 * side))
    if factor == 1:
        return cube
    h, w = (cube.shape[0] // factor) * factor, (cube.shape[1] // factor) * factor
    trimmed = cube[:h, :w]
    return trimmed.reshape(h // factor, factor, w // factor, factor, -1).mean(axis=(1, 3))


def real_sr_thumbnail(source: str, identifier: str, side: int = REAL_SR_THUMB) -> bytes:
    """A colour JPEG (the viewer's Temp rendering, longest side ≤ ``side``)
    of one cached production SR, memoised per file state.
    :class:`FileNotFoundError` when the tile has no cached production SR."""
    side = max(64, min(int(side), REAL_SR_MAX_THUMB))
    directory = model_catalog.output_dir(source, identifier)
    path = directory / f"{model_catalog.spec_slug(model_catalog.SPEC_PRODUCTION)}.fits"
    if not path.is_file():
        raise FileNotFoundError(f"no cached production SR for {source}/{identifier}")
    stat = path.stat()
    key = (os.fspath(path), int(stat.st_mtime_ns), side)
    with _REAL_THUMB_LOCK:
        cached = _REAL_THUMBS.get(key)
        if cached is not None:
            _REAL_THUMBS.move_to_end(key)
            return cached
    cube, _header, _meta = model_catalog.load_output(source, identifier, model_catalog.SPEC_PRODUCTION)
    bands = tuple(Config.LR_INPUT_BAND_NAMES[:cube.shape[-1]])
    image, _style = nexus_plates.panel_image(
        _block_mean(np.nan_to_num(cube), side), {"bands": bands, "asinh": Config.STRETCH_SCALE_E}, "temp")
    pixels = np.asarray(image, np.float32)
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[..., None], 3, axis=-1)
    picture = PILImage.fromarray((np.clip(pixels[..., :3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGB")
    picture.thumbnail((side, side), PILImage.Resampling.LANCZOS)
    buffer = BytesIO()
    picture.save(buffer, format="JPEG", quality=85, optimize=True)
    body = buffer.getvalue()
    with _REAL_THUMB_LOCK:
        _REAL_THUMBS[key] = body
        while len(_REAL_THUMBS) > _REAL_THUMB_CACHE_MAX:
            _REAL_THUMBS.popitem(last=False)
    return body


def _int_arg(name: str, default: int) -> int:
    raw = request.args.get(name)
    if raw in (None, ""):
        return default
    return int(raw)


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

    @app.get("/api/figures/nexus-plates/<tag>/export")
    def api_figures_nexus_plate_export(tag: str):
        """Redraw one rendered run's contact sheet (``?tile=N``: that tile's
        plate) at ``?dpi=150|300|600`` as ``?format=png|pdf|svg`` from the
        cached tile outputs (read-only; the run's ``band`` and ``model`` pick
        the render). A long sheet's dpi drops to stay under 40 Mpx: the
        ``X-Plate-Dpi`` header and the file name carry the dpi used."""
        try:
            dpi = _int_arg("dpi", 300)
            tile_raw = request.args.get("tile")
            tile = int(tile_raw) if tile_raw not in (None, "") else None
        except ValueError:
            return jsonify({"ok": False, "error": "dpi and tile must be integers"}), 400
        fmt = (request.args.get("format") or "png").strip().lower()
        try:
            with _RENDER_LOCK:
                body, name, used = nexus_plates.export_render(
                    tag, request.args.get("band") or nexus_plates.DEFAULT_BAND,
                    (request.args.get("model") or "").strip(), fmt=fmt, dpi=dpi, tile=tile)
        except nexus_plates.PlateError as exc:
            return _error(exc)
        response = send_file(BytesIO(body), mimetype=nexus_plates.EXPORT_FORMATS[fmt],
                             as_attachment=True, download_name=name, max_age=0)
        response.headers["X-Plate-Dpi"] = str(used)
        return response

    @app.get("/api/figures/real-sr")
    def api_figures_real_sr():
        try:
            limit = max(0, min(_int_arg("limit", REAL_SR_LIMIT), REAL_SR_MAX_LIMIT))
        except ValueError:
            return jsonify({"ok": False, "error": "limit must be an integer"}), 400
        response = jsonify(real_sr_items(limit))
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/figures/real-sr/<source>/<identifier>.jpg")
    def api_figures_real_sr_thumb(source: str, identifier: str):
        try:
            side = _int_arg("size", REAL_SR_THUMB)
        except ValueError:
            return jsonify({"ok": False, "error": "size must be an integer"}), 400
        try:
            body = real_sr_thumbnail(source, identifier, side)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        return send_file(BytesIO(body), mimetype="image/jpeg", max_age=0, etag=False)

    @app.post("/api/figures/nexus-plates/<tag>/delete")
    def api_figures_nexus_plates_delete(tag: str):
        try:
            removed = nexus_plates.delete_run(tag)
        except nexus_plates.PlateError as exc:
            return _error(exc)
        return jsonify({"ok": True, "tag": removed})
