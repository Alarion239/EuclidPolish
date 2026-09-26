"""Routes feeding the unified client-side cutout viewer.

Cube/meta endpoints are backed by ``helpers.viewer_data`` (the collection
registry), while durable crop and figure endpoints use
``helpers.viewer_results``:

* ``GET /viewer/meta/<collection>``       — JSON meta + colour constants
  (``?id=`` adds the ``index`` of that object).
* ``GET /viewer/cube/<collection>/<i>``    — raw Float32 ``(H, W, C)`` cube.
* ``GET /viewer/cube/<collection>?id=``    — the same, addressed by the
  object's stable meta ``id`` instead of its position.
* ``GET|POST /viewer/results``             — list or save matched raw crops.
* ``GET /viewer/results/<id>``             — one saved-result summary.
* ``GET /viewer/results/<id>/panel.png``   — render one saved panel.
* ``GET /viewer/results/grid.<png|pdf>``   — render an A4 result grid.

The cube body is little-endian Float32 in C order; shape and per-cube
metadata travel in ``X-Cube-*`` response headers so the browser can
reshape without a JSON envelope (contract C6 adds ``X-Cube-WCS`` — the
compact celestial WCS keywords of the tier's pixel grid — and
``X-Cube-Unit``). Every failure under ``/viewer/`` is JSON ``{"error"}`` with
the status code. All heavy lifting (TFRecord/FITS reads, calibration
constants) lives in ``viewer_data``; this module is just the HTTP surface.
"""
from __future__ import annotations

import json

from flask import Response, jsonify, request

from euclid_polish.image import Image
from euclid_polish.web import errors
from euclid_polish.web.helpers import viewer_data, viewer_results
from euclid_polish.web.helpers.viewer_data import ViewerError


def _error(code: int, message: str):
    return jsonify({"error": message}), code


def _band_names(info: dict, channels: int) -> tuple[str, ...]:
    """Channel names for ``X-Cube-Bands``: the collection's own, the four
    Euclid bands, or generic ``ch<i>`` for multi-channel (multi-knee) heads."""
    bands = tuple(info.get("bands") or ())
    if len(bands) == channels:
        return bands
    if channels <= len(viewer_data.BAND_NAMES):
        return tuple(viewer_data.BAND_NAMES[:channels])
    return tuple(f"ch{i}" for i in range(channels))


def _params() -> dict:
    """Whitelisted collection params from the query string. ``members`` is the
    ensemble disagreement movie's member subset (CSV of indices) — the sr/pcaN
    cubes are then recomputed on the fly over just those members."""
    out = {}
    for key in (
        "subset", "mode", "members", "field", "jwst_band",
        # ``real`` collection (C9): real-tile source + model-spec tiers.
        "source", "models",
        viewer_data.BHR_FWHM_PARAM,
        # PSF-page live preview: the client changes only the replay seed every
        # few seconds.  These remain harmless for every other collection.
        "psf_warp", "psf_warp_seed",
    ):
        val = request.args.get(key)
        if val is not None:
            out[key] = val
    return out


def register(app):

    # ``/viewer/*`` answers JSON errors (C6) — routing 404/405 and crashes
    # included — through the app's one shared HTTPException handler (never a
    # module-local ``@app.errorhandler``, which would replace the others').
    errors.json_errors_for(app, "/viewer/")

    @app.get("/viewer/results")
    def viewer_result_list():
        return jsonify(viewer_results.list_results())

    @app.post("/viewer/results")
    def viewer_result_save():
        if request.is_json:
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                return jsonify({"error": "request body must be a JSON object"}), 400
        else:
            payload = request.form.to_dict(flat=True)
        try:
            result = viewer_results.save_result(payload)
        except viewer_results.ViewerResultError as exc:
            return jsonify({"error": str(exc)}), exc.code
        response = jsonify({"id": result["id"], "result_id": result["id"], "result": result})
        response.status_code = 201
        response.headers["Location"] = f"/viewer/results/{result['id']}"
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/viewer/results/<result_id>")
    def viewer_result_get(result_id: str):
        try:
            result = viewer_results.get_result_summary(result_id)
        except viewer_results.ViewerResultError as exc:
            return jsonify({"error": str(exc)}), exc.code
        response = jsonify({"result": result})
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/viewer/results/<result_id>/panel.png")
    def viewer_result_panel(result_id: str):
        try:
            body = viewer_results.render_panel(
                result_id,
                request.args.get("tier", ""),
                request.args.get("mode", ""),
            )
        except viewer_results.ViewerResultError as exc:
            return jsonify({"error": str(exc)}), exc.code
        response = Response(body, mimetype="image/png")
        response.headers["Content-Disposition"] = (
            f'inline; filename="{result_id}_{request.args.get("tier", "panel")}.png"'
        )
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/viewer/results/grid.<output_format>")
    def viewer_result_grid(output_format: str):
        try:
            body = viewer_results.render_grid(
                request.args.getlist("result"),
                request.args.getlist("row"),
                output_format,
                request.args.get("dpi", str(viewer_results.DEFAULT_GRID_DPI)),
            )
        except viewer_results.ViewerResultError as exc:
            return jsonify({"error": str(exc)}), exc.code
        mimetype = "image/png" if output_format == "png" else "application/pdf"
        response = Response(body, mimetype=mimetype)
        response.headers["Content-Disposition"] = (
            f'inline; filename="viewer_results_grid.{output_format}"'
        )
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.route("/viewer/meta/<collection>")
    def viewer_meta(collection: str):
        object_id = request.args.get("id")
        try:
            meta = viewer_data.get_meta(collection, _params())
            if object_id is not None:
                meta["index"] = viewer_data.index_of(meta.get("objects") or [],
                                                     object_id)
            resp = jsonify(meta)
        except ViewerError as e:
            return _error(e.code, str(e))
        # Never let a stale meta stick: the ensemble cube cache is wiped +
        # rebuilt during an evaluation, so a meta fetched mid-run is briefly
        # empty — caching that would leave the viewer showing "no members"
        # long after the eval finished.
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.route("/viewer/cube/<collection>")
    def viewer_cube_by_id(collection: str):
        object_id = request.args.get("id")
        if not object_id:
            return _error(400, "pass ?id=<object id> or use /viewer/cube/<collection>/<index>")
        try:
            index = viewer_data.resolve_index(collection, object_id, _params())
        except ViewerError as e:
            return _error(e.code, str(e))
        return _cube_response(collection, index)

    @app.route("/viewer/cube/<collection>/<int:index>")
    def viewer_cube(collection: str, index: int):
        return _cube_response(collection, index)

    def _cube_response(collection: str, index: int):
        tier = (request.args.get("tier") or "").strip()
        try:
            cube, info = viewer_data.get_cube(collection, index, tier, _params())
        except ViewerError as e:
            return _error(e.code, str(e))

        # Serialize via the Image atom: little-endian float32, C-contiguous,
        # so the browser reads the raw bytes straight into a Float32Array.
        # One source of truth for the wire format (Image.to_raw_bytes).
        c = cube.shape[-1]
        cube_bands = _band_names(info, c)
        img = Image(data=cube,
                    pixel_scale_arcsec=float(info.get("pixscale", 0.0)),
                    band_names=cube_bands,
                    is_clean=True)
        body = img.to_raw_bytes()
        h, w, c = img.wire_meta()["shape"]
        resp = Response(body, mimetype="application/octet-stream")
        resp.headers["X-Cube-Shape"] = f"{h},{w},{c}"
        resp.headers["X-Cube-Bands"] = ",".join(cube_bands)
        resp.headers["X-Cube-Label"] = str(info.get("label", ""))
        resp.headers["X-Cube-Asinh"] = repr(float(info.get("asinh", 100.0)))
        resp.headers["X-Cube-Pixscale"] = repr(float(info.get("pixscale", 0.0)))
        resp.headers["X-Cube-Index"] = str(index)
        exposed = ["X-Cube-Shape", "X-Cube-Bands", "X-Cube-Label",
                   "X-Cube-Asinh", "X-Cube-Pixscale", "X-Cube-Index"]
        # PCA eigen-image cubes carry the (subset-dependent) amplitude + variance
        # the disagreement movie animates by — the client reads them per-PC off
        # the header rather than a static per-field manifest.
        if "amp" in info:
            resp.headers["X-Cube-Amp"] = repr(float(info["amp"]))
            exposed.append("X-Cube-Amp")
        if "var" in info:
            resp.headers["X-Cube-Var"] = repr(float(info["var"]))
            exposed.append("X-Cube-Var")
        if "tint" in info:
            tint = info["tint"]
            if isinstance(tint, (list, tuple)) and len(tint) == 3:
                resp.headers["X-Cube-Tint"] = ",".join(str(float(value)) for value in tint)
                exposed.append("X-Cube-Tint")
        if info.get("direct_rgb"):
            resp.headers["X-Cube-Direct-RGB"] = "1"
            exposed.append("X-Cube-Direct-RGB")
        if "transfer_group" in info:
            resp.headers["X-Cube-Transfer-Group"] = str(info["transfer_group"])
            exposed.append("X-Cube-Transfer-Group")
        # This is a viewer-only contrast factor.  The raw FITS payload is
        # deliberately left unchanged for download and science use.
        if "display_scale" in info:
            resp.headers["X-Cube-Display-Scale"] = repr(float(info["display_scale"]))
            exposed.append("X-Cube-Display-Scale")
        # C6: celestial WCS of THIS tier's pixel grid (FITS 1-based, axis 1 =
        # column) and the physical unit of the values.
        if info.get("wcs"):
            resp.headers["X-Cube-WCS"] = json.dumps(
                info["wcs"], separators=(",", ":"), sort_keys=True)
            exposed.append("X-Cube-WCS")
        if info.get("unit"):
            resp.headers["X-Cube-Unit"] = str(info["unit"])
            exposed.append("X-Cube-Unit")
        # Expose the custom headers to fetch() under any CORS posture.
        resp.headers["Access-Control-Expose-Headers"] = ",".join(exposed)
        resp.headers["Cache-Control"] = "no-cache"
        return resp
