"""Routes feeding the unified client-side cutout viewer.

Two endpoints, both backed by ``helpers/viewer_data`` (the collection
registry):

* ``GET /viewer/meta/<collection>``       — JSON meta + colour constants.
* ``GET /viewer/cube/<collection>/<i>``    — raw Float32 ``(H, W, C)`` cube.

The cube body is little-endian Float32 in C order; shape and per-cube
metadata travel in ``X-Cube-*`` response headers so the browser can
reshape without a JSON envelope. All heavy lifting (TFRecord/FITS reads,
calibration constants) lives in ``viewer_data``; this module is just the
HTTP surface.
"""
from __future__ import annotations

from flask import Response, abort, jsonify, request

from euclid_polish.web.helpers import viewer_data
from euclid_polish.web.helpers.viewer_data import ViewerError


def _params() -> dict:
    """Whitelisted collection params from the query string."""
    out = {}
    for key in ("subset",):
        val = request.args.get(key)
        if val is not None:
            out[key] = val
    return out


def register(app):

    @app.route("/viewer/meta/<collection>")
    def viewer_meta(collection: str):
        try:
            return jsonify(viewer_data.get_meta(collection, _params()))
        except ViewerError as e:
            abort(e.code)

    @app.route("/viewer/cube/<collection>/<int:index>")
    def viewer_cube(collection: str, index: int):
        tier = (request.args.get("tier") or "").strip()
        try:
            cube, info = viewer_data.get_cube(collection, index, tier, _params())
        except ViewerError as e:
            abort(e.code)

        # Force little-endian float32, C-contiguous, so the browser reads
        # the raw bytes straight into a Float32Array.
        body = cube.astype("<f4", copy=False).tobytes(order="C")
        h, w, c = cube.shape
        resp = Response(body, mimetype="application/octet-stream")
        resp.headers["X-Cube-Shape"] = f"{h},{w},{c}"
        resp.headers["X-Cube-Bands"] = ",".join(viewer_data.BAND_NAMES)
        resp.headers["X-Cube-Label"] = str(info.get("label", ""))
        resp.headers["X-Cube-Asinh"] = repr(float(info.get("asinh", 100.0)))
        resp.headers["X-Cube-Pixscale"] = repr(float(info.get("pixscale", 0.0)))
        # Expose the custom headers to fetch() under any CORS posture.
        resp.headers["Access-Control-Expose-Headers"] = (
            "X-Cube-Shape,X-Cube-Bands,X-Cube-Label,X-Cube-Asinh,X-Cube-Pixscale")
        resp.headers["Cache-Control"] = "no-cache"
        return resp
