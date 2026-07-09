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

from euclid_polish.image import Image
from euclid_polish.web.helpers import viewer_data
from euclid_polish.web.helpers.viewer_data import ViewerError


def _params() -> dict:
    """Whitelisted collection params from the query string. ``members`` is the
    ensemble disagreement movie's member subset (CSV of indices) — the sr/pcaN
    cubes are then recomputed on the fly over just those members."""
    out = {}
    for key in ("subset", "mode", "members"):
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

        # Serialize via the Image atom: little-endian float32, C-contiguous,
        # so the browser reads the raw bytes straight into a Float32Array.
        # One source of truth for the wire format (Image.to_raw_bytes).
        c = cube.shape[-1]
        img = Image(data=cube,
                    pixel_scale_arcsec=float(info.get("pixscale", 0.0)),
                    band_names=tuple(viewer_data.BAND_NAMES[:c]),
                    is_clean=True)
        body = img.to_raw_bytes()
        h, w, c = img.wire_meta()["shape"]
        resp = Response(body, mimetype="application/octet-stream")
        resp.headers["X-Cube-Shape"] = f"{h},{w},{c}"
        resp.headers["X-Cube-Bands"] = ",".join(viewer_data.BAND_NAMES)
        resp.headers["X-Cube-Label"] = str(info.get("label", ""))
        resp.headers["X-Cube-Asinh"] = repr(float(info.get("asinh", 100.0)))
        resp.headers["X-Cube-Pixscale"] = repr(float(info.get("pixscale", 0.0)))
        exposed = ["X-Cube-Shape", "X-Cube-Bands", "X-Cube-Label",
                   "X-Cube-Asinh", "X-Cube-Pixscale"]
        # PCA eigen-image cubes carry the (subset-dependent) amplitude + variance
        # the disagreement movie animates by — the client reads them per-PC off
        # the header rather than a static per-field manifest.
        if "amp" in info:
            resp.headers["X-Cube-Amp"] = repr(float(info["amp"]))
            exposed.append("X-Cube-Amp")
        if "var" in info:
            resp.headers["X-Cube-Var"] = repr(float(info["var"]))
            exposed.append("X-Cube-Var")
        # Expose the custom headers to fetch() under any CORS posture.
        resp.headers["Access-Control-Expose-Headers"] = ",".join(exposed)
        resp.headers["Cache-Control"] = "no-cache"
        return resp
