"""files routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

import io
import os

from flask import abort, jsonify, redirect, render_template, request, send_file, url_for

from euclid_polish.config import Config
from euclid_polish.web import fasrc_fetcher as _fasrc_fetcher
from euclid_polish.web.helpers.fits_render import (
    _fits_file_info,
    _read_fits_header_rows,
    _render_fits_to_png_adaptive,
)
from euclid_polish.web.helpers.paths import (
    _inspectable_roots,
    _resolve_inspectable_fits,
    _safe_relpath,
    _sky_records_local_dir,
)
from euclid_polish.web.helpers.sky_render import _export_sky_record_fits
from euclid_polish.web.helpers.status import (
    _catalog_status,
    _checkpoints_status,
    _psf_status,
    _tfrecords_status,
)
from euclid_polish.web.jobs import REGISTRY


def register(app):

    @app.route("/sky/fits")
    def sky_fits():
        """Export one sky-record band+index as FITS and return the file."""
        path = _export_sky_record_fits(
            subset=request.args.get("subset", ""),
            kind=request.args.get("kind", ""),
            band=request.args.get("band", ""),
            index=request.args.get("i", "0"),
            records_dir=_sky_records_local_dir(),
        )
        return send_file(path, as_attachment=True,
                         download_name=os.path.basename(path),
                         mimetype="application/fits")

    @app.route("/sky/inspect")
    def sky_inspect():
        """Export the requested record then redirect into the inspector."""
        path = _export_sky_record_fits(
            subset=request.args.get("subset", ""),
            kind=request.args.get("kind", ""),
            band=request.args.get("band", ""),
            index=request.args.get("i", "0"),
            records_dir=_sky_records_local_dir(),
        )
        return redirect(url_for("inspect_fits_page",
                                fits=_safe_relpath(path)))

    # ---------------- Static PNG server (data/vis/) ----------------
    @app.route("/vis/<path:relpath>")
    def serve_vis(relpath: str):
        full = os.path.realpath(os.path.join(Config.VIS_DIR, relpath))
        vis_root = os.path.realpath(Config.VIS_DIR)
        # Refuse anything that resolves outside data/vis (path traversal).
        if not full.startswith(vis_root + os.sep):
            abort(403)
        if not os.path.isfile(full):
            abort(404)
        return send_file(full, mimetype="image/png")

    @app.route("/inference-files/<path:relpath>")
    def serve_inference_files(relpath: str):
        """Serve FITS / PNG files from data/euclid_inference/.

        Used by the inference UI to download the persisted cutouts and
        SR result. Path is jailed to ``Config.EUCLID_INFERENCE_DIR`` to
        prevent traversal — anything resolving outside that tree 403s.
        """
        root = os.path.realpath(Config.EUCLID_INFERENCE_DIR)
        full = os.path.realpath(os.path.join(root, relpath))
        if not full.startswith(root + os.sep):
            abort(403)
        if not os.path.isfile(full):
            abort(404)
        # FITS gets the application/fits MIME so browsers prompt to
        # save instead of trying to render it as text.
        mt = ("application/fits" if full.lower().endswith(".fits")
              else "application/octet-stream")
        return send_file(
            full, mimetype=mt, as_attachment=True,
            download_name=os.path.basename(full),
        )

    # ---------------- Job tracker API ----------------
    @app.route("/api/jobs")
    def api_jobs():
        return jsonify(REGISTRY.list())

    @app.route("/api/jobs/<job_id>")
    def api_job(job_id: str):
        job = REGISTRY.get(job_id)
        if not job:
            abort(404)
        return jsonify(job.to_dict())

    @app.route("/api/status")
    def api_status():
        return jsonify({
            "catalog":     _catalog_status(),
            "psfs":        _psf_status(),
            "tfrecords":   _tfrecords_status(),
            "checkpoints": _checkpoints_status(),
        })

    # =========================================================================
    # Universal FITS inspector — every image card across the UI links here.
    # =========================================================================

    @app.route("/api/inspect")
    def api_inspect_fits():
        """Return the inspector payload consumed by the React page."""
        path = _resolve_inspectable_fits(request.args.get("fits", ""))
        return jsonify({
            "file": _fits_file_info(path),
            "hdus": _read_fits_header_rows(path),
            "rel": _safe_relpath(path),
            "allowed_roots": _inspectable_roots(),
        })

    @app.route("/inspect")
    def inspect_fits_page():
        path = _resolve_inspectable_fits(request.args.get("fits", ""))
        info = _fits_file_info(path)
        rows = _read_fits_header_rows(path)
        # Project-relative path is what shows in the UI + what the
        # download/preview routes echo back (so refresh from a bookmark
        # keeps working as long as the file is still at that location).
        rel = _safe_relpath(path)
        return render_template(
            "inspect_fits.html",
            file=info,
            hdus=rows,
            rel=rel,
            # Roots are displayed so the user can confirm which data
            # subtree the file came from (useful when triaging mismatched
            # outputs from multiple runs).
            allowed_roots=_inspectable_roots(),
        )

    @app.route("/inspect/download")
    def inspect_fits_download():
        path = _resolve_inspectable_fits(request.args.get("fits", ""))
        return send_file(
            path, as_attachment=True,
            download_name=os.path.basename(path),
            mimetype="application/fits",
        )

    @app.route("/fasrc/file/inspect")
    def fasrc_file_inspect():
        """Fetch one file from FASRC (cached) then redirect to /inspect.

        Query param: ``remote_path=<absolute path on FASRC>``. Subject
        to all the safeguards in :mod:`euclid_polish.web.fasrc_fetcher`
        (size cap, allowed roots, cache TTL).
        """
        remote = request.args.get("remote_path", "").strip()
        if not remote:
            abort(400)
        result = _fasrc_fetcher.fetch_one_file(remote)
        if not result.ok:
            return render_template(
                "fasrc_fetch_error.html", remote=remote, error=result.error,
            ), 502
        # Hand off to the existing inspector with the local cache path.
        return redirect(url_for(
            "inspect_fits_page",
            fits=_safe_relpath(result.local_path),
        ))

    @app.route("/fasrc/file/download")
    def fasrc_file_download():
        """Fetch one file from FASRC (cached) and send it back directly."""
        remote = request.args.get("remote_path", "").strip()
        if not remote:
            abort(400)
        result = _fasrc_fetcher.fetch_one_file(remote)
        if not result.ok:
            return jsonify({"ok": False, "error": result.error}), 502
        return send_file(
            result.local_path, as_attachment=True,
            download_name=os.path.basename(remote),
        )

    @app.route("/inspect/preview.png")
    def inspect_fits_preview():
        path = _resolve_inspectable_fits(request.args.get("fits", ""))
        try:
            size = int(request.args.get("size", 512))
        except ValueError:
            size = 512
        if size < 16 or size > 2048:
            abort(400)
        # /inspect is universal — could be a sky cutout, a PSF, a diff
        # kernel, a residual map, anything in the allowed roots. The
        # band-aware renderer assumes Euclid cutout units (~1000 e⁻ asinh
        # knee) and silently misrenders everything else; use the
        # data-adaptive ZScale + Asinh renderer instead.
        png = _render_fits_to_png_adaptive(path, size=size)
        # /inspect is interactive debugging — when the underlying FITS
        # gets regenerated (e.g. you re-run the differential-kernel
        # script with different params), the preview must reflect the
        # new file on the next page load. A long ``max_age`` here means
        # the browser shows the stale render for an hour, which is the
        # opposite of what an inspector is for.
        resp = send_file(io.BytesIO(png), mimetype="image/png", max_age=0)
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp
