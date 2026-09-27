"""files routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

import io
import os
from urllib.parse import urlencode

from flask import abort, jsonify, redirect, request, send_file

from euclid_polish.config import Config
from euclid_polish.web import errors
from euclid_polish.web import fasrc_fetcher as _fasrc_fetcher
from euclid_polish.web.fasrc_gate import requires_fasrc
from euclid_polish.web.helpers import fits_inspect
from euclid_polish.web.helpers.fits_render import (
    _fits_file_info,
    _render_fits_to_png_adaptive,
)
from euclid_polish.web.helpers.paths import (
    _inspectable_roots,
    _resolve_inspectable_fits,
    _safe_relpath,
    browse_inspectable,
    inspect_roots,
    root_of,
)
from euclid_polish.web.helpers.status import (
    _cached_catalog_status,
    _catalog_status,
    _checkpoints_status,
    _psf_status,
    _tfrecords_status,
)
from euclid_polish.web.jobs import REGISTRY
from euclid_polish.web.version import process_tracker


def _int_arg(name: str, default: int, *, lo: int | None = None, hi: int | None = None) -> int:
    """An integer query parameter; 400 ``{error}`` when malformed or out of range."""
    raw = request.args.get(name, "")
    if raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        abort(400, description=f"{name} must be an integer, got {raw!r}")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        abort(400, description=f"{name} must be within {lo}…{hi}")
    return value


def _inspect_call(fn, *args, **kwargs):
    """Run a :mod:`fits_inspect` helper; its ``InspectError`` → JSON ``{error}``."""
    try:
        return fn(*args, **kwargs)
    except fits_inspect.InspectError as exc:
        abort(exc.code, description=str(exc))


def register(app):

    # Every error under the inspector's endpoints is JSON ``{error}`` with the
    # status (the SPA shows the server's text), never an HTML abort page.
    errors.json_errors_for(app, "/api/inspect", "/inspect/")

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

    # ---------------- Job tracker API (contract C2) ----------------
    @app.route("/api/jobs")
    def api_jobs():
        """Local background jobs, newest first; ``?summary=1`` omits logs."""
        summary = request.args.get("summary", "").lower() in ("1", "true", "yes")
        return jsonify(REGISTRY.list(summary=summary))

    @app.route("/api/jobs/<job_id>")
    def api_job(job_id: str):
        job = REGISTRY.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": f"unknown job {job_id}"}), 404
        return jsonify(job.to_dict())

    @app.post("/api/jobs/<job_id>/cancel")
    def api_job_cancel(job_id: str):
        """Cooperative cancel: the job stops at its next ``cap.tick``."""
        outcome = REGISTRY.cancel(job_id)
        if outcome is None:
            return jsonify({"ok": False, "error": f"unknown job {job_id}"}), 404
        if outcome is False:
            job = REGISTRY.get(job_id)
            status = job.status if job is not None else "finished"
            return jsonify({"ok": False,
                            "error": f"job {job_id} is already {status}"}), 409
        return jsonify({"ok": True})

    # ---------------- Server version (contract C3) ----------------
    @app.get("/api/version")
    def api_version():
        """Boot commit vs live HEAD, dirty flag and the served SPA build."""
        return jsonify(process_tracker().payload())

    @app.route("/api/status")
    def api_status():
        """Local status summary — cache-only and cheap (no SSH, no rsync)."""
        return jsonify({
            "catalog":     _cached_catalog_status(),
            "psfs":        _psf_status(),
            "tfrecords":   _tfrecords_status(),
            "checkpoints": _checkpoints_status(),
        })

    @app.post("/api/status/refresh-catalog")
    @requires_fasrc
    def api_status_refresh_catalog():
        """Explicitly re-pull the FASRC ``stars.csv`` (forced rsync)."""
        return jsonify({"ok": True, "catalog": _catalog_status()})

    # =========================================================================
    # Universal FITS inspector (the Inspect workspace, spec §8.6). Every
    # path is jailed to the inspectable roots (helpers/paths.py); headers are
    # read without pixels; planes are memory-mapped and binned.
    # =========================================================================

    @app.route("/api/inspect")
    def api_inspect_fits():
        """File card + every HDU's header summary and cards (no pixels)."""
        path = _resolve_inspectable_fits(request.args.get("fits", ""))
        summary = _inspect_call(fits_inspect.file_summary, path, cards=True)
        return jsonify({
            "file": _fits_file_info(path),
            "hdus": summary["hdus"],
            "band_groups": summary["band_groups"],
            "scan_truncated": summary["scan_truncated"],
            "stamp": summary["stamp"],
            "rel": _safe_relpath(path),
            "root": root_of(path),
            "allowed_roots": _inspectable_roots(),
            "roots": inspect_roots(),
        })

    @app.get("/api/inspect/browse")
    def api_inspect_browse():
        """The file browser: roots, one directory, or a bounded FITS search."""
        return jsonify(browse_inspectable(request.args.get("dir", "").strip(),
                                          request.args.get("q", "")))

    @app.get("/api/inspect/image/stats")
    def api_inspect_image_stats():
        """Statistics + histogram of one plane (or a 1-D HDU's series)."""
        path = _resolve_inspectable_fits(request.args.get("fits", ""))
        hdu = _int_arg("hdu", 0, lo=0)
        plane = _int_arg("plane", 0, lo=0)
        summary = _inspect_call(fits_inspect.file_summary, path)
        row = next((h for h in summary["hdus"] if h["index"] == hdu), None)
        if row is not None and row.get("type") == "vector":
            return jsonify(_inspect_call(fits_inspect.vector_series, path, hdu))
        return jsonify(_inspect_call(fits_inspect.plane_stats, path, hdu, plane))

    @app.get("/api/inspect/table")
    def api_inspect_table():
        """One page of a table HDU's rows (optionally sorted by a column)."""
        path = _resolve_inspectable_fits(request.args.get("fits", ""))
        return jsonify(_inspect_call(
            fits_inspect.table_page, path, _int_arg("hdu", 1, lo=0),
            offset=_int_arg("offset", 0, lo=0),
            limit=_int_arg("limit", 200, lo=1, hi=fits_inspect.TABLE_PAGE_MAX),
            sort=(request.args.get("sort") or "").strip() or None,
            desc=request.args.get("desc", "").lower() in ("1", "true", "yes"),
        ))

    @app.get("/api/inspect/table/stats")
    def api_inspect_table_stats():
        """Per-column statistics of a table HDU."""
        path = _resolve_inspectable_fits(request.args.get("fits", ""))
        return jsonify(_inspect_call(fits_inspect.table_stats, path, _int_arg("hdu", 1, lo=0)))

    @app.get("/api/inspect/provenance")
    def api_inspect_provenance():
        """The file's PROVID stamp, its sidecars and related records."""
        path = _resolve_inspectable_fits(request.args.get("fits", ""))
        return jsonify(fits_inspect.provenance(path))

    @app.route("/inspect/download")
    def inspect_fits_download():
        path = _resolve_inspectable_fits(request.args.get("fits", ""))
        return send_file(
            path, as_attachment=True,
            download_name=os.path.basename(path),
            mimetype="application/fits",
        )

    @app.route("/fasrc/file/inspect")
    @requires_fasrc
    def fasrc_file_inspect():
        """Fetch one file from FASRC (cached) then redirect to ``/inspect``.

        Query param: ``remote_path=<absolute path on FASRC>``. Subject
        to all the safeguards in :mod:`euclid_polish.web.fasrc_fetcher`
        (size cap, allowed roots, cache TTL).
        """
        remote = request.args.get("remote_path", "").strip()
        if not remote:
            abort(400)
        result = _fasrc_fetcher.fetch_one_file(remote)
        if not result.ok or result.local_path is None:
            return jsonify({"ok": False, "error": result.error}), 502
        # Hand off to the Inspect workspace with the local cache path.
        return redirect("/inspect?" + urlencode(
            {"fits": _safe_relpath(result.local_path)}))

    @app.route("/fasrc/file/download")
    @requires_fasrc
    def fasrc_file_download():
        """Fetch one file from FASRC (cached) and send it back directly."""
        remote = request.args.get("remote_path", "").strip()
        if not remote:
            abort(400)
        result = _fasrc_fetcher.fetch_one_file(remote)
        if not result.ok or result.local_path is None:
            return jsonify({"ok": False, "error": result.error}), 502
        return send_file(
            result.local_path, as_attachment=True,
            download_name=os.path.basename(remote),
        )

    @app.route("/inspect/preview.png")
    def inspect_fits_preview():
        """PNG thumbnail of one plane (``hdu``, ``plane``; default the first
        image HDU's first plane), longer side ``size`` px (16–2048)."""
        path = _resolve_inspectable_fits(request.args.get("fits", ""))
        try:
            size = int(request.args.get("size", 512))
        except ValueError:
            size = 512
        if size < 16 or size > 2048:
            abort(400, description="size must be within 16…2048")
        hdu_raw = request.args.get("hdu", "")
        hdu = _int_arg("hdu", 0, lo=0) if hdu_raw != "" else None
        # /inspect is universal — could be a sky cutout, a PSF, a diff
        # kernel, a residual map, anything in the allowed roots. The
        # band-aware renderer assumes Euclid cutout units (~1000 e⁻ asinh
        # knee) and silently misrenders everything else; use the
        # data-adaptive ZScale + Asinh renderer instead.
        png = _render_fits_to_png_adaptive(path, size=size, hdu=hdu,
                                           plane=_int_arg("plane", 0, lo=0))
        # /inspect is interactive debugging — when the underlying FITS
        # gets regenerated (e.g. you re-run the differential-kernel
        # script with different params), the preview must reflect the
        # new file on the next page load. A long ``max_age`` here means
        # the browser shows the stale render for an hour, which is the
        # opposite of what an inspector is for.
        resp = send_file(io.BytesIO(png), mimetype="image/png", max_age=0)
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp
