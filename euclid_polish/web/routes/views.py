"""views routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

from euclid_polish.config import Config
from euclid_polish.training.log_plot import plot_training_log
from euclid_polish.web import fasrc_config
from euclid_polish.web import fasrc_fetcher as _fasrc_fetcher
from euclid_polish.web.fasrc_fetcher import _local_path_for
from flask import abort
from flask import jsonify
from flask import render_template
from flask import request
from flask import send_file
from typing import Any
from typing import Dict
import io
import os
import threading as _t
from euclid_polish.web.helpers.fits_render import _render_psf_panel_png
from euclid_polish.web.helpers.paths import _sky_records_local_dir, _sky_records_remote_dir
from euclid_polish.web.helpers.sky_render import _render_catalog_view_png, _render_sky_record_png
from euclid_polish.web.helpers.status import _fasrc_catalog_dir, _list_vis_pngs, _record_count, _resolve_training_log


def register(app):

    # ---------------- Visualization page ----------------
    @app.route("/visualization")
    def visualization_page():
        return render_template("visualization.html",
                               pngs=_list_vis_pngs())

    # ---------------- Live view renderers (PNG) ----------------
    @app.route("/view/psfs")
    def view_psfs():
        band = request.args.get("band", "all")
        png = _render_psf_panel_png(None if band == "all" else band)
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=0)

    @app.route("/view/sky")
    def view_sky():
        subset = request.args.get("subset", "validate")
        kind   = request.args.get("kind",   "clean")
        band   = request.args.get("band",   "VIS")
        try:
            idx = int(request.args.get("i", 0))
        except ValueError:
            idx = 0
        # Render from the FASRC-synced records cache (populated by
        # /api/sky/sync), not the local data dir.
        png = _render_sky_record_png(
            subset, kind, band, idx, records_dir=_sky_records_local_dir(),
        )
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=0)

    @app.route("/view/catalog")
    def view_catalog():
        view = request.args.get("view", "positions")
        # Render from the FASRC catalog (pulled to the local cache), not a
        # stale local stars.csv — the query writes it on netscratch.
        out = _fasrc_catalog_dir(force=True)
        if out is None:
            abort(404)
        png = _render_catalog_view_png(view, out)
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=0)

    @app.route("/view/training-log")
    def view_training_log():
        ckpt = request.args.get("checkpoint_dir", Config.DEFAULT_CHECKPOINT_DIR)
        log_path = _resolve_training_log(ckpt)
        # ABSOLUTE path: Config.VIS_DIR is "./data/vis" (relative), but Flask's
        # send_file() resolves a *relative* path against app.root_path
        # (euclid_polish/web/), not the CWD — so a relative out_png renders to
        # one dir and is served from another → 500. abspath pins both to CWD.
        out_png  = os.path.abspath(
            os.path.join(Config.VIS_DIR, "training_log.png"))
        if log_path is None:
            abort(404)
        # Render if missing, stale, or explicitly forced (``?force=1`` — used
        # by the "Visualize" button so the PNG is current before tracking).
        force = request.args.get("force") in ("1", "true", "yes")
        if (force or not os.path.exists(out_png)
                or os.path.getmtime(log_path) > os.path.getmtime(out_png)):
            os.makedirs(os.path.dirname(out_png), exist_ok=True)
            # Render to a private temp file, then publish atomically. The PNG
            # is a single shared path the page polls + the Visualize/track
            # buttons force-render; without the temp+rename a concurrent read
            # could send_file() a half-written (or briefly absent) file.
            # Keep a ``.png`` suffix so matplotlib infers the format from the
            # temp path; the pid/thread infix keeps concurrent renders apart.
            tmp_png = f"{out_png}.tmp-{os.getpid()}-{_t.get_ident()}.png"
            try:
                plot_training_log(log_path, tmp_png)
                os.replace(tmp_png, out_png)
            except Exception as e:
                # Empty / header-only / mid-write log → there's nothing to
                # plot yet. Don't 500: serve the last good render if we have
                # one, otherwise 404 so the page shows its placeholder.
                if os.path.exists(tmp_png):
                    try:
                        os.remove(tmp_png)
                    except OSError:
                        pass
                if not os.path.exists(out_png):
                    print(f"  ⚠ training-log plot skipped: {type(e).__name__}: {e}")
                    abort(404)
        if not os.path.exists(out_png):
            abort(404)
        return send_file(out_png, mimetype="image/png", max_age=0)

    @app.route("/api/sky/totals")
    def api_sky_totals():
        local = _sky_records_local_dir()
        return jsonify({
            name: _record_count(name, records_dir=local)
            for name in ("clean_train", "clean_validate",
                         "dirty_train", "dirty_validate",
                         "hr_train",    "hr_validate")
        })

    @app.route("/api/sky/sync", methods=["POST"])
    def api_sky_sync():
        """Rsync the synthetic TFRecord shards from FASRC into the local
        cache so the preview can render them.

        ``include_train`` (default false) controls whether the large
        train-split files are pulled; validate shards are always included.
        Lifts the fetcher's 50 MB cap to 5 GB since TFRecords are large and
        this is an explicit user-requested transfer."""
        remote_dir = _sky_records_remote_dir()
        include_train = (request.values.get("include_train", "false")
                         .lower() in ("1", "true", "yes", "on"))
        targets: Dict[str, str] = {}
        for kind in ("clean", "dirty", "hr"):
            targets[f"{kind}_validate"] = f"{remote_dir}/{kind}_validate.tfrecord"
            if include_train:
                targets[f"{kind}_train"] = f"{remote_dir}/{kind}_train.tfrecord"
        max_bytes = 5 * 1024 * 1024 * 1024
        results: Dict[str, Dict[str, Any]] = {}
        any_ok = False
        for key, remote in targets.items():
            r = _fasrc_fetcher.fetch_one_file(remote, force=True, max_bytes=max_bytes)
            entry: Dict[str, Any] = {"ok": r.ok, "size_bytes": r.size_bytes}
            if r.ok:
                any_ok = True
            else:
                entry["error"] = r.error
            results[key] = entry
        return jsonify({"ok": any_ok, "files": results,
                        "include_train": include_train})
