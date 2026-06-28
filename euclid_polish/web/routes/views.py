"""views routes for the EuclidPolish web UI (extracted from app.py)."""
from __future__ import annotations

import contextlib
import io
import os
import threading as _t
from typing import Any

from flask import abort, jsonify, render_template, request, send_file

from euclid_polish.config import Config
from euclid_polish.training.log_plot import plot_training_log
from euclid_polish.web import fasrc_fetcher as _fasrc_fetcher
from euclid_polish.web.helpers import sky_records
from euclid_polish.web.helpers.fits_render import _render_psf_panel_png
from euclid_polish.web.helpers.paths import _sky_records_local_dir, _sky_records_remote_dir
from euclid_polish.web.helpers.sky_render import _render_catalog_view_png, _render_psf_clusters_png
from euclid_polish.web.helpers.status import (
    _cached_fasrc_psf_dir,
    _fasrc_catalog_dir,
    _list_vis_pngs,
    _record_count,
    _resolve_training_log,
)
from euclid_polish.web.jobs import REGISTRY as JOB_REGISTRY


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

    @app.route("/view/psf-clusters")
    def view_psf_clusters():
        # Local sky map of the ePSF clusters + per-cluster diameter. Reads the
        # VIS ePSF from the SAME synced FASRC cache the rest of the /psfs page
        # uses (so the plot tracks a freshly-synced ePSF, not a stale local
        # copy). 404s (via the renderer) when no ePSF is synced yet — the page
        # hides the panel then. Uses the cached catalog for member diameters,
        # degrading to a centroids-only map when absent.
        psf_dir = _cached_fasrc_psf_dir()
        psf_path = (os.path.join(psf_dir, Config.BAND_VIS.psf_fits_filename)
                    if psf_dir else None)
        out = _fasrc_catalog_dir(force=False)
        png = _render_psf_clusters_png(out, psf_path)
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
        # The VIS-only model logs to ``<ckpt>-vis``; give its plot a distinct
        # filename so it doesn't clobber (or get clobbered by) the 4-channel
        # plot's cache. The default dir keeps the canonical name for
        # back-compat (tracking image backups reference it).
        is_vis = any(part.endswith("-vis")
                     for part in os.path.normpath(ckpt).split(os.sep))
        png_name = "training_log-vis.png" if is_vis else "training_log.png"
        out_png  = os.path.abspath(os.path.join(Config.VIS_DIR, png_name))
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
                    with contextlib.suppress(OSError):
                        os.remove(tmp_png)
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
        targets: dict[str, str] = {}
        for kind in ("clean", "dirty", "hr"):
            targets[f"{kind}_validate"] = f"{remote_dir}/{kind}_validate.tfrecord"
            if include_train:
                targets[f"{kind}_train"] = f"{remote_dir}/{kind}_train.tfrecord"
        targets["sources_validate"] = f"{remote_dir}/sources_validate.csv"
        if include_train:
            targets["sources_train"] = f"{remote_dir}/sources_train.csv"
        max_bytes = 5 * 1024 * 1024 * 1024
        results: dict[str, dict[str, Any]] = {}
        any_ok = False
        for key, remote in targets.items():
            r = _fasrc_fetcher.fetch_one_file(remote, force=True, max_bytes=max_bytes)
            entry: dict[str, Any] = {"ok": r.ok, "size_bytes": r.size_bytes}
            if r.ok:
                any_ok = True
            else:
                entry["error"] = r.error
            results[key] = entry
        return jsonify({"ok": any_ok, "files": results,
                        "include_train": include_train})

    @app.route("/api/sky/sr-status")
    def api_sky_sr_status():
        """State for the /sky "Generate SR" button + SR tier.

        ``can_generate`` is true only when both the dirty records and a local
        checkpoint are present; ``sr`` reports how many SR cubes exist per
        subset (which drives the SR tier's enabled state in the viewer)."""
        records_dir = _sky_records_local_dir()
        subsets = sky_records.present_subsets(records_dir)
        ckpt = sky_records.checkpoint_present()
        return jsonify({
            "records": bool(subsets),
            "checkpoint": ckpt,
            "can_generate": bool(subsets) and ckpt,
            "subsets": subsets,
            "sr": {s: sky_records.sr_count(s) for s in sky_records.SUBSETS},
        })

    @app.route("/api/sky/generate-sr", methods=["POST"])
    def api_sky_generate_sr():
        """Run the SR model over the local dirty records as a background job.

        Gated on records + checkpoint (refuses 400 otherwise). Returns a
        ``job_id`` the page polls; the SR tier enables once cubes land."""
        records_dir = _sky_records_local_dir()
        subsets = sky_records.present_subsets(records_dir)
        if not subsets:
            return jsonify({"error": "no sky records — sync them first"}), 400
        if not sky_records.checkpoint_present():
            return jsonify({"error": "no checkpoint in ./ckpt/wdsr"}), 400
        overwrite = (str(request.values.get("overwrite", "")).lower()
                     in ("1", "true", "yes", "on"))

        def _run(cap):
            import numpy as np

            from euclid_polish.image import ImageSet
            from euclid_polish.image.tfio import tfrecord_path as _trp
            from euclid_polish.model import Model
            model = Model(Config.DEFAULT_CHECKPOINT_DIR)
            os.makedirs(sky_records.sky_sr_dir(), exist_ok=True)
            done = 0
            for subset in subsets:
                lr_path = _trp(records_dir, f"dirty_{subset}")
                if not os.path.exists(lr_path) or (
                    not overwrite and sky_records.sr_count(subset) > 0
                ):
                    continue
                lr = ImageSet.read(lr_path)
                sr_set = model.upsample_batch(
                    lr,
                    on_progress=lambda i, t, l: cap.tick(i, t, l),
                    log=lambda m: cap.write(m if m.endswith("\n") else m + "\n"),
                )
                for i, sr in enumerate(sr_set):
                    np.save(sky_records.sr_path(subset, i), sr.data)
                    done += 1
            return done

        job_id = JOB_REGISTRY.spawn("sky: generate SR", _run)
        return jsonify({"job_id": job_id})
