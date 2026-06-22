"""evaluation routes — catalog-based SR evaluation.

Drives the ``eval_catalog`` FASRC pipeline step (run the model over a catalog
of real sky targets — the headline use is the Natalie Lines Euclid Q1
strong-lens catalog) and surfaces the mirrored-back results as a gallery.

Job *submission* reuses the generic FASRC plumbing: the page POSTs to the
existing ``/api/fasrc/hst/eval_catalog/submit`` endpoint (queue + confirm +
sbatch are all shared). This module only adds read-only views:

  * ``/evaluation``               — the page (submit form + results gallery)
  * ``/api/evaluation/runs``      — list runs, or one run's manifest rows
  * ``/eval-files/<path>``        — serve PNG/FITS from Config.EVAL_RESULTS_DIR
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, List

from flask import abort, jsonify, render_template, request, send_file

from euclid_polish.config import Config
from euclid_polish.web import fasrc_config
from euclid_polish.web.fasrc_pipeline import REGISTRY as STEP_REGISTRY
from euclid_polish.web.remote import STATE


def _list_catalogs() -> List[Dict[str, Any]]:
    """List normalized evaluation catalogs (``*.csv``) under EVAL_CATALOG_DIR.

    Skips the raw Zenodo download (``q1_discovery_engine_lens_catalog.csv``) so
    only the normalized ``id,ra,dec`` catalogs the pipeline consumes are shown.
    """
    from euclid_polish.euclid.lens_catalog import SOURCE_CSV

    root = Config.EVAL_CATALOG_DIR
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(root):
        return out
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith(".csv") or fn == SOURCE_CSV:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            try:
                with open(full) as f:
                    rows = max(0, sum(1 for _ in f) - 1)
            except OSError:
                rows = 0
            # The script resolves --catalog relative to the data dir; report
            # the project-relative path so the form sends something the FASRC
            # job can find.
            out.append({
                "rel":  rel,
                "rows": rows,
                "data_rel": os.path.relpath(full, Config.DATA_DIR),
            })
    return out


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_manifest(run_dir: str) -> List[Dict[str, Any]]:
    """Manifest rows for a run, with Zoobot morphology deltas merged in by id.

    When ``morphology_manifest.csv`` (from the Zoobot step) is present, each
    object's row gets a ``morph`` dict so the gallery can show the before/after
    morphology delta alongside the pixel-level residual.
    """
    rows = _read_csv(os.path.join(run_dir, "manifest.csv"))
    morph = {m.get("id"): m
             for m in _read_csv(os.path.join(run_dir, "morphology_manifest.csv"))}
    if morph:
        for r in rows:
            m = morph.get(r.get("id"))
            if m:
                r["morph"] = m
    return rows


#: Gallery PNG prefix → plot_reconstruction rgb_mode. Rendering is done LOCALLY
#: from the per-object FITS (FASRC only runs the model + writes FITS), so
#: changing the plotting code (or the clip) is picked up by a re-render with no
#: cluster round-trip.
_REGIME_BY_PREFIX = {"eye": "eye", "solar": "calibrated"}

#: Default upper-clip percentile for the dirty LR panel.
_DEFAULT_CLIP = 99.5


def _render_object_png(obj_dir: str, rgb_mode: str, out_png: str,
                       hi_percentile: float | None = None,
                       asinh_scale: float | None = None) -> str | None:
    """Render an object's eye/solar PNG from its FITS.

    Reads ``original_stack.fits`` (4-band LR cube) + ``SR.fits`` (SR cube) in
    ``obj_dir`` and runs the shared ``plot_reconstruction`` — the same renderer
    the local inference page uses — writing to ``out_png``. ``hi_percentile``
    raises the dirty-panel clip so a bright central galaxy isn't saturated;
    ``asinh_scale`` overrides the asinh knee (``None`` → the SR FITS header's
    ASINH). These are the two knobs the interactive viewer drives. Returns
    ``out_png``, or ``None`` when the FITS inputs are missing.
    """
    sr_fits = os.path.join(obj_dir, "SR.fits")
    stack_fits = os.path.join(obj_dir, "original_stack.fits")
    if not (os.path.isfile(sr_fits) and os.path.isfile(stack_fits)):
        return None

    # Heavy imports (astropy / TF-backed inference) deferred to first render.
    import numpy as np
    from astropy.io import fits
    from euclid_polish.training.inference import plot_reconstruction

    with fits.open(sr_fits) as h:
        sr = np.asarray(h[0].data, dtype=np.float32)
        header_asinh = float(h[0].header.get("ASINH", Config.STRETCH_SCALE_E))
    with fits.open(stack_fits) as h:
        stack = np.asarray(h[0].data, dtype=np.float32)   # (4, H, W) or (H, W)

    lr_cube = np.moveaxis(stack, 0, -1) if stack.ndim == 3 else stack
    lr_vis = lr_cube[..., 0] if lr_cube.ndim == 3 else lr_cube
    sr_data = np.moveaxis(sr, 0, -1) if sr.ndim == 3 else sr
    plot_reconstruction(
        lr_vis, sr_data, hr_data=None, output_path=out_png,
        lr_cube=lr_cube if lr_cube.ndim == 3 else None,
        asinh_scale=(asinh_scale if asinh_scale is not None else header_asinh),
        rgb_mode=rgb_mode, dirty_hi_pct=hi_percentile,
    )
    return out_png


def _list_runs() -> List[Dict[str, Any]]:
    """List evaluation runs: sub-dirs of EVAL_RESULTS_DIR holding a manifest."""
    root = Config.EVAL_RESULTS_DIR
    runs: List[Dict[str, Any]] = []
    if not os.path.isdir(root):
        return runs
    for name in sorted(os.listdir(root)):
        rd = os.path.join(root, name)
        mani = os.path.join(rd, "manifest.csv")
        if not (os.path.isdir(rd) and os.path.isfile(mani)):
            continue
        rows = _read_manifest(rd)
        n_ok = sum(1 for r in rows if str(r.get("ok", "")).lower() == "true")
        runs.append({
            "name":  name,
            "n":     len(rows),
            "n_ok":  n_ok,
            "mtime": os.path.getmtime(mani),
        })
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def register(app):

    @app.route("/evaluation")
    def evaluation_page():
        step = STEP_REGISTRY.get("eval_catalog")
        return render_template(
            "evaluation.html",
            step=step,
            defaults=step.defaults.to_dict(),
            catalogs=_list_catalogs(),
            runs=_list_runs(),
            default_asinh=float(Config.STRETCH_SCALE_E),
            default_clip=_DEFAULT_CLIP,
        )

    @app.route("/api/evaluation/runs")
    def api_evaluation_runs():
        run = request.args.get("run", "").strip()
        if run:
            # One run's manifest. Jail the name to a single path component so a
            # crafted ``run`` can't escape EVAL_RESULTS_DIR.
            if os.sep in run or (os.altsep and os.altsep in run) \
                    or run in ("", ".", ".."):
                abort(400)
            rd = os.path.join(Config.EVAL_RESULTS_DIR, run)
            if not os.path.isdir(rd):
                abort(404)
            return jsonify({"run": run, "rows": _read_manifest(rd)})
        return jsonify({"runs": _list_runs()})

    @app.route("/api/evaluation/fetch-catalog", methods=["POST"])
    def api_evaluation_fetch_catalog():
        """Download + normalize the Euclid Q1 strong-lens catalog (Zenodo).

        Pulls the ~0.4 MB discovery CSV and writes the normalized
        ``lens_catalog/lenses.csv`` so the page is self-sufficient (no CLI
        step needed). Network failures surface as a 502 with the message.
        """
        from euclid_polish.euclid import lens_catalog

        try:
            out_csv, n = lens_catalog.fetch()
        except Exception as e:  # noqa: BLE001 — report any fetch failure to the UI
            return jsonify({"ok": False,
                            "error": f"{type(e).__name__}: {e}"}), 502
        return jsonify({
            "ok":   True,
            "rows": n,
            "path": out_csv,
            "rel":  os.path.relpath(out_csv, Config.EVAL_CATALOG_DIR),
        })

    @app.route("/api/evaluation/sync", methods=["POST"])
    def api_evaluation_sync():
        """Pull ``<data_dir>/eval_results`` down from FASRC into the gallery.

        The checkpoint auto-mirror (``fasrc_mirror``) only syncs the ckpt
        dir — eval-catalog runs land in ``<data_dir>/eval_results`` on the
        cluster and otherwise never reach the local ``/evaluation`` page.
        This is the one-shot rsync_pull, reusing the same ControlMaster
        transport the checkpoint mirror uses, so the user never has to drop
        to a terminal. ``--delete-after`` keeps local in lockstep with the
        remote (drops runs deleted on the cluster) without leaving a partial
        window mid-transfer.
        """
        if STATE.ssh is None or not STATE.ssh.is_connected():
            return jsonify({"ok": False, "error": "not connected"}), 400
        cfg = fasrc_config.load()
        remote = cfg.data_dir.rstrip("/") + "/eval_results/"
        local = Config.EVAL_RESULTS_DIR
        os.makedirs(local, exist_ok=True)
        try:
            rc, out, err = STATE.ssh.rsync_pull(
                remote, local,
                extra_args=["--delete-after"],
                timeout=600,
            )
        except Exception as e:  # noqa: BLE001 — surface any transport error to UI
            return jsonify({"ok": False,
                            "error": f"{type(e).__name__}: {e}"}), 500
        if rc != 0:
            return jsonify({"ok": False,
                            "error": err.strip() or f"rsync exit {rc}"}), 500
        runs = _list_runs()
        return jsonify({
            "ok":     True,
            "stdout": out.strip()[-2000:],
            "n_runs": len(runs),
            "runs":   runs,
        })

    @app.route("/api/evaluation/rerender", methods=["POST"])
    def api_evaluation_rerender():
        """Drop a run's cached eye/solar PNGs so they re-render from the FITS.

        Rendering is local and lazy (see :func:`serve_eval_files`), so removing
        the PNGs makes the next gallery view regenerate them with the current
        plotting code — no cluster round-trip.
        """
        run = (request.form.get("run") or request.args.get("run") or "").strip()
        if not run or os.sep in run or (os.altsep and os.altsep in run) \
                or run in (".", ".."):
            abort(400)
        rd = os.path.join(Config.EVAL_RESULTS_DIR, run)
        if not os.path.isdir(rd):
            abort(404)
        removed = 0
        for dirpath, _dirs, files in os.walk(rd):
            for fn in files:
                low = fn.lower()
                # eye.png / solar.png and their per-clip caches (eye__c99.9.png).
                if low.endswith(".png") and (low.startswith("eye")
                                             or low.startswith("solar")):
                    try:
                        os.remove(os.path.join(dirpath, fn))
                        removed += 1
                    except OSError:
                        pass
        return jsonify({"ok": True, "removed": removed})

    @app.route("/eval-files/<path:relpath>")
    def serve_eval_files(relpath: str):
        """Serve PNG / FITS from data/eval_results/ (jailed against traversal).

        Gallery PNGs are rendered locally on demand from the per-object FITS
        (FASRC writes only FITS): a missing PNG — or any request with
        ``?fresh=1`` — is (re-)rendered from ``SR.fits`` + ``original_stack.fits``
        with the current plotting code, then served. So tweaking the renderer
        is reflected by a re-render, never a cluster re-run.
        """
        root = os.path.realpath(Config.EVAL_RESULTS_DIR)
        full = os.path.realpath(os.path.join(root, relpath))
        if not full.startswith(root + os.sep):
            abort(403)
        lower = full.lower()
        if lower.endswith(".png"):
            obj_dir = os.path.dirname(full)
            prefix = os.path.basename(full).split(".")[0].split("__")[0].lower()
            rgb_mode = _REGIME_BY_PREFIX.get(prefix)

            # The two interactive knobs: upper-clip percentile and asinh knee.
            # Each (clip, asinh) combination caches to its own filename
            # (e.g. eye__c99.9__a120.png) so settings coexist and the browser
            # caches per URL; a slider revisit is then instant.
            def _farg(name):
                try:
                    v = request.args.get(name, "")
                    return float(v) if v else None
                except ValueError:
                    return None
            clip = _farg("clip") if rgb_mode else None
            asinh = _farg("asinh") if rgb_mode else None

            suffix = ""
            if clip is not None and abs(clip - _DEFAULT_CLIP) > 1e-6:
                suffix += f"__c{clip:g}"
            if asinh is not None and asinh > 0:
                suffix += f"__a{asinh:g}"
            cache = (os.path.join(obj_dir, f"{prefix}{suffix}.png")
                     if suffix else full)

            fresh = request.args.get("fresh", "").lower() in ("1", "true", "yes")
            if rgb_mode is not None and (fresh or not os.path.isfile(cache)):
                _render_object_png(obj_dir, rgb_mode, cache,
                                   hi_percentile=clip, asinh_scale=asinh)
            if not os.path.isfile(cache):
                abort(404)
            return send_file(cache, mimetype="image/png", max_age=0)
        if not os.path.isfile(full):
            abort(404)
        mt = ("application/fits" if lower.endswith(".fits")
              else "application/octet-stream")
        return send_file(full, mimetype=mt, as_attachment=True,
                         download_name=os.path.basename(full))
