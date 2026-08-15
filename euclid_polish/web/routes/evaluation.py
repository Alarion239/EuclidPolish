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
from typing import Any

from flask import abort, jsonify, render_template, request, send_file

from euclid_polish.config import Config
from euclid_polish.web import euclid_session, fasrc_config
from euclid_polish.web.jobs import REGISTRY as JOB_REGISTRY
from euclid_polish.web.remote import STATE


def _list_catalogs() -> list[dict[str, Any]]:
    """List normalized evaluation catalogs (``*.csv``) under EVAL_CATALOG_DIR.

    Skips the raw Zenodo download (``q1_discovery_engine_lens_catalog.csv``) so
    only the normalized ``id,ra,dec`` catalogs the pipeline consumes are shown.
    """
    from euclid_polish.eval.lens_catalog import SOURCE_CSV

    root = Config.EVAL_CATALOG_DIR
    out: list[dict[str, Any]] = []
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


def _read_csv(path: str) -> list[dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_manifest(run_dir: str) -> list[dict[str, Any]]:
    """Return manifest rows for an evaluation run."""
    return _read_csv(os.path.join(run_dir, "manifest.csv"))


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


def _run_summary(run_dir: str, run_name: str) -> dict[str, Any]:
    """Summary for one resolved evaluation run directory."""
    rows = _read_manifest(run_dir)
    n_ok = sum(1 for r in rows if str(r.get("ok", "")).lower() == "true")
    mani = os.path.join(run_dir, "manifest.csv")
    return {
        "name": run_name,
        "run": run_name,
        "rows": rows,
        "n": len(rows),
        "n_ok": n_ok,
        "mtime": os.path.getmtime(mani) if os.path.isfile(mani) else 0,
    }


def _shared_run_summary() -> dict[str, Any]:
    """Summary for the root shared evaluation store."""
    return _run_summary(Config.EVAL_RESULTS_DIR, "eval_results")


def _list_runs() -> list[dict[str, Any]]:
    """Evaluation runs that landed as sub-dirs of EVAL_RESULTS_DIR.

    The local grouped run writes one shared store (root ``manifest.csv``, see
    ``_shared_run_summary``), but the remote ``eval_results/`` pulled by the
    FASRC sync is organized per-catalog (``eval_results/<catalog>/manifest.csv``).
    This enumerates those run sub-dirs so the sync route can report how many
    appeared. Newest first by manifest mtime.
    """
    root = Config.EVAL_RESULTS_DIR
    runs: list[dict[str, Any]] = []
    if not os.path.isdir(root):
        return runs
    for name in sorted(os.listdir(root)):
        rd = os.path.join(root, name)
        if os.path.dirname(os.path.realpath(rd)) != os.path.realpath(root):
            continue
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


def _bad_run_arg(value: str) -> bool:
    return bool(value) and (
        "/" in value or "\\" in value
        or value in (".", "..")
    )


def _resolve_run_dir(
    value: str | None,
    *,
    required_file: str | None = None,
    allow_missing_root_file: bool = False,
) -> tuple[str, str]:
    """Resolve a root alias or direct child run without allowing escapes."""
    run = (value or "").strip()
    if _bad_run_arg(run) or "\x00" in run:
        abort(400)

    root = os.path.realpath(Config.EVAL_RESULTS_DIR)
    is_root = run in {"", "eval_results"}
    run_name = "eval_results" if is_root else run
    run_dir = root if is_root else os.path.realpath(os.path.join(root, run))
    if not is_root and os.path.dirname(run_dir) != root:
        abort(400)
    if not os.path.isdir(run_dir):
        if is_root and (required_file is None or allow_missing_root_file):
            return run_dir, run_name
        abort(404)
    if (required_file
            and not os.path.isfile(os.path.join(run_dir, required_file))
            and not (is_root and allow_missing_root_file)):
        abort(404)
    return run_dir, run_name


def register(app):

    @app.route("/evaluation")
    def evaluation_page():
        return render_template(
            "evaluation.html",
            catalogs=_list_catalogs(),
            run_summary=_shared_run_summary(),
            default_asinh=float(Config.STRETCH_SCALE_E),
            default_clip=_DEFAULT_CLIP,
            authenticated=euclid_session.is_authenticated(),
            current_user=euclid_session.current_user(),
        )

    @app.route("/api/evaluation/run-grouped", methods=["POST"])
    def api_evaluation_run_grouped():
        """Prepare the unified grouped dataset LOCALLY (A/B/C + synthetic).

        One in-process background job: N lens cutouts per grade + N synthetic
        validation triptychs → one run dir with a single grouped manifest. Every
        object is held at the canonical eval geometry (53² LR, 106² SR/HR), so
        there are no size knobs.

        The real-galaxy group is **cache-only**: it consumes whatever the
        standalone Query-galaxies step (``/api/evaluation/query-galaxies``) has
        already downloaded into ``galaxies.csv``. This run never logs in or
        queries the archive — if no galaxies are cached, that group is simply
        absent (the job log says so).
        """
        f = request.form
        try:
            n = int(f.get("n", 5) or 5)
        except ValueError:
            return jsonify({"ok": False, "error": "n must be an int"}), 400
        include_synth = str(f.get("synthetic", "1")).lower() in ("1", "true", "on", "yes")
        out_dir = Config.EVAL_RESULTS_DIR

        from euclid_polish.eval import grouped_runner

        def _run(cap):
            cap.write("model: ensemble mean (registry-active members)\n")
            return grouped_runner.run_grouped_analysis(
                out_dir=out_dir, n=n, include_synthetic=include_synth,
                include_galaxies=True,           # real galaxies always included (fixed control)
                on_progress=lambda i, t, lbl: cap.tick(i, t, lbl),
                log=lambda m: cap.write(m if m.endswith("\n") else m + "\n"))
        job_id = JOB_REGISTRY.spawn("grouped: eval_results", _run)
        return jsonify({"ok": True, "job_id": job_id})

    @app.route("/api/evaluation/query-galaxies", methods=["POST"])
    def api_evaluation_query_galaxies():
        """Query + cache the real-galaxy eval catalog as its own LOCAL step.

        Split out of the grouped run so the archive query is observable in
        isolation: this spawns a background job that runs
        :func:`~euclid_polish.eval.galaxy_catalog.build` with verbose logging
        (ADQL echo + per-field raw/kept/pool counts), streamed to the shared job
        panel. Needs the WebUI's authenticated Euclid session (``euclid_session``
        — ``Euclid.login`` on the process-global singleton); 400 if not logged
        in. The drawn set is cached to ``galaxies.csv``, which the grouped run
        then consumes (cache-only). ``n_galaxies`` is this step's own count.
        """
        client = euclid_session.catalog()
        if client is None:
            return jsonify({"ok": False, "error": (
                "Log in to the Euclid archive first — the galaxy cone queries "
                "need an authenticated session.")}), 400
        try:
            n_gal = int(request.form.get("n_galaxies", 15) or 15)
        except ValueError:
            return jsonify({"ok": False, "error": "n_galaxies must be an int"}), 400
        if n_gal <= 0:
            return jsonify({"ok": False, "error": "n_galaxies must be positive"}), 400
        # The drawn set is cached and only topped up; toggle this to discard the
        # cache and re-query (needed after a selection-criteria change so a stale
        # set isn't kept).
        regenerate = str(request.form.get("regenerate", "")).lower() in (
            "1", "true", "on", "yes")

        from euclid_polish.eval import catalog_runner, galaxy_catalog, lens_catalog

        # Galaxies are drawn from the strong-lens fields, so the lens catalog
        # must exist; fetch it from Zenodo if it's missing (same as the grouped
        # run), so this step is self-sufficient.
        catalog = catalog_runner.default_catalog_path()

        def _run(cap):
            def _log(m):
                cap.write(m if m.endswith("\n") else m + "\n")
            if not os.path.isfile(catalog):
                _log(f"lens catalog {catalog} not found — fetching from Zenodo…")
                lens_catalog.fetch(catalog)
            out_csv = galaxy_catalog.default_out_csv()
            path, n = galaxy_catalog.build(
                out_csv, n_galaxies=n_gal, lens_catalog_path=catalog,
                regenerate=regenerate, client=client, log=_log)
            return {"path": path, "n": n, "n_galaxies": n_gal}
        job_id = JOB_REGISTRY.spawn("galaxies: eval_results", _run)
        return jsonify({"ok": True, "job_id": job_id})

    @app.route("/api/evaluation/runs")
    def api_evaluation_runs():
        run_dir, run_name = _resolve_run_dir(
            request.args.get("run"),
            required_file="manifest.csv",
            allow_missing_root_file=True,
        )
        return jsonify(_run_summary(run_dir, run_name))

    @app.route("/api/evaluation/fetch-catalog", methods=["POST"])
    def api_evaluation_fetch_catalog():
        """Download + normalize the Euclid Q1 strong-lens catalog (Zenodo).

        Pulls the ~0.4 MB discovery CSV and writes the normalized
        ``lens_catalog/lenses.csv`` so the page is self-sufficient (no CLI
        step needed). Network failures surface as a 502 with the message.
        """
        from euclid_polish.eval import lens_catalog

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
        summary = _shared_run_summary()
        runs = _list_runs()
        return jsonify({
            "ok":     True,
            "stdout": out.strip()[-2000:],
            "n":      summary["n"],
            "n_ok":   summary["n_ok"],
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
        rd, _run_name = _resolve_run_dir(run)
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

    @app.route("/api/evaluation/transformation")
    def api_evaluation_transformation():
        """Render + serve the run-level SR-transformation summary PNG.

        404 when the run has no ``manifest.csv``. Cached to
        ``<run>/transformation_summary.png``; ``?fresh=1`` re-renders.
        """
        run = (request.args.get("run") or "").strip()
        run_dir, _run_name = _resolve_run_dir(run, required_file="manifest.csv")
        out_png = os.path.join(run_dir, "transformation_summary.png")
        fresh = request.args.get("fresh", "").lower() in ("1", "true", "yes")
        if fresh or not os.path.isfile(out_png):
            from euclid_polish.eval import transformation_summary
            if transformation_summary.render_transformation_summary(
                    run_dir, out_png) is None:
                abort(404)
        return send_file(out_png, mimetype="image/png", max_age=0)

    @app.route("/api/evaluation/angular-power-spectrum")
    def api_evaluation_angular_power_spectrum():
        """Render + serve the per-band HR-vs-SR angular power-spectrum PNG.

        Per-band T(k) and r(k) (linear + asinh) over the **sky validation
        fields** synced through /sky (HR ``clean`` record vs generated SR cube).
        404 until the records are synced and SR has been generated. Cached to
        ``<eval_results>/angular_power_spectrum.png``; ``?fresh=1`` re-renders.
        """
        run = (request.args.get("run") or "").strip()
        run_dir, _run_name = _resolve_run_dir(run)
        if not os.path.isdir(run_dir):
            abort(404)
        out_png = os.path.join(run_dir, "angular_power_spectrum.png")
        fresh = request.args.get("fresh", "").lower() in ("1", "true", "yes")
        if fresh or not os.path.isfile(out_png):
            from euclid_polish.eval import power_spectrum
            if power_spectrum.render_power_spectrum_summary(out_png) is None:
                abort(404)
        return send_file(out_png, mimetype="image/png", max_age=0)

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
