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
import re
import subprocess
from typing import Any, Dict, List, Optional

from flask import abort, jsonify, render_template, request, send_file

from euclid_polish.config import Config
from euclid_polish.web import fasrc_config
from euclid_polish.web.jobs import REGISTRY as JOB_REGISTRY
from euclid_polish.web.remote import STATE

#: Repo root (…/euclid_polish/web/routes/evaluation.py → up 4).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

#: STEP: cur/tot — emitted by the eval scripts via Reporter; we parse it from a
#: subprocess job's output to drive the local progress bar.
_STEP_RE = re.compile(r"STEP:\s*([\d,]+)\s*/\s*([\d,]+)")


def _zoobot_python() -> Optional[str]:
    """Locate the isolated EuclidPolishZoobot env's Python (torch is not in the
    main env). ``EUCLID_POLISH_ZOOBOT_PYTHON`` overrides; else probe the usual
    conda locations. Returns the interpreter path or ``None``."""
    override = os.environ.get("EUCLID_POLISH_ZOOBOT_PYTHON")
    if override and os.path.exists(override):
        return override
    for base in ("~/miniforge3", "~/mambaforge", "~/miniconda3", "~/anaconda3",
                 "/opt/miniforge3", "/opt/anaconda3", "/opt/miniconda3"):
        cand = os.path.expanduser(
            os.path.join(base, "envs", "EuclidPolishZoobot", "bin", "python"))
        if os.path.exists(cand):
            return cand
    return None


def _spawn_subprocess_job(label: str, cmd: list, result: dict):
    """Spawn a local background job running ``cmd``; stream stdout to the job
    log and parse ``STEP: cur/tot`` lines into the progress bar. Returns the
    job id. Raises (→ job 'failed') on non-zero exit."""
    def _run(cap):
        cap.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(
            cmd, cwd=_REPO_ROOT, env=os.environ.copy(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1)
        for line in proc.stdout:
            cap.write(line)
            m = _STEP_RE.search(line)
            if m:
                cap.tick(int(m.group(1).replace(",", "")),
                         int(m.group(2).replace(",", "")), "")
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"{os.path.basename(cmd[1])} exited {rc}")
        return result
    return JOB_REGISTRY.spawn(label, _run)


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


def _shared_run_summary() -> Dict[str, Any]:
    """Summary for the single shared evaluation store."""
    root = Config.EVAL_RESULTS_DIR
    rows = _read_manifest(root)
    n_ok = sum(1 for r in rows if str(r.get("ok", "")).lower() == "true")
    mani = os.path.join(root, "manifest.csv")
    return {
        "name": "eval_results",
        "run": "eval_results",
        "rows": rows,
        "n": len(rows),
        "n_ok": n_ok,
        "mtime": os.path.getmtime(mani) if os.path.isfile(mani) else 0,
    }


def _bad_run_arg(value: str) -> bool:
    return bool(value) and (
        os.sep in value or (os.altsep and os.altsep in value)
        or value in (".", "..")
    )


def register(app):

    @app.route("/evaluation")
    def evaluation_page():
        return render_template(
            "evaluation.html",
            catalogs=_list_catalogs(),
            run_summary=_shared_run_summary(),
            default_cutout=256,
            default_asinh=float(Config.STRETCH_SCALE_E),
            default_clip=_DEFAULT_CLIP,
        )

    @app.route("/api/evaluation/run-eval", methods=["POST"])
    def api_evaluation_run_eval():
        """Run the SR model over a catalog LOCALLY as a background job.

        In-process (TF is in the WebUI env): loads the local checkpoint, loops
        the catalog writing per-object FITS + manifest, reporting progress to
        the job's bar + log. Returns a job id to poll via ``/api/jobs/<id>``.
        """
        f = request.form
        grade = (f.get("grade") or "").strip() or None
        try:
            max_n = int(f.get("max_n", 0) or 0)
            cutout = int(f.get("cutout_size", 256) or 256)
        except ValueError:
            return jsonify({"ok": False, "error": "max_n / cutout must be ints"}), 400
        catalog = (f.get("catalog") or "").strip() or None
        if catalog:
            catalog = os.path.join(Config.DATA_DIR, catalog)
        out_dir = Config.EVAL_RESULTS_DIR

        from euclid_polish.eval import catalog_runner

        def _run(cap):
            return catalog_runner.run_catalog_eval(
                out_dir=out_dir, catalog_path=catalog, grade=grade,
                max_n=(max_n or None), cutout_size=cutout,
                on_progress=lambda i, n, lbl: cap.tick(i, n, lbl),
                log=lambda m: cap.write(m if m.endswith("\n") else m + "\n"),
            )
        job_id = JOB_REGISTRY.spawn("eval: eval_results", _run)
        return jsonify({"ok": True, "job_id": job_id})

    @app.route("/api/evaluation/run-grouped", methods=["POST"])
    def api_evaluation_run_grouped():
        """Prepare the unified grouped dataset LOCALLY (A/B/C + synthetic).

        One in-process background job: N lens cutouts per grade + N synthetic
        validation triptychs → one run dir with a single grouped manifest.
        """
        f = request.form
        try:
            n = int(f.get("n", 5) or 5)
            cutout = int(f.get("cutout_size", 256) or 256)
            stamp_m = int(f.get("stamp_m", 64) or 64)
        except ValueError:
            return jsonify({"ok": False, "error": "n / cutout / M must be ints"}), 400
        stamp_m = max(16, min(256, stamp_m + (stamp_m % 2)))  # even, bounded
        include_synth = str(f.get("synthetic", "1")).lower() in ("1", "true", "on", "yes")
        out_dir = Config.EVAL_RESULTS_DIR

        from euclid_polish.eval import grouped_runner

        def _run(cap):
            return grouped_runner.run_grouped_analysis(
                out_dir=out_dir, n=n, cutout_size=cutout, stamp_m=stamp_m,
                include_synthetic=include_synth,
                on_progress=lambda i, t, lbl: cap.tick(i, t, lbl),
                log=lambda m: cap.write(m if m.endswith("\n") else m + "\n"))
        job_id = JOB_REGISTRY.spawn("grouped: eval_results", _run)
        return jsonify({"ok": True, "job_id": job_id})

    @app.route("/api/evaluation/run-zoobot", methods=["POST"])
    def api_evaluation_run_zoobot():
        """Score Zoobot morphology for a run LOCALLY (CPU) as a background job.

        Zoobot is PyTorch (separate env), so this runs scripts/zoobot_morphology.py
        in the EuclidPolishZoobot env as a subprocess, streaming its output to
        the job log. Returns a job id, or 400 with an install hint if the env
        is missing.
        """
        f = request.form
        run = (f.get("run") or "").strip()
        if _bad_run_arg(run):
            return jsonify({"ok": False, "error": "bad run name"}), 400
        if not os.path.isdir(Config.EVAL_RESULTS_DIR):
            return jsonify({"ok": False, "error": "eval_results not found"}), 404
        py = _zoobot_python()
        if py is None:
            return jsonify({"ok": False, "error": (
                "Zoobot env not found. Create it once with "
                "`mamba env create -f environment-zoobot.yml`, or set "
                "EUCLID_POLISH_ZOOBOT_PYTHON to its python.")}), 400
        cmd = [py, os.path.join(_REPO_ROOT, "scripts", "zoobot_morphology.py"),
               "--run-dir", Config.EVAL_RESULTS_DIR, "--device", "cpu"]
        tree_ckpt = (f.get("tree_checkpoint") or "").strip()
        if tree_ckpt:
            cmd += ["--tree-checkpoint", tree_ckpt]
        job_id = _spawn_subprocess_job(
            "zoobot: eval_results", cmd, {"run": "eval_results"})
        return jsonify({"ok": True, "job_id": job_id})

    @app.route("/api/evaluation/run-lensfinder", methods=["POST"])
    def api_evaluation_run_lensfinder():
        """Score eval objects with the trained lens-finder heads LOCALLY (CPU).

        Runs scripts/lensfinder_score_eval.py in the EuclidPolishZoobot env over
        the shared store, writing ``lens_scores.csv`` (P(lens) per object/recon)
        that the PCA opacity + lens-identification panel consume. Returns a job
        id, or 400 with an install hint if the env is missing.
        """
        f = request.form
        if not os.path.isdir(Config.EVAL_RESULTS_DIR):
            return jsonify({"ok": False, "error": "eval_results not found"}), 404
        py = _zoobot_python()
        if py is None:
            return jsonify({"ok": False, "error": (
                "Zoobot env not found. Create it once with "
                "`mamba env create -f environment-zoobot.yml`.")}), 400
        heads = (f.get("heads_dir") or "").strip() or os.path.join(
            Config.DATA_DIR, "lensfinder", "heads")
        cmd = [py, os.path.join(_REPO_ROOT, "scripts", "lensfinder_score_eval.py"),
               "--run-dir", Config.EVAL_RESULTS_DIR, "--heads-dir", heads,
               "--device", "cpu"]
        job_id = _spawn_subprocess_job(
            "lensfinder: eval_results", cmd, {"run": "eval_results"})
        return jsonify({"ok": True, "job_id": job_id})

    @app.route("/api/evaluation/runs")
    def api_evaluation_runs():
        run = request.args.get("run", "").strip()
        if _bad_run_arg(run):
            abort(400)
        return jsonify(_shared_run_summary())

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
        summary = _shared_run_summary()
        return jsonify({
            "ok":     True,
            "stdout": out.strip()[-2000:],
            "n":      summary["n"],
            "n_ok":   summary["n_ok"],
        })

    @app.route("/api/evaluation/rerender", methods=["POST"])
    def api_evaluation_rerender():
        """Drop a run's cached eye/solar PNGs so they re-render from the FITS.

        Rendering is local and lazy (see :func:`serve_eval_files`), so removing
        the PNGs makes the next gallery view regenerate them with the current
        plotting code — no cluster round-trip.
        """
        run = (request.form.get("run") or request.args.get("run") or "").strip()
        if _bad_run_arg(run):
            abort(400)
        rd = Config.EVAL_RESULTS_DIR
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

    @app.route("/api/evaluation/morphology")
    def api_evaluation_morphology():
        """Render + serve the run-level Zoobot morphology summary PNG.

        404 when the run has no ``morphology_manifest.csv`` yet (Zoobot hasn't
        been run). Rendered locally from the manifest + raw predictions and
        cached to ``<run>/morphology_summary.png``; ``?fresh=1`` re-renders.
        """
        run = (request.args.get("run") or "").strip()
        if _bad_run_arg(run):
            abort(400)
        # Absolute path: send_file resolves a *relative* path against the app
        # root (euclid_polish/web), not the CWD, so a relative EVAL_RESULTS_DIR
        # would 500 at serve time.
        run_dir = os.path.abspath(Config.EVAL_RESULTS_DIR)
        if not os.path.isfile(os.path.join(run_dir, "morphology_manifest.csv")):
            abort(404)
        out_png = os.path.join(run_dir, "morphology_summary.png")
        fresh = request.args.get("fresh", "").lower() in ("1", "true", "yes")
        if fresh or not os.path.isfile(out_png):
            from euclid_polish.eval import zoobot_morph
            if zoobot_morph.render_morphology_summary(run_dir, out_png) is None:
                abort(404)
        return send_file(out_png, mimetype="image/png", max_age=0)

    @app.route("/api/evaluation/morphology-embedding")
    def api_evaluation_morphology_embedding():
        """Return 3-D PCA and MDS coordinates for Zoobot feature vectors."""
        run = (request.args.get("run") or "").strip()
        if _bad_run_arg(run):
            abort(400)
        run_dir = os.path.abspath(Config.EVAL_RESULTS_DIR)
        if not os.path.isfile(os.path.join(run_dir, "zoobot_predictions.csv")):
            abort(404)
        from euclid_polish.eval import zoobot_morph
        payload = zoobot_morph.morphology_embedding_payload(run_dir)
        if payload is None:
            abort(404)
        payload["run"] = "eval_results"
        return jsonify(payload)

    @app.route("/api/evaluation/transformation")
    def api_evaluation_transformation():
        """Render + serve the run-level SR-transformation summary PNG.

        404 when the run has no ``manifest.csv``. Cached to
        ``<run>/transformation_summary.png``; ``?fresh=1`` re-renders.
        """
        run = (request.args.get("run") or "").strip()
        if _bad_run_arg(run):
            abort(400)
        run_dir = os.path.abspath(Config.EVAL_RESULTS_DIR)
        if not os.path.isfile(os.path.join(run_dir, "manifest.csv")):
            abort(404)
        out_png = os.path.join(run_dir, "transformation_summary.png")
        fresh = request.args.get("fresh", "").lower() in ("1", "true", "yes")
        if fresh or not os.path.isfile(out_png):
            from euclid_polish.eval import zoobot_morph
            if zoobot_morph.render_transformation_summary(run_dir, out_png) is None:
                abort(404)
        return send_file(out_png, mimetype="image/png", max_age=0)

    @app.route("/api/evaluation/lensfinder-summary")
    def api_evaluation_lensfinder_summary():
        """Render + serve the lens-identification analysis PNG.

        404 until the lens-finder has scored this run (``lens_scores.csv``).
        Cached to ``<run>/lensfinder_summary.png``; ``?fresh=1`` re-renders.
        """
        run = (request.args.get("run") or "").strip()
        if _bad_run_arg(run):
            abort(400)
        run_dir = os.path.abspath(Config.EVAL_RESULTS_DIR)
        if not os.path.isfile(os.path.join(run_dir, "lens_scores.csv")):
            abort(404)
        out_png = os.path.join(run_dir, "lensfinder_summary.png")
        fresh = request.args.get("fresh", "").lower() in ("1", "true", "yes")
        if fresh or not os.path.isfile(out_png):
            from euclid_polish.eval import lensfinder_eval
            if lensfinder_eval.render_lensfinder_summary(run_dir, out_png) is None:
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
