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

    @app.route("/eval-files/<path:relpath>")
    def serve_eval_files(relpath: str):
        """Serve PNG / FITS from data/eval_results/ (jailed against traversal)."""
        root = os.path.realpath(Config.EVAL_RESULTS_DIR)
        full = os.path.realpath(os.path.join(root, relpath))
        if not full.startswith(root + os.sep):
            abort(403)
        if not os.path.isfile(full):
            abort(404)
        lower = full.lower()
        if lower.endswith(".png"):
            return send_file(full, mimetype="image/png")
        mt = ("application/fits" if lower.endswith(".fits")
              else "application/octet-stream")
        return send_file(full, mimetype=mt, as_attachment=True,
                         download_name=os.path.basename(full))
