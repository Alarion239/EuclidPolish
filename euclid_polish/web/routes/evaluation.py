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
from euclid_polish.web.fasrc_pipeline import REGISTRY as STEP_REGISTRY


def _list_catalogs() -> List[Dict[str, Any]]:
    """List normalized evaluation catalogs (``*.csv``) under EVAL_CATALOG_DIR."""
    root = Config.EVAL_CATALOG_DIR
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(root):
        return out
    for dirpath, _dirs, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith(".csv"):
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
