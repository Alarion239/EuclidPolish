"""evaluation routes — catalog-based SR evaluation.

Drives the ``eval_catalog`` FASRC pipeline step (run the model over a catalog
of real sky targets — the headline use is the Natalie Lines Euclid Q1
strong-lens catalog) and surfaces the mirrored-back results as a gallery.

The grouped run, galaxy query and lens-catalogue fetch are local jobs; the
results are browsed through the viewer collection ``evaluation`` (the page is
Sky › Targets, Models › Images and Models › Diagnostics). This module adds:

  * ``/api/evaluation/runs``      — one run's manifest rows, each with the
    staleness of its SR against the model an evaluation would load now
  * ``/api/evaluation/objects/<id>`` — one object's provenance card
  * ``/api/evaluation/sync``      — pull FASRC results (``confirm=1``)
  * run-level PNG summaries (transformation, angular power spectrum)
  * ``/eval-files/<path>``        — jailed per-object FITS download

Errors under ``/api/evaluation/`` and ``/eval-files/`` are JSON ``{error}``.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
from typing import Any

from astropy.io import fits
from flask import abort, jsonify, request, send_file

from euclid_polish.config import Config
from euclid_polish.eval import (
    catalog_runner,
    galaxy_catalog,
    grouped_runner,
    lens_catalog,
    power_spectrum,
    transformation_summary,
)
from euclid_polish.sky.observation.q1_fields import q1_field_for
from euclid_polish.web import errors, euclid_session, fasrc_config
from euclid_polish.web.fasrc_gate import requires_fasrc
from euclid_polish.web.jobs import REGISTRY as JOB_REGISTRY
from euclid_polish.web.remote import STATE
from euclid_polish.web.security import fresh_requested, refuse_cross_site_cache_fill

#: The angular power-spectrum curves written beside its PNG (the interactive plot).
_APS_JSON = "angular_power_spectrum.json"
#: Object sub-directory names (``out_subdir``): one path segment, no dot-files.
_SAFE_OBJECT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,219}$")
#: Manifest grade → object kind.
_KINDS = {"A": "lens", "B": "lens", "C": "lens", "gal": "galaxy",
          "syn-lens": "synthetic", "syn-gal": "synthetic"}
#: The object FITS offered for download (tier → file).
_DOWNLOADS = {"LR": "original_stack.fits", "SR": "SR.fits", "mean": "mean.fits",
              "HR": "HR.fits", "BHR": "BHR.fits", "std": "std.fits"}
#: SR.fits header cards shown on the object card (provenance + geometry).
_SR_CARDS = ("OBJECT", "BANDS", "BUNIT", "CKPT", "ASINH", "RA", "DEC", "CSIZE",
             "SRCX", "SRCY", "PROVID", "PROVKIND", "PROVGIT")


def _read_csv(path: str) -> list[dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_manifest(run_dir: str) -> list[dict[str, Any]]:
    """Return manifest rows for an evaluation run."""
    return _read_csv(os.path.join(run_dir, "manifest.csv"))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_ok(row: dict[str, Any]) -> bool:
    return str(row.get("ok", "")).strip().lower() == "true"


def _identity_payload(identity: dict[str, Any]) -> dict[str, Any]:
    labels = list(identity.get("member_labels") or [])
    return {"n_members": len(labels), "member_labels": labels,
            "combiner_kind": identity.get("combiner_kind"),
            "combiner_fingerprint": identity.get("combiner_fingerprint")}


def _enrich_row(row: dict[str, Any], run_dir: str,
                identity: dict[str, Any]) -> dict[str, Any]:
    """A manifest row + ``kind``, ``field``, the SR's model ``state`` against
    ``identity`` (``current|stale|unknown``, ``null`` for a failed object),
    ``state_reason``, the recorded model (``n_members``, ``combiner_kind``),
    ``tiers`` on disk, the viewer object id and the C9 real-tile ref."""
    out = dict(row)
    sub = str(row.get("out_subdir") or row.get("id") or "")
    grade = str(row.get("grade") or "").strip()
    ra, dec = _finite(row.get("ra")), _finite(row.get("dec"))
    kind = _KINDS.get(grade, "synthetic" if grade.startswith("syn") else "lens")
    out.update({"kind": kind, "field": q1_field_for(ra, dec) if ra is not None and dec is not None
                else None, "viewer_id": sub or None, "realtile": None, "state": None,
                "state_reason": None, "n_members": None, "combiner_kind": None, "tiers": []})
    if not (_is_ok(row) and sub and _SAFE_OBJECT.fullmatch(sub)):
        return out
    obj_dir = os.path.join(run_dir, sub)
    state = catalog_runner.object_model_state(obj_dir, identity)
    out.update({"state": state["state"], "state_reason": state["reason"],
                "n_members": state["n_members"], "combiner_kind": state["combiner_kind"],
                "tiers": [tier for tier, name in _DOWNLOADS.items()
                          if os.path.isfile(os.path.join(obj_dir, name))]})
    if kind != "synthetic" and ra is not None and dec is not None:
        out["realtile"] = f"eval/{sub}"
    return out


def _run_summary(run_dir: str, run_name: str) -> dict[str, Any]:
    """Summary for one resolved evaluation run directory: every manifest row
    enriched (:func:`_enrich_row`), the model an evaluation would load now
    (``current``), state ``counts`` and per-grade ``groups`` of the ok rows."""
    identity = catalog_runner.current_eval_identity()
    rows = [_enrich_row(r, run_dir, identity) for r in _read_manifest(run_dir)]
    ok_rows = [r for r in rows if _is_ok(r)]
    counts = {"current": 0, "stale": 0, "unknown": 0}
    groups: dict[str, int] = {}
    for r in ok_rows:
        if r.get("state") in counts:
            counts[str(r["state"])] += 1
        grade = str(r.get("grade") or "").strip() or "—"
        groups[grade] = groups.get(grade, 0) + 1
    mani = os.path.join(run_dir, "manifest.csv")
    return {
        "name": run_name,
        "run": run_name,
        "rows": rows,
        "n": len(rows),
        "n_ok": len(ok_rows),
        "mtime": os.path.getmtime(mani) if os.path.isfile(mani) else 0,
        "current": _identity_payload(identity),
        "counts": counts,
        "groups": groups,
        "columns": list(catalog_runner.MANIFEST_COLS),
    }


def _provenance(obj_dir: str) -> list[dict[str, Any]]:
    """The SR's provenance sidecars (``*.srcutoutartifact.json``), newest first."""
    out = []
    for path in glob.glob(os.path.join(obj_dir, "*.srcutoutartifact.json")):
        try:
            with open(path) as f:
                record = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        git = record.get("git") if isinstance(record.get("git"), dict) else {}
        out.append({"id": record.get("id"), "created_at": record.get("created_at"),
                    "produced_by": record.get("produced_by"),
                    "git": git.get("short"), "dirty": git.get("dirty"),
                    "descriptors": record.get("descriptors"),
                    "file": os.path.basename(path)})
    out.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    return out


def _sr_header(obj_dir: str) -> dict[str, Any]:
    path = os.path.join(obj_dir, "SR.fits")
    try:
        header = fits.getheader(path)
    except (OSError, ValueError):
        return {}
    return {card: header[card] for card in _SR_CARDS if card in header}


def _object_card(run_dir: str, sub: str) -> dict[str, Any]:
    if not _SAFE_OBJECT.fullmatch(sub or ""):
        abort(400, description=f"bad object id {sub!r}")
    obj_dir = os.path.realpath(os.path.join(run_dir, sub))
    if os.path.dirname(obj_dir) != os.path.realpath(run_dir):
        abort(400, description=f"bad object id {sub!r}")
    rows = _read_manifest(run_dir)
    row = next((r for r in rows if str(r.get("out_subdir") or r.get("id")) == sub), None)
    if row is None and not os.path.isdir(obj_dir):
        abort(404, description=f"unknown evaluation object {sub!r}")
    identity = catalog_runner.current_eval_identity()
    enriched = _enrich_row(row or {"id": sub, "out_subdir": sub, "ok": "True"},
                           run_dir, identity)
    files = []
    if os.path.isdir(obj_dir):
        for name in sorted(os.listdir(obj_dir)):
            full = os.path.join(obj_dir, name)
            if os.path.isfile(full) and not name.startswith("."):
                stat = os.stat(full)
                files.append({"name": name, "bytes": stat.st_size, "mtime": stat.st_mtime})
    disagreement = None
    try:
        with open(os.path.join(obj_dir, "disagreement.json")) as f:
            disagreement = json.load(f)
    except (OSError, ValueError):
        pass
    rel = os.path.relpath(obj_dir, os.path.realpath(Config.EVAL_RESULTS_DIR))
    return {
        **enriched,
        "id": str((row or {}).get("id") or sub),
        "out_subdir": sub,
        "row": row,
        "members": catalog_runner.read_model_identity(obj_dir),
        "current": _identity_payload(identity),
        "disagreement": disagreement,
        "files": files,
        "provenance": _provenance(obj_dir),
        "sr_header": _sr_header(obj_dir),
        "downloads": {tier: f"/eval-files/{rel}/{name}" for tier, name in _DOWNLOADS.items()
                      if os.path.isfile(os.path.join(obj_dir, name))},
        "viewer": {"collection": "evaluation", "id": sub},
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
        abort(400, description=f"bad run name {run!r} (one directory under eval_results)")

    root = os.path.realpath(Config.EVAL_RESULTS_DIR)
    is_root = run in {"", "eval_results"}
    run_name = "eval_results" if is_root else run
    run_dir = root if is_root else os.path.realpath(os.path.join(root, run))
    if not is_root and os.path.dirname(run_dir) != root:
        abort(400, description=f"run {run!r} resolves outside eval_results")
    if not os.path.isdir(run_dir):
        if is_root and (required_file is None or allow_missing_root_file):
            return run_dir, run_name
        abort(404, description=f"no evaluation run {run_name!r}")
    if (required_file
            and not os.path.isfile(os.path.join(run_dir, required_file))
            and not (is_root and allow_missing_root_file)):
        abort(404, description=f"run {run_name!r} has no {required_file}")
    return run_dir, run_name


def register(app):
    errors.json_errors_for(app, "/api/evaluation/", "/eval-files/")

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

        def _run(cap):
            cap.write("model: STARFULL members through the production combiner "
                      "(member mean when no current combiner loads)\n")
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

    @app.route("/api/evaluation/objects/<object_id>")
    def api_evaluation_object(object_id: str):
        run_dir, _run_name = _resolve_run_dir(request.args.get("run"))
        return jsonify(_object_card(run_dir, object_id))

    @app.route("/api/evaluation/fetch-catalog", methods=["POST"])
    def api_evaluation_fetch_catalog():
        """Download + normalize the Euclid Q1 strong-lens catalog (Zenodo).

        Pulls the ~0.4 MB discovery CSV and writes the normalized
        ``lens_catalog/lenses.csv`` so the page is self-sufficient (no CLI
        step needed). Network failures surface as a 502 with the message.
        """
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
    @requires_fasrc
    def api_evaluation_sync():
        """Pull ``<data_dir>/eval_results`` down from FASRC into the gallery.

        The checkpoint auto-mirror (``fasrc_mirror``) only syncs the ckpt
        dir — eval-catalog runs land in ``<data_dir>/eval_results`` on the
        cluster and otherwise never reach the local ``/evaluation`` page.
        This is the one-shot rsync_pull, reusing the same ControlMaster
        transport the checkpoint mirror uses, so the user never has to drop
        to a terminal. ``--delete-after`` keeps local in lockstep with the
        remote (drops runs deleted on the cluster) without leaving a partial
        window mid-transfer — which also DELETES local results the cluster
        lacks, so the request must carry an explicit ``confirm=1``.
        """
        if str(request.form.get("confirm", "")).strip().lower() not in (
                "1", "true", "yes"):
            return jsonify({"ok": False, "code": "confirm_required", "error": (
                "syncing mirrors FASRC with rsync --delete-after and removes "
                "local results the cluster does not have; resend with confirm=1"
            )}), 400
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

        The viewer renders the FITS client-side; this only clears PNG caches
        a run may still carry (older gallery renders) — no cluster round-trip.
        """
        run = (request.form.get("run") or request.args.get("run") or "").strip()
        rd, _run_name = _resolve_run_dir(run)
        if not os.path.isdir(rd):
            abort(404, description="no evaluation results yet")
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

    @app.route("/api/evaluation/transformation", methods=["GET", "POST"])
    def api_evaluation_transformation():
        """Render + serve the run-level SR-transformation summary PNG.

        404 when the run has no ``manifest.csv``. Cached to
        ``<run>/transformation_summary.png``. ``POST`` re-renders (JSON
        ``{ok}``); a same-origin ``GET ?fresh=1`` still does (the SPA's
        button), a cross-site one gets the cached render — or 404 when
        nothing is rendered yet (a cross-site GET never renders).
        """
        run = (request.values.get("run") or "").strip()
        run_dir, _run_name = _resolve_run_dir(run, required_file="manifest.csv")
        out_png = os.path.join(run_dir, "transformation_summary.png")
        refuse_cross_site_cache_fill(out_png)
        fresh = request.method == "POST" or fresh_requested()
        if ((fresh or not os.path.isfile(out_png))
                and transformation_summary.render_transformation_summary(
                    run_dir, out_png) is None):
            abort(404, description="no synthetic objects with HR truth to summarise yet — "
                                   "run the grouped analysis with synthetic groups")
        if request.method == "POST":
            return jsonify({"ok": True, "rendered": True})
        return send_file(out_png, mimetype="image/png", max_age=0)

    @app.route("/api/evaluation/angular-power-spectrum", methods=["GET", "POST"])
    def api_evaluation_angular_power_spectrum():
        """Render + serve the per-band HR-vs-SR angular power-spectrum PNG.

        Per-band T(k) and r(k) (linear + asinh) over the **sky validation
        fields** synced through /sky (HR ``clean`` record vs generated SR cube).
        404 until the records are synced and SR has been generated. Cached to
        ``<eval_results>/angular_power_spectrum.png``. ``POST`` re-renders
        (JSON ``{ok}``); a same-origin ``GET ?fresh=1`` still does (the SPA's
        button), a cross-site one gets the cached render — or 404 when
        nothing is rendered yet (a cross-site GET never renders).
        """
        run = (request.values.get("run") or "").strip()
        run_dir, _run_name = _resolve_run_dir(run)
        if not os.path.isdir(run_dir):
            abort(404, description="no evaluation results yet")
        out_png = os.path.join(run_dir, "angular_power_spectrum.png")
        refuse_cross_site_cache_fill(out_png)
        fresh = request.method == "POST" or fresh_requested()
        if ((fresh or not os.path.isfile(out_png))
                and power_spectrum.render_power_spectrum_summary(
                    out_png, out_json=os.path.join(run_dir, _APS_JSON)) is None):
            abort(404, description="needs the synced validation records and their generated "
                                   "SR cube (Models › Images: Generate SR)")
        if request.method == "POST":
            return jsonify({"ok": True, "rendered": True})
        return send_file(out_png, mimetype="image/png", max_age=0)

    @app.get("/api/evaluation/angular-power-spectrum.json")
    def api_evaluation_angular_power_spectrum_json():
        """The per-band HR-vs-SR angular power spectrum as curves (Models ›
        Diagnostics › Recovery draws them): per band and space (linear,
        asinh) θ = 1/2k with the per-field median T(k) and r(k), their
        16–84% spread and the field count. Cache only — written next to the
        PNG whenever it renders (``POST /api/evaluation/angular-power-spectrum``);
        404 in words until then."""
        run = (request.values.get("run") or "").strip()
        run_dir, _run_name = _resolve_run_dir(run)
        path = os.path.join(run_dir, _APS_JSON)
        if not os.path.isfile(path):
            abort(404, description="the angular power spectrum is not measured yet — compute it "
                                   "(it needs the synced validation records and their generated SR)")
        return send_file(path, mimetype="application/json", max_age=0)

    @app.route("/eval-files/<path:relpath>")
    def serve_eval_files(relpath: str):
        """Download one per-object FITS from ``Config.EVAL_RESULTS_DIR``.

        Jailed against traversal (403 outside the results tree). Only
        ``.fits`` is served (as an attachment); the classic server-side PNG
        renderer this route used to carry is gone — the SPA browses results
        through the ``evaluation`` viewer collection.
        """
        root = os.path.realpath(Config.EVAL_RESULTS_DIR)
        full = os.path.realpath(os.path.join(root, relpath))
        if not full.startswith(root + os.sep):
            return jsonify({"ok": False, "error": "path outside eval_results"}), 403
        if not full.lower().endswith(".fits") or not os.path.isfile(full):
            return jsonify({"ok": False,
                            "error": f"no FITS file at eval_results/{relpath}"}), 404
        return send_file(full, mimetype="application/fits", as_attachment=True,
                         download_name=os.path.basename(full))
