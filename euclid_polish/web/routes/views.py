"""The ``data/vis`` gallery listing and the synthetic training records
(Synthetic › Records).

Records: ``GET /api/sky/sr-status`` (inventory + SR tier state),
``POST /api/sky/sync`` (FASRC pull as a background job),
``POST /api/sky/generate-sr`` (production SR over the local records, a job),
``GET /api/sky/records/sources`` / ``/api/sky/records/source`` (the truth
sources of a record from ``sources_<subset>.csv``). Everything but the sync
is local and works offline.
"""
from __future__ import annotations

import os
import posixpath
import threading
from typing import Any

import numpy as np
from flask import abort, jsonify, request

from euclid_polish.eval.catalog_runner import current_eval_identity, eval_model_identity
from euclid_polish.eval.ensemble_infer import load_eval_ensemble
from euclid_polish.image import ImageSet
from euclid_polish.image.tfio import tfrecord_path
from euclid_polish.web import fasrc_fetcher as _fasrc_fetcher
from euclid_polish.web.fasrc_gate import requires_fasrc
from euclid_polish.web.helpers import sky_records
from euclid_polish.web.helpers.paths import _sky_records_local_dir, _sky_records_remote_dir
from euclid_polish.web.helpers.status import (
    _list_vis_pngs,
)
from euclid_polish.web.jobs import REGISTRY as JOB_REGISTRY


def register(app):

    @app.route("/api/vis/list.json")
    def api_vis_list():
        """The `data/vis/` PNG gallery as JSON (newest first) — each entry has
        `rel` (served at /vis/<rel>) + an optional `inspect_fits` sibling. The
        React Visualization page's gallery reads this."""
        return jsonify({"pngs": _list_vis_pngs()})

    # ---------------- synthetic training records (Synthetic › Records) ---

    @app.route("/api/sky/sync", methods=["POST"])
    @requires_fasrc
    def api_sky_sync():
        """Rsync the synthetic TFRecord shards from FASRC into the local
        cache — as a background job (``kind="sky-sync"``, one at a time: a
        running sync's id is returned with ``already_running``).

        The held-out **test** split is the eval set and **validate** is what
        the ensemble combiner fits on, so both are pulled by default; the
        large ``train`` split is opt-in (``include_train=1`` or
        ``subsets=test,validate,train``). ``kinds`` narrows the files (comma
        list of ``dirty,hr,clean,sources``; default all). Missing shards (a
        dataset generated before a split) report not-ok and are skipped. The
        fetcher's 50 MB cap is lifted to 5 GB for this explicit transfer.
        Job result: ``{ok, files: {<kind>_<subset>: {ok, size_bytes,
        error?}}, subsets, include_train}``."""
        running = _running_job(SYNC_JOB_KIND)
        if running:
            return jsonify({"ok": True, "job_id": running, "already_running": True})
        try:
            subsets, kinds = _sync_selection(request.values)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        targets = sync_targets(_sky_records_remote_dir(), subsets, kinds)
        with _SPAWN_LOCK:
            running = _running_job(SYNC_JOB_KIND)
            if running:
                return jsonify({"ok": True, "job_id": running, "already_running": True})
            job_id = JOB_REGISTRY.spawn(
                f"records: sync {'+'.join(subsets)} from FASRC",
                lambda cap: _job_sky_sync(cap, targets, subsets), kind=SYNC_JOB_KIND)
        return jsonify({"ok": True, "job_id": job_id, "subsets": subsets, "files": list(targets)})

    @app.route("/api/sky/sr-status")
    def api_sky_sr_status():
        """State of Synthetic › Records (local, cheap: headers only).

        ``can_generate`` is true when dirty records and an active ensemble
        are present; ``sr`` counts SR cubes per split; ``splits`` has the
        per-split inventory (dirty/hr/clean TFRecords with record counts, the
        sources CSV) and the SR tier's ``state`` against the model an SR run
        would load now (``model``); ``sync_job`` / ``generate_job`` are the
        running jobs, if any."""
        records_dir = _sky_records_local_dir()
        subsets = sky_records.present_subsets(records_dir)
        ckpt = sky_records.checkpoint_present()
        inventory = sky_records.records_inventory(records_dir)
        identity = _current_identity()
        splits = {}
        for subset, entry in inventory.items():
            lr = tfrecord_path(records_dir, f"dirty_{subset}")
            splits[subset] = {**entry, "sr": sky_records.sr_state(
                subset, identity, lr if os.path.exists(lr) else None)}
        return jsonify({
            "records": bool(subsets),
            "checkpoint": ckpt,
            "can_generate": bool(subsets) and ckpt,
            "subsets": subsets,
            "sr": {s: sky_records.sr_count(s) for s in sky_records.SUBSETS},
            "records_dir": records_dir,
            "splits": splits,
            "model": identity,
            "sync_job": _running_job(SYNC_JOB_KIND),
            "generate_job": _running_job(GENERATE_JOB_KIND),
        })

    @app.route("/api/sky/generate-sr", methods=["POST"])
    def api_sky_generate_sr():
        """Run the production SR over the local dirty records (a job,
        ``kind="sky-generate-sr"``): STARFULL members through the production
        combiner (member mean when no current combiner loads).

        ``subsets`` (comma list; default every split with dirty records)
        picks the splits; a split that already has SR cubes is skipped unless
        ``overwrite=1``, which first deletes that split's cubes. Each split
        records the model identity + input records in ``sr_<split>.json`` so
        the SR tier's staleness can be shown. Refuses (400) without records
        or active members; a running SR job's id is returned with
        ``already_running``."""
        records_dir = _sky_records_local_dir()
        present = sky_records.present_subsets(records_dir)
        if not present:
            return jsonify({"ok": False, "error": "no sky records — sync them first"}), 400
        if not sky_records.checkpoint_present():
            return jsonify({"ok": False, "error": "no active ensemble members — train or "
                            "pull them on the /ensemble page"}), 400
        raw = str(request.values.get("subsets", "") or "").strip()
        wanted = [s.strip() for s in raw.split(",") if s.strip()] if raw else list(present)
        unknown = [s for s in wanted if s not in sky_records.SUBSETS]
        if unknown:
            return jsonify({"ok": False, "error": f"unknown split(s): {', '.join(unknown)}"}), 400
        missing = [s for s in wanted if s not in present]
        if missing:
            return jsonify({"ok": False, "error": f"no dirty records for: {', '.join(missing)}"}), 400
        overwrite = _flag(request.values, "overwrite")
        running = _running_job(GENERATE_JOB_KIND)
        if running:
            return jsonify({"ok": True, "job_id": running, "already_running": True})
        job_id = JOB_REGISTRY.spawn(
            f"records: generate SR ({'+'.join(wanted)}{', overwrite' if overwrite else ''})",
            lambda cap: _job_generate_sr(cap, records_dir, wanted, overwrite),
            kind=GENERATE_JOB_KIND)
        return jsonify({"ok": True, "job_id": job_id, "subsets": wanted, "overwrite": overwrite})

    @app.route("/api/sky/records/sources")
    def api_sky_record_sources():
        """Truth sources of the synthetic records (``sources_<subset>.csv``).

        ``?subset=&index=`` → one record: ``{subset, field_index, present,
        sources: [{row, type, render, x_pix, y_pix, off_field, flux_*_e,
        mag_vis, z, re_arcsec, theta_E_arcsec, …}], counts, geometry}``;
        without ``index`` → the split's per-record census ``{present,
        fields: [...], geometry}``. Positions are HR pixels (0-based, pixel
        centres at integers); ``geometry`` gives the HR and LR grids."""
        records_dir = _sky_records_local_dir()
        subset = _subset_arg()
        geometry = sky_records.split_geometry(records_dir, subset)
        raw = request.args.get("index")
        if raw in (None, ""):
            return jsonify({**sky_records.sources_summary(records_dir, subset),
                            "geometry": geometry})
        return jsonify({**sky_records.record_sources(records_dir, subset, _int_arg("index")),
                        "geometry": geometry})

    @app.route("/api/sky/records/source")
    def api_sky_record_source():
        """Every column of one truth source (``?subset=&index=&row=``; ``row``
        = its position among the record's sources). 404 when absent."""
        records_dir = _sky_records_local_dir()
        subset = _subset_arg()
        try:
            detail = sky_records.record_source_detail(
                records_dir, subset, _int_arg("index"), _int_arg("row"))
        except KeyError as exc:
            return jsonify({"ok": False, "error": str(exc).strip("'\"")}), 404
        return jsonify(detail)


# ---------------------------------------------------------------------------
# records helpers
# ---------------------------------------------------------------------------

SYNC_JOB_KIND = "sky-sync"
GENERATE_JOB_KIND = "sky-generate-sr"
SYNC_KINDS = ("dirty", "hr", "clean", "sources")
_SYNC_MAX_BYTES = 5 * 1024 * 1024 * 1024
_SPAWN_LOCK = threading.Lock()


def _flag(values, name: str) -> bool:
    return str(values.get(name, "false")).lower() in ("1", "true", "yes", "on")


def _subset_arg() -> str:
    subset = (request.args.get("subset") or "test").strip()
    if subset not in sky_records.SUBSETS:
        abort(400, description=f"subset must be {'|'.join(sky_records.SUBSETS)}")
    return subset


def _int_arg(name: str) -> int:
    try:
        value = int(request.args.get(name, ""))
    except (TypeError, ValueError):
        abort(400, description=f"{name} must be an integer")
    if value < 0:
        abort(400, description=f"{name} must be ≥ 0")
    return value


def _running_job(kind: str) -> str | None:
    for job in JOB_REGISTRY.list(summary=True):
        if job.get("kind") == kind and job.get("status") == "running":
            return str(job["job_id"])
    return None


def _current_identity() -> dict[str, Any] | None:
    """The model an SR run would load now (members + production combiner)."""
    try:
        return current_eval_identity()
    except Exception:  # noqa: BLE001 - no registry / unreadable combiner: unknown
        return None


def _sync_selection(values) -> tuple[list[str], list[str]]:
    raw = str(values.get("subsets", "") or "").strip()
    if raw:
        subsets = list(dict.fromkeys(s.strip() for s in raw.split(",") if s.strip()))
    else:
        subsets = ["test", "validate"] + (["train"] if _flag(values, "include_train") else [])
    unknown = [s for s in subsets if s not in sky_records.SUBSETS]
    if unknown or not subsets:
        raise ValueError(f"subsets must be a comma list of {'|'.join(sky_records.SUBSETS)}")
    raw_kinds = str(values.get("kinds", "") or "").strip()
    kinds = (list(dict.fromkeys(k.strip() for k in raw_kinds.split(",") if k.strip()))
             if raw_kinds else list(SYNC_KINDS))
    bad = [k for k in kinds if k not in SYNC_KINDS]
    if bad or not kinds:
        raise ValueError(f"kinds must be a comma list of {'|'.join(SYNC_KINDS)}")
    return subsets, kinds


def sync_targets(remote_dir: str, subsets: list[str], kinds: list[str]) -> dict[str, str]:
    """``<kind>_<subset> → remote path`` of one records sync (sources = CSV)."""
    targets: dict[str, str] = {}
    for kind in kinds:
        for subset in subsets:
            name = f"{kind}_{subset}"
            targets[name] = f"{remote_dir}/{name}.csv" if kind == "sources" else \
                f"{remote_dir}/{name}.tfrecord"
    return targets


def _job_sky_sync(cap, targets: dict[str, str], subsets: list[str]) -> dict[str, Any]:
    # A sky sync is one coherent dataset operation. Do not let the generic
    # cache's LRU evict an older test shard while the validation shards are
    # arriving; otherwise an immediate ensemble evaluation fails only after
    # restoring every checkpoint. The fetcher may evict unrelated cache
    # entries (such as bulky PSFs), but all requested records survive.
    protected = {_fasrc_fetcher._local_path_for(remote) for remote in targets.values()}
    results: dict[str, dict[str, Any]] = {}
    any_ok = False
    total = len(targets)
    for position, (key, remote) in enumerate(targets.items()):
        cap.tick(position, total, f"pulling {key}")
        r = _fasrc_fetcher.fetch_one_file(
            remote, force=True, max_bytes=_SYNC_MAX_BYTES, protect_paths=protected)
        entry: dict[str, Any] = {"ok": r.ok, "size_bytes": r.size_bytes}
        if r.ok:
            any_ok = True
            cap.write(f"{key}: {(r.size_bytes or 0) / 1e6:.1f} MB\n")
        else:
            entry["error"] = r.error
            cap.write(f"{key}: not pulled — {r.error}\n")
        results[key] = entry
    _pull_generation_sidecars(cap, targets, results)
    cap.tick(total, total, "done")
    if not any_ok:
        raise RuntimeError("nothing pulled — generate the records on FASRC first? "
                           + "; ".join(f"{k}: {v.get('error')}" for k, v in results.items())[:600])
    return {"ok": any_ok, "files": results, "subsets": subsets,
            "include_train": "train" in subsets}


def _pull_generation_sidecars(cap, targets: dict[str, str], results: dict[str, dict[str, Any]]) -> None:
    """Pull the provenance sidecar of every pulled record file (best-effort,
    small JSON): it records what the generation run used, such as the PSF of
    each band (Synthetic › PSF reads it)."""
    for key, remote in targets.items():
        if key.startswith("sources_") or not results.get(key, {}).get("ok"):
            continue
        artifact_id = sky_records.records_artifact_id(_fasrc_fetcher._local_path_for(remote))
        if artifact_id is None:
            continue
        sidecar = posixpath.join(posixpath.dirname(remote), sky_records.artifact_sidecar_name(artifact_id))
        r = _fasrc_fetcher.fetch_one_file(sidecar, force=True)
        results[key]["provenance"] = bool(r.ok)
        if not r.ok:
            cap.write(f"{key}: no provenance sidecar ({r.error})\n")


def _job_generate_sr(cap, records_dir: str, subsets: list[str], overwrite: bool) -> dict[str, Any]:
    def log(message: str) -> None:
        cap.write(message if message.endswith("\n") else message + "\n")

    # The production gate over only the STARFULL members it reads (the plain
    # mean of every STARFULL member, logged as a warning, when no current gate
    # loads) — never the mixed-regime plain mean.
    model = load_eval_ensemble(log=log)
    identity = eval_model_identity(model)
    os.makedirs(sky_records.sky_sr_dir(), exist_ok=True)
    generated: dict[str, int] = {}
    skipped: list[str] = []
    for subset in subsets:
        lr_path = tfrecord_path(records_dir, f"dirty_{subset}")
        if not os.path.exists(lr_path):
            skipped.append(subset)
            continue
        if sky_records.sr_count(subset) > 0:
            if not overwrite:
                log(f"{subset}: SR exists — skipped (overwrite to regenerate)")
                skipped.append(subset)
                continue
            log(f"{subset}: removed {sky_records.clear_sr(subset)} old SR cubes")
        lr = ImageSet.read(lr_path)
        sr_set = model.upsample_batch(
            lr, on_progress=lambda i, t, label, _s=subset: cap.tick(i, t, f"{_s}: {label}"),
            log=log)
        for i, sr in enumerate(sr_set):
            np.save(sky_records.sr_path(subset, i), sr.data)
        sky_records.write_sr_manifest(subset, identity, lr_path, count=len(sr_set),
                                      model_label=model.label)
        generated[subset] = len(sr_set)
    return {"generated": generated, "skipped": skipped, "model": model.label,
            "identity": identity}
