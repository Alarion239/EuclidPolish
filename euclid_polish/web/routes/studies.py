"""Model studies (Figures › Studies): frozen, whole-ensemble comparisons.

* ``GET  /api/studies``                                — every study (complete + incomplete).
* ``GET  /api/studies/candidates?mode=``               — what a freeze would capture now (read-only).
* ``GET  /api/studies/candidates/thumb/<fid>.jpg``     — a candidate field's preview.
* ``POST /api/studies``                                — freeze (local job ``study-freeze``).
* ``GET  /api/studies/<id>``                           — manifest + numbers (read-only).
* ``POST /api/studies/<id>/resume``                    — continue an incomplete study.
* ``POST /api/studies/<id>/note`` · ``/selections``    — the two mutable sidecars.
* ``POST /api/studies/<id>/delete``                    — delete (``confirm=1``), with its holylabs copies.
* ``POST /api/studies/<id>/fields/<fid>/fetch``        — fetch one field (job ``study-fetch``, FASRC).
* ``GET  /api/studies/<id>/figure/<chart>[.csv]``      — a publication figure or its CSV.
* ``GET  /api/studies/<id>/numbers/<path>``            — one frozen numbers file.

Opening or reading never starts a job and never fetches a field; freezing,
resuming, fetching and deleting are explicit POSTs. The numbers never need
FASRC; attaching and fetching fields do (fetch is ``@requires_fasrc``; a
freeze with fields answers 503 ``fasrc_offline`` while disconnected).
"""
from __future__ import annotations

import json
from io import BytesIO
from typing import Any

from flask import jsonify, request, send_file

from euclid_polish.studies import candidates as study_candidates
from euclid_polish.studies import fields as study_fields
from euclid_polish.studies import freeze as study_freeze
from euclid_polish.studies import render as study_render
from euclid_polish.studies import stats as study_stats
from euclid_polish.studies import store as study_store
from euclid_polish.studies.cache import FieldCache
from euclid_polish.studies.store import StudyError, StudyStore
from euclid_polish.tracking import sync as tracking_sync
from euclid_polish.web import errors, fasrc_config
from euclid_polish.web.fasrc_gate import FASRC_OFFLINE_PAYLOAD, fasrc_connected, requires_fasrc
from euclid_polish.web.helpers import experiments
from euclid_polish.web.jobs import REGISTRY, start_exclusive
from euclid_polish.web.remote import STATE

FREEZE_KIND = "study-freeze"
FETCH_KIND = "study-fetch"
_MEDIA = {"csv": "text/csv", "json": "application/json", "jpg": "image/jpeg"}


def _fail(message: str, status: int = 400, **extra):
    return jsonify({"ok": False, "error": message, **extra}), status


def _study_error(exc: StudyError):
    if exc.code == 503:
        return jsonify({**FASRC_OFFLINE_PAYLOAD, "error": str(exc)}), 503
    return _fail(str(exc), exc.code)


def _disk_error(exc: experiments.DiskSpaceError):
    return _fail(str(exc), 507, code="insufficient_storage",
                 needed_bytes=exc.needed, free_bytes=exc.free)


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _form() -> dict[str, Any]:
    if request.is_json:
        body = request.get_json(silent=True)
        return body if isinstance(body, dict) else {}
    return request.form.to_dict(flat=True)


def _starless(value: Any) -> bool:
    mode = str(value or "starfull").strip().lower()
    if mode not in ("starfull", "starless"):
        raise StudyError(400, f"mode must be starfull or starless, got {mode!r}")
    return mode == "starless"


def _split(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value or "").split(",") if v.strip()]


def _remote_tracking_dir() -> str:
    cfg = fasrc_config.load()
    return tracking_sync.remote_tracking_dir(cfg.repo_path, cfg.tracking_remote_dir)


def _selection(store: StudyStore, study_id: str) -> study_render.Selection:
    """The chart selection from the query (``selection=<name>`` loads a
    saved one; explicit ``members`` / ``group`` override it)."""
    members: list[str] | None = None
    group: str | None = None
    name = (request.args.get("selection") or "").strip()
    if name:
        saved = next((s for s in store.selections(study_id) if s.get("name") == name), None)
        if saved is None:
            raise StudyError(404, f"no saved selection {name!r}")
        members, group = saved.get("members"), saved.get("group")
    if request.args.get("members"):
        members = _split(request.args.get("members"))
    if "group" in request.args:
        group = (request.args.get("group") or "").strip() or None
    return study_render.Selection(
        members=members, group=group,
        reference=(request.args.get("reference") or "mean").strip(),
        source=(request.args.get("source") or "all").strip(),
        metric=(request.args.get("metric") or "psnr").strip(),
        experiment=(request.args.get("experiment") or "").strip() or None,
        seed=errors.int_arg("seed", 0, lo=0, hi=2 ** 31 - 1),
        resamples=errors.int_arg("resamples", study_stats.BOOTSTRAP_RESAMPLES,
                                 lo=100, hi=20000))


def _fields_view(store: StudyStore, study_id: str, manifest: dict[str, Any]) -> list[dict]:
    cache = FieldCache()
    out = []
    for record in manifest.get("fields") or []:
        fid = record.get("fid")
        uploaded = record.get("state") == "uploaded"
        thumb = f"thumbs/{fid}.jpg" in (manifest.get("numbers") or {})
        products = record.get("products") or {}
        core = study_fields.core_products(record)
        members = study_fields.member_products(record)
        cached = cache.cached_products(study_id, fid, products) if uploaded else set()
        out.append({k: record.get(k) for k in ("fid", "kind", "ref", "label", "state", "bytes",
                                                "estimated_bytes", "gate")}
                   | {"fetched": uploaded and bool(core) and set(core) <= cached,
                      "core_bytes": sum(int(products[n].get("bytes") or 0) for n in core),
                      "member_bytes": {n: int(products[n].get("bytes") or 0) for n in members},
                      "cached_products": sorted(cached),
                      "members_fetched": sum(1 for n in members if n in cached),
                      "products": sorted(products),
                      "thumb_url": (f"/api/studies/{study_id}/numbers/thumbs/{fid}.jpg"
                                    if thumb else None),
                      "viewer": {"collection": "study", "params": {"study": study_id},
                                 "id": fid}})
    return out


def register(app):
    errors.json_errors_for(app, "/api/studies")

    @app.get("/api/studies")
    def api_studies():
        store = StudyStore()
        running = REGISTRY.running(FREEZE_KIND)
        return jsonify({"ok": True, "studies": store.list(), "root": store.root,
                        "max_fields": study_store.MAX_FIELDS,
                        "freezing": ({"job_id": running.job_id, "study_id": running.key}
                                     if running is not None else None)})

    @app.get("/api/studies/candidates")
    def api_studies_candidates():
        try:
            payload = study_candidates.candidates(_starless(request.args.get("mode")))
        except StudyError as exc:
            return _study_error(exc)
        return jsonify({"ok": True, **payload})

    @app.get("/api/studies/candidates/thumb/<fid>.jpg")
    def api_studies_candidate_thumb(fid: str):
        try:
            starless = _starless(request.args.get("mode"))
            size = errors.int_arg("size", 160, lo=32, hi=480, clamp=True)
            body = study_candidates.thumbnail(fid, starless, size)
        except StudyError as exc:
            return _study_error(exc)
        except (ValueError, KeyError) as exc:
            return _fail(str(exc), 400)
        except (FileNotFoundError, OSError) as exc:
            return _fail(str(exc), 404)
        return send_file(BytesIO(body), mimetype="image/jpeg", max_age=0)

    @app.post("/api/studies")
    def api_studies_freeze():
        form = _form()
        store = StudyStore()
        running = REGISTRY.running(FREEZE_KIND)
        if running is not None:
            return _fail(f"busy: another study is being frozen (job {running.job_id}); "
                         "try again when it finishes", 409, code="busy", job_id=running.job_id)
        try:
            starless = _starless(form.get("mode"))
            fids = _split(form.get("fields"))
            study_id = study_freeze.start(
                store, name=str(form.get("name") or ""), note=str(form.get("note") or ""),
                starless=starless, fids=fids, connected=fasrc_connected())
        except StudyError as exc:
            return _study_error(exc)
        except experiments.DiskSpaceError as exc:
            return _disk_error(exc)
        return _spawn_freeze(store, study_id, "freeze")

    def _spawn_freeze(store: StudyStore, study_id: str, verb: str):
        remote_dir = _remote_tracking_dir()
        manifest = store.manifest(study_id)

        def target(cap):
            return study_freeze.run(store, study_id, ssh=STATE.ssh,
                                    remote_tracking_dir=remote_dir, progress=cap.tick,
                                    check=cap.check_cancelled)

        payload, status = start_exclusive(
            f"study {verb}: {manifest.get('name')}", target, kind=FREEZE_KIND, key=study_id,
            busy="another study is being frozen")
        if verb == "freeze" and not payload.get("ok"):
            # Lost the race to another freeze: the study just created never ran.
            store.delete(study_id, None, remote_tracking_dir=remote_dir, local_only=True)
            return jsonify({**payload, "study_id": None}), status
        fields = manifest.get("fields") or []
        return jsonify({**payload, "study_id": study_id, "fields": [f["fid"] for f in fields],
                        "upload_bytes": sum(int(f.get("estimated_bytes") or 0)
                                            for f in fields)}), status

    @app.get("/api/studies/<study_id>")
    def api_study(study_id: str):
        store = StudyStore()
        try:
            manifest = store.manifest(study_id)
            sha = store.manifest_sha256(study_id)
            out: dict[str, Any] = {
                "ok": True, "study": store.summary(manifest), "manifest": manifest,
                "manifest_sha256": sha,
                "note": store.note(study_id), "selections": store.selections(study_id),
                "fields": _fields_view(store, study_id, manifest),
                "charts": list(study_render.CHARTS),
                "group_fields": list(study_render.GROUP_FIELDS),
                "citation": (f"model study {study_id} “{manifest.get('name')}” "
                             f"(study.json sha256 {sha[:16]})"),
            }
            if manifest.get("complete") and request.args.get("numbers", "1") != "0":
                out["numbers"] = {name.removesuffix(".json"): store.read_json(study_id, name)
                                  for name in ("knee_psnr.json", "training_curves.json",
                                               "gate.json", "real.json")}
        except StudyError as exc:
            return _study_error(exc)
        return jsonify(out)

    @app.post("/api/studies/<study_id>/resume")
    def api_study_resume(study_id: str):
        store = StudyStore()
        try:
            manifest = store.manifest(study_id)
            if manifest.get("complete"):
                return _fail(f"study {study_id} is already complete", 409)
            pending = [f for f in manifest.get("fields") or [] if f.get("state") != "uploaded"]
            if pending and not fasrc_connected():
                raise StudyError(503, "FASRC not connected: the study's fields cannot be "
                                      "uploaded — connect first")
        except StudyError as exc:
            return _study_error(exc)
        return _spawn_freeze(store, study_id, "resume")

    @app.post("/api/studies/<study_id>/note")
    def api_study_note(study_id: str):
        try:
            note = StudyStore().set_note(study_id, str(_form().get("note") or ""))
        except StudyError as exc:
            return _study_error(exc)
        return jsonify({"ok": True, "id": study_id, "note": note})

    @app.post("/api/studies/<study_id>/selections")
    def api_study_selections(study_id: str):
        form = _form()
        raw = form.get("selections")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return _fail("selections must be a JSON list", 400)
        if not isinstance(raw, list):
            return _fail("selections must be a list of {name, members?, group?, note?}", 400)
        store = StudyStore()
        try:
            labels = [str(m.get("label")) for m in
                      (store.manifest(study_id).get("ensemble") or {}).get("members") or []]
            saved = store.set_selections(study_id, raw, members=labels,
                                         groups=study_stats.GROUP_FIELDS)
        except StudyError as exc:
            return _study_error(exc)
        return jsonify({"ok": True, "id": study_id, "selections": saved})

    @app.post("/api/studies/<study_id>/delete")
    def api_study_delete(study_id: str):
        form = _form()
        if not _truthy(form.get("confirm")):
            return _fail("deleting a study also deletes its holylabs field store; "
                         "resend with confirm=1", 400, code="confirm_required")
        running = REGISTRY.running(FREEZE_KIND, key=study_id)
        if running is not None:
            return _fail(f"study {study_id} is being frozen (job {running.job_id}); cancel it "
                         "first", 409, code="busy", job_id=running.job_id)
        fetching = REGISTRY.running(FETCH_KIND)
        if fetching is not None and str(fetching.key or "").startswith(f"{study_id}/"):
            return _fail(f"a field of study {study_id} is being fetched (job "
                         f"{fetching.job_id}); cancel it first", 409, code="busy",
                         job_id=fetching.job_id)
        store = StudyStore()
        try:
            out = store.delete(study_id, STATE.ssh, remote_tracking_dir=_remote_tracking_dir(),
                               local_only=_truthy(form.get("local_only")))
        except StudyError as exc:
            return _study_error(exc)
        FieldCache().remove_study(study_id)
        return jsonify(out)

    @app.post("/api/studies/<study_id>/fields/<fid>/fetch")
    @requires_fasrc
    def api_study_field_fetch(study_id: str, fid: str):
        store = StudyStore()
        try:
            manifest = store.manifest(study_id)
            record = next((f for f in manifest.get("fields") or [] if f.get("fid") == fid), None)
            if record is None or record.get("state") != "uploaded":
                raise StudyError(404, f"study {study_id} has no uploaded field {fid!r}")
        except StudyError as exc:
            return _study_error(exc)
        remote_dir = str(record.get("remote_dir") or "")
        if not remote_dir:
            remote_dir = study_store.remote_field_dir(_remote_tracking_dir(), study_id) + f"/{fid}"
        raw = str(_form().get("products") or "core").strip()
        request_spec: str | list[str] = raw if raw in ("core", "members") else _split(raw)
        try:
            names, _fit = study_fields.fetch_request(record, request_spec)
        except ValueError as exc:
            return _fail(str(exc), 400)
        products = record.get("products") or {}

        def target(cap):
            result = study_fields.fetch_field(
                STATE.ssh, study_id, record, remote_dir, FieldCache(), products=request_spec,
                check=cap.check_cancelled, progress=cap.tick)
            return {"study_id": study_id, **result}

        # One fetch at a time (the cache budget and the disk margin are shared);
        # the same request re-attaches, another one is busy.
        payload, status = start_exclusive(
            f"study {study_id}: fetch {fid} ({raw})", target, kind=FETCH_KIND,
            key=f"{study_id}/{fid}:{raw}", busy="another study field is being fetched")
        return jsonify({**payload, "study_id": study_id, "fid": fid, "products": names,
                        "bytes": sum(int(products[n].get("bytes") or 0) for n in names)}), status

    @app.get("/api/studies/<study_id>/figure/<name>")
    def api_study_figure(study_id: str, name: str):
        store = StudyStore()
        chart, _, suffix = name.partition(".")
        fmt = (suffix or request.args.get("format") or "png").lower()
        try:
            data = study_render.StudyData.load(store, study_id)
            selection = _selection(store, study_id)
            filename = f"study-{study_id}-{chart}.{fmt}"
            if fmt == "csv":
                columns, rows = study_render.table(chart, data, selection)
                body = study_render.csv_text(columns, rows).encode("utf-8")
                mimetype = "text/csv"
            else:
                dpi = errors.int_arg("dpi", 300, lo=120, hi=600, clamp=True)
                body = study_render.render(chart, data, selection, output_format=fmt, dpi=dpi)
                mimetype = study_render.FORMATS[fmt]
                filename = f"study-{study_id}-{chart}-{dpi}dpi.{fmt}"
        except StudyError as exc:
            return _study_error(exc)
        return send_file(BytesIO(body), mimetype=mimetype, max_age=0,
                         as_attachment=_truthy(request.args.get("download")),
                         download_name=filename)

    @app.get("/api/studies/<study_id>/numbers/<path:name>")
    def api_study_numbers(study_id: str, name: str):
        store = StudyStore()
        try:
            body = store.read_number(study_id, name)
        except StudyError as exc:
            return _study_error(exc)
        media = _MEDIA.get(name.rsplit(".", 1)[-1].lower(), "application/octet-stream")
        return send_file(BytesIO(body), mimetype=media, max_age=0,
                         as_attachment=_truthy(request.args.get("download")),
                         download_name=f"study-{study_id}-{name.replace('/', '-')}")
