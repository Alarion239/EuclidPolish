"""The study freeze: validate → create (incomplete) → numbers → fields →
complete → mirror.

:func:`start` validates a freeze request against the live candidates (at
most :data:`~euclid_polish.studies.store.MAX_FIELDS` fields, each available,
FASRC connected when fields are attached, the disk margin kept) and creates
the incomplete study with its plan; nothing is written when it refuses.
:func:`run` (the ``study-freeze`` job, also ``resume``) then

1. writes the numbers files locally (once; a resume keeps them) — refused
   when the members, their checkpoints, the test records or the gate changed
   since the study started (:func:`numbers.live_identity`);
2. packs every pending field one product at a time into a temp file,
   uploads it to ``<holylabs>/study_fields/<id>/<field>/``, checks the remote
   sha256 and deletes the temp file, recording each field in the manifest;
3. seals the study (``complete``) and mirrors its directory to
   ``<remote tracking dir>/studies/<id>/`` (best-effort, no ``--delete``).

A failure records ``error`` in the (still incomplete) manifest: the list
shows the study as "incomplete — resume or delete".
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from euclid_polish.provenance.gitinfo import capture_git
from euclid_polish.studies import candidates, fields, numbers
from euclid_polish.studies import store as study_store
from euclid_polish.studies.cache import reserved_bytes, staging_root
from euclid_polish.studies.store import MAX_FIELDS, StudyError, StudyStore
from euclid_polish.web.helpers import experiments
from euclid_polish.web.jobs import JobCancelled

Progress = Callable[[int, int, str], None]
#: Progress units of the numbers phase (each uploaded product is one more).
NUMBERS_UNITS = 100


def plan_fields(payload: dict[str, Any], fids: Sequence[str]) -> list[dict[str, Any]]:
    """The requested candidate fields (deduplicated, request order);
    :class:`StudyError` for too many, unknown or unavailable ones."""
    wanted = list(dict.fromkeys(str(f).strip() for f in fids if str(f).strip()))
    if len(wanted) > MAX_FIELDS:
        raise StudyError(400, f"a study holds at most {MAX_FIELDS} fields ({len(wanted)} asked)")
    by_fid = {f["fid"]: f for f in payload["fields"]}
    unknown = [fid for fid in wanted if fid not in by_fid]
    if unknown:
        raise StudyError(400, "unknown field(s) for this setup: " + ", ".join(unknown))
    refused = [f"{fid}: {by_fid[fid]['reason']}" for fid in wanted
               if not by_fid[fid]["available"]]
    if refused:
        raise StudyError(409, "field(s) not available — " + "; ".join(refused))
    return [by_fid[fid] for fid in wanted]


def check_disk(store: StudyStore, needed: int) -> None:
    free = experiments.free_bytes(Path(store.root)) - reserved_bytes()
    if free - int(needed) < experiments.MIN_FREE_BYTES:
        raise experiments.DiskSpaceError(
            f"not enough disk space to freeze: {needed / 1024 ** 2:.0f} MB are written locally "
            f"(numbers + one product at a time), {free / 1024 ** 3:.2f} GiB is free and "
            f"{experiments.MIN_FREE_BYTES / 1024 ** 3:.0f} GiB must stay free",
            needed=int(needed), free=free)


def start(store: StudyStore, *, name: str, note: str, starless: bool, fids: Sequence[str],
          connected: bool) -> str:
    """Validate a freeze and create its incomplete study; returns the id."""
    if not str(name or "").strip():
        raise StudyError(400, "a study needs a name")
    payload = candidates.candidates(starless)
    if not payload["can_freeze"]:
        raise StudyError(409, payload["blocking"] or "nothing to freeze")
    planned = plan_fields(payload, fids)
    if planned and not connected:
        raise StudyError(503, "FASRC not connected: fields are stored on holylabs — "
                              "connect, or freeze without fields")
    largest = max((int(f.get("largest_product_bytes") or 0) for f in planned), default=0)
    check_disk(store, int(payload["ensemble"]["numbers_bytes"]) + 2 * largest)
    return store.create(name, note, regime=candidates.regime_slug(starless), extra={
        "commit": capture_git(),
        "identity": numbers.live_identity(starless),
        "fields": [{"fid": f["fid"], "kind": f["kind"], "ref": f["ref"], "label": f["label"],
                    "state": "pending", "estimated_bytes": f["bytes"]} for f in planned],
        "numbers_complete": False,
    })


def _connected(ssh: Any) -> bool:
    return ssh is not None and bool(getattr(ssh, "is_connected", lambda: False)())


def run(store: StudyStore, study_id: str, *, ssh: Any, remote_tracking_dir: str,
        progress: Progress | None = None, check: Callable[[], None] | None = None,
        log: Callable[[str], None] = print) -> dict[str, Any]:
    """Freeze (or resume) ``study_id``; returns the job result."""
    tick = progress or (lambda *_a: None)
    manifest = store.manifest(study_id)
    if manifest.get("complete"):
        return {"study_id": study_id, "complete": True, "already_complete": True}
    starless = manifest.get("regime") == "starless"
    try:
        live = numbers.live_identity(starless)
        if manifest.get("identity") and manifest["identity"] != live:
            changed = [k for k in live if live.get(k) != manifest["identity"].get(k)]
            raise StudyError(409, (
                f"the ensemble changed since this study started ({', '.join(changed)}); "
                "delete it and freeze again"))
        remote = {"field_root": study_store.remote_field_dir(remote_tracking_dir, study_id),
                  "numbers_dir": study_store.remote_numbers_dir(remote_tracking_dir, study_id)}
        manifest = store.update_manifest(study_id, remote=remote, error=None)
        pending = [f for f in manifest.get("fields") or [] if f.get("state") != "uploaded"]
        n_products = sum(len(fields.product_names(f["kind"], live["labels"])) + 2
                         for f in pending)
        # The numbers take the first NUMBERS_UNITS of the bar, each product one unit.
        total = NUMBERS_UNITS + n_products
        done = 0
        if not manifest.get("numbers_complete"):
            log(f"study {study_id}: computing the numbers")
            bundle = numbers.build(starless, check=check, progress=lambda i, n, label: tick(
                int(NUMBERS_UNITS * i / max(1, n)), total, f"numbers · {label}"))
            if bundle.snapshot["identity"] != manifest.get("identity", bundle.snapshot["identity"]):
                raise StudyError(409, "the ensemble changed while the numbers were computed")
            for name, data in bundle.files.items():
                store.write_number(study_id, name, data)
            snapshot = {k: v for k, v in bundle.snapshot.items() if k != "identity"}
            manifest = store.update_manifest(study_id, **snapshot, numbers_complete=True)
            log(f"study {study_id}: numbers written "
                f"({sum(len(v) for v in bundle.files.values()) / 1024 ** 2:.1f} MB)")
        done = NUMBERS_UNITS
        tick(done, total, "numbers written")
        if pending and not _connected(ssh):
            raise StudyError(503, "FASRC not connected: the study's fields cannot be uploaded")
        labels = [m["label"] for m in (manifest.get("ensemble") or {}).get("members") or []]
        for entry in pending:
            fid = entry["fid"]
            if check is not None:
                check()
            log(f"study {study_id}: packing {fid}")
            source = fields.open_field(fid, starless=starless, labels=labels)

            def on_product(name: str, _k: int, _n: int, _fid: str = fid) -> None:
                nonlocal done
                done += 1
                tick(min(done, total), total, f"{_fid} · {name}")

            record = fields.pack_field(ssh, source, f"{remote['field_root']}/{fid}",
                                       staging=staging_root(), on_product=on_product,
                                       check=check)
            # Record the upload first: a resume must never re-upload this field.
            manifest = store.manifest(study_id)
            manifest["fields"] = [({**f, **record} if f["fid"] == fid else f)
                                  for f in manifest.get("fields") or []]
            store.write_manifest(study_id, manifest)
            if source.thumbnail_cube is not None:
                store.write_number(study_id, f"thumbs/{fid}.jpg",
                                   candidates.render_thumbnail(source.thumbnail_cube))
            log(f"study {study_id}: {fid} uploaded ({record['bytes'] / 1024 ** 2:.1f} MB)")
        store.mark_complete(study_id)
    except (JobCancelled, KeyboardInterrupt) as exc:
        _record_error(store, study_id, f"cancelled ({type(exc).__name__})")
        raise
    except Exception as exc:
        _record_error(store, study_id, str(exc) or type(exc).__name__)
        raise
    mirror = mirror_numbers(store, study_id, ssh, remote_tracking_dir)
    log(f"study {study_id}: complete; mirror: "
        f"{'ok' if mirror['ok'] else mirror.get('error')}")
    final = store.manifest(study_id)
    return {"study_id": study_id, "complete": True, "fields": len(final.get("fields") or []),
            "fields_bytes": sum(int(f.get("bytes") or 0) for f in final.get("fields") or []),
            "numbers_bytes": sum(int(v.get("bytes") or 0)
                                 for v in (final.get("numbers") or {}).values()),
            "mirror": mirror}


def _record_error(store: StudyStore, study_id: str, message: str) -> None:
    with contextlib.suppress(StudyError):
        store.update_manifest(study_id, error=message[:1000])


def mirror_numbers(store: StudyStore, study_id: str, ssh: Any,
                   remote_tracking_dir: str) -> dict[str, Any]:
    """Best-effort ``rsync`` of the study directory to holylabs (no delete)."""
    target = study_store.remote_numbers_dir(remote_tracking_dir, study_id)
    if not _connected(ssh):
        return {"ok": False, "error": "not connected to FASRC (the next tracking sync "
                                      "mirrors it)", "remote_dir": target}
    try:
        rc, _out, err = ssh.rsync_push(str(store.path(study_id)), target, timeout=900)
    except Exception as exc:  # noqa: BLE001 — the mirror is best-effort
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "remote_dir": target}
    if rc != 0:
        return {"ok": False, "error": f"rsync exit {rc}: {str(err).strip()[:300]}",
                "remote_dir": target}
    return {"ok": True, "remote_dir": target}


__all__ = ["NUMBERS_UNITS", "check_disk", "mirror_numbers", "plan_fields", "run", "start"]
