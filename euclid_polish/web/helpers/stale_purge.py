"""Automatic purge of cached cubes no reader can reuse.

Jobs that make cubes stale request a purge
(:func:`euclid_polish.web.helpers.purge_requests.request_stale_purge`); the
job registry's finish hook (:func:`on_job_finished`) starts one job of kind
:data:`JOB_KIND` once no job runs, so a running Evaluate, fit or compare never
loses cubes it reads. The purge deletes exactly what each cache's next writer
would discard (:func:`~euclid_polish.eval.ensemble_cube_cache.purge_stale_bucket`,
:func:`~euclid_polish.web.helpers.real_field.purge_stale_real_fields`,
:func:`~euclid_polish.web.helpers.experiments.purge_stale_member_sr_cache`).
Spec: docs/superpowers/specs/2026-10-06-stale-cube-purge-design.md."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from euclid_polish.ensemble import member_fingerprints
from euclid_polish.ensemble_registry import default_ensemble_dir
from euclid_polish.eval.ensemble_cube_cache import (
    BLACKOUT_INDEX,
    VIZ_INDEX,
    is_label_keyed,
    purge_stale_bucket,
    read_bucket_manifest,
)
from euclid_polish.eval.subsets import eval_subset
from euclid_polish.web.helpers import ensemble_viz, experiments, real_field
from euclid_polish.web.helpers.purge_requests import clear_pending, read_pending
from euclid_polish.web.jobs import REGISTRY, Job, JobRegistry

JOB_KIND = "storage-purge-stale"
JOB_LABEL = "Storage: delete stale cubes"
#: ``(bucket dir, manifest name, records subset)``; ``"eval"`` is the
#: regime's evaluation subset (the blackout copies of the test cubes too).
_BUCKETS = (("cubes", VIZ_INDEX, "eval"),
            ("cubes_validate", VIZ_INDEX, "validate"),
            ("cubes_blackout", BLACKOUT_INDEX, "eval"),
            ("cubes_validate_blackout", BLACKOUT_INDEX, "validate"))

Progress = Callable[[int, int, str], None]


def _records_fps(records_dir: str | None, starless: bool) -> dict[str, str | None]:
    """The records fingerprints the bucket writers use now (``None``: unknown)."""
    if not records_dir:
        return {"eval": None, "validate": None}
    return {"eval": ensemble_viz._eval_records_fingerprint(
                records_dir, eval_subset(records_dir), starless=starless),
            "validate": ensemble_viz._eval_records_fingerprint(
                records_dir, "validate", starless=starless)}


def _test_adoption(starless: bool, cubes_dir: str) -> dict[str, str | None] | None:
    """The fingerprints a positional test bucket's migration would adopt."""
    manifest = read_bucket_manifest(cubes_dir)
    if manifest is None or is_label_keyed(manifest):
        return None
    return ensemble_viz._proven_test_fingerprints(starless, manifest)


def purge_stale_caches(progress: Progress | None = None) -> dict[str, Any]:
    """Delete every cached cube no reader can reuse; what went, per cache.

    A regime without active members is skipped (an unreadable ensemble dir
    must not read as "every member departed")."""
    base = default_ensemble_dir()
    records_dir = ensemble_viz._sky_records_local_dir()
    steps = 2 * len(_BUCKETS) + 2
    buckets: list[dict[str, Any]] = []
    for regime, starless in enumerate((False, True)):
        labels = ensemble_viz._regime_labels(base, starless)
        current = member_fingerprints(base, labels)
        fps = _records_fps(records_dir, starless)
        regime_dir = ensemble_viz._regime_dir_ro(starless)
        slug = ensemble_viz._regime_slug(starless)
        for position, (dir_name, manifest_name, subset) in enumerate(_BUCKETS):
            if progress is not None:
                progress(regime * len(_BUCKETS) + position, steps, f"{slug}/{dir_name}")
            if not labels:
                continue
            cubes_dir = os.path.join(regime_dir, dir_name)
            result = purge_stale_bucket(
                cubes_dir, name=manifest_name, current=current, records_fp=fps[subset],
                adopt=_test_adoption(starless, cubes_dir) if dir_name == "cubes" else None)
            if result.files_deleted or result.dropped or result.wiped:
                buckets.append({"bucket": f"{slug}/{dir_name}", "wiped": result.wiped,
                                "dropped": result.dropped,
                                "bytes_freed": result.bytes_freed,
                                "files_deleted": result.files_deleted})
    if progress is not None:
        progress(steps - 2, steps, "real fields")
    starfull = ensemble_viz._regime_labels(base, False)
    fields = (real_field.purge_stale_real_fields(member_fingerprints(base, starfull))
              if starfull else [])
    if progress is not None:
        progress(steps - 1, steps, "experiments member-SR cache")
    member_sr = (experiments.purge_stale_member_sr_cache(base) if starfull
                 else {"bytes_freed": 0, "files_deleted": 0, "members": []})
    if progress is not None:
        progress(steps, steps, "done")
    parts = [*buckets, *fields, member_sr]
    return {"buckets": buckets, "real_fields": fields, "member_sr_cache": member_sr,
            "bytes_freed": sum(int(p["bytes_freed"]) for p in parts),
            "files_deleted": sum(int(p["files_deleted"]) for p in parts)}


def _size(nbytes: int) -> str:
    return f"{nbytes / 1e9:.2f} GB" if nbytes >= 1e8 else f"{nbytes / 1e6:.1f} MB"


def job_stale_purge(cap) -> dict[str, Any]:
    """The purge as a job: log every deletion, then clear the request it served."""
    token = (read_pending() or {}).get("requested_at")
    report = purge_stale_caches(progress=cap.tick)
    for bucket in report["buckets"]:
        what = bucket["wiped"] or (
            f"dropped {', '.join(bucket['dropped'])}" if bucket["dropped"]
            else "leftover files")
        print(f"  {bucket['bucket']}: {what} — {bucket['files_deleted']} files, "
              f"{_size(bucket['bytes_freed'])}")
    for field in report["real_fields"]:
        print(f"  real field {field['field_id']}: dropped {', '.join(field['dropped'])} — "
              f"{field['files_deleted']} files, {_size(field['bytes_freed'])}")
    member_sr = report["member_sr_cache"]
    if member_sr["files_deleted"]:
        print(f"  experiments member-SR cache: {', '.join(member_sr['members'])} — "
              f"{member_sr['files_deleted']} files, {_size(member_sr['bytes_freed'])}")
    print(f"freed {_size(report['bytes_freed'])} in {report['files_deleted']} files"
          if report["files_deleted"] else "nothing stale")
    clear_pending(token)
    return report


def maybe_start_stale_purge(registry: JobRegistry | None = None) -> str | None:
    """Start the purge job when one is requested and no job runs; its id."""
    registry = registry or REGISTRY
    if read_pending() is None:
        return None
    if any(job.get("status") == "running" for job in registry.list(summary=True)):
        return None
    job, started = registry.spawn_exclusive(JOB_LABEL, job_stale_purge, kind=JOB_KIND)
    return job.job_id if started else None


def on_job_finished(job: Job, registry: JobRegistry | None = None) -> None:
    """Registry finish hook. Never reacts to the purge's own job, so a failing
    purge retries after the next other job instead of in a loop."""
    if job.kind != JOB_KIND:
        maybe_start_stale_purge(registry)
