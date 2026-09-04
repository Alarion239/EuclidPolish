"""Routes for the shared multipoint Euclid archive-field collection."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from flask import jsonify

from euclid_polish.web import fasrc_config, fasrc_fetcher
from euclid_polish.web.helpers import archive_fields
from euclid_polish.web.jobs import REGISTRY
from euclid_polish.web.remote import ensure_ssh_connected

_COMPLETE_STATUSES = frozenset({"written", "cached", "complete", "completed"})
_SYNC_LOCK = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return payload


def _fetch_required(remote_path: str, *, max_bytes: int, protected: set[str]) -> Path:
    result = fasrc_fetcher.fetch_one_file(
        remote_path,
        force=True,
        max_bytes=max_bytes,
        protect_paths=protected,
    )
    if not result.ok or not result.local_path:
        raise RuntimeError(result.error or f"failed to synchronize {remote_path}")
    protected.add(result.local_path)
    return Path(result.local_path)


def _remote_sample_path(sample: dict[str, Any], remote_root: str) -> str:
    sample_id = int(sample.get("sample_id", -1))
    expected_name = f"field_{sample_id:04d}.fits"
    expected_path = f"{remote_root.rstrip('/')}/cutouts/{expected_name}"
    remote_path = str(sample.get("output_path") or "").strip()
    if not remote_path:
        remote_path = expected_path
    if remote_path != expected_path:
        raise RuntimeError(
            f"archive sample {sample_id} path is outside the expected collection"
        )
    return remote_path


def _synced_source_payload(
    remote_payload: dict[str, Any],
    *,
    remote_manifest: str,
    remote_sha256: str,
) -> dict[str, Any]:
    if (
        remote_payload.get("kind") != "euclid_vis_noise_sampling"
        or remote_payload.get("version") != 1
    ):
        raise RuntimeError("remote VIS source manifest has an unsupported schema")
    payload = dict(remote_payload)
    payload["sync"] = {
        "remote_manifest": remote_manifest,
        "remote_manifest_sha256": remote_sha256,
    }
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _sync_archive_fields(cap) -> dict[str, Any]:
    cfg = fasrc_config.load()
    remote_sky_root = f"{cfg.data_dir.rstrip('/')}/euclid_sky"
    remote_root = f"{remote_sky_root}/{archive_fields.ARCHIVE_FIELDS_SUBDIR}"
    remote_manifest = f"{remote_root}/{archive_fields.ARCHIVE_FIELDS_MANIFEST}"
    remote_source_manifest = (
        f"{remote_sky_root}/vis_noise_samples/vis_noise_sampling_manifest.json"
    )

    cap.tick(0, archive_fields.SAMPLE_COUNT + 2, "connecting to FASRC")
    ensure_ssh_connected()
    protected: set[str] = set()
    archive_cache = _fetch_required(
        remote_manifest,
        max_bytes=32 * 1024 * 1024,
        protected=protected,
    )
    source_cache = _fetch_required(
        remote_source_manifest,
        max_bytes=32 * 1024 * 1024,
        protected=protected,
    )
    remote_archive = archive_fields.load_manifest(archive_cache)
    remote_source = _read_json(source_cache, "remote VIS source manifest")
    source_sha = _sha256(source_cache)
    if source_sha != str(remote_archive["source"]["manifest_sha256"]):
        raise RuntimeError(
            "archive collection was built from a different VIS-pointing manifest"
        )
    synced_source = _synced_source_payload(
        remote_source,
        remote_manifest=remote_source_manifest,
        remote_sha256=source_sha,
    )

    samples = list(remote_archive.get("samples") or [])
    if len(samples) != archive_fields.SAMPLE_COUNT:
        raise RuntimeError(
            f"archive collection has {len(samples)} samples; "
            f"expected {archive_fields.SAMPLE_COUNT}"
        )

    local_root = archive_fields.collection_root()
    local_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sync-", dir=local_root) as temporary:
        staging_root = Path(temporary)
        staging_cutouts = staging_root / "cutouts"
        staging_cutouts.mkdir()
        local_samples: list[dict[str, Any]] = []
        for index, raw_sample in enumerate(samples, start=1):
            if not isinstance(raw_sample, dict):
                raise RuntimeError(f"archive sample {index - 1} is invalid")
            sample = dict(raw_sample)
            sample_id = int(sample.get("sample_id", -1))
            if str(sample.get("status") or "").lower() not in _COMPLETE_STATUSES:
                raise RuntimeError(f"archive sample {sample_id} is not complete")
            remote_path = _remote_sample_path(sample, remote_root)
            cached = _fetch_required(
                remote_path,
                max_bytes=64 * 1024 * 1024,
                protected=protected,
            )
            expected_sha = str(sample.get("bundle_sha256") or "")
            if _sha256(cached) != expected_sha:
                raise RuntimeError(
                    f"archive sample {sample_id} failed its SHA256 check"
                )
            destination = staging_cutouts / f"field_{sample_id:04d}.fits"
            try:
                os.link(cached, destination)
            except OSError:
                shutil.copy2(cached, destination)
            sample["status"] = "cached"
            sample["output_path"] = str(
                (local_root / "cutouts" / destination.name).resolve()
            )
            sample["error"] = None
            local_samples.append(sample)
            cap.tick(index, archive_fields.SAMPLE_COUNT + 2, f"archive field {index}/220")

        local_payload = dict(remote_archive)
        local_payload["samples"] = local_samples
        local_payload["sync"] = {
            "remote_manifest": remote_manifest,
            "remote_manifest_sha256": _sha256(archive_cache),
            "source_manifest": remote_source_manifest,
            "source_manifest_sha256": source_sha,
            "completed_samples": len(local_samples),
        }
        if archive_fields.compute_collection_fingerprint(local_payload) != str(
            remote_archive["collection_fingerprint"]
        ):
            raise RuntimeError("local archive relocation changed collection identity")
        staged_manifest = staging_root / archive_fields.ARCHIVE_FIELDS_MANIFEST
        _write_json(staged_manifest, local_payload)
        archive_fields.load_manifest(staged_manifest)
        staged_source = staging_root / "vis_noise_sampling_manifest.json"
        _write_json(staged_source, synced_source)

        final_cutouts = local_root / "cutouts"
        final_cutouts.mkdir(exist_ok=True)
        for staged in sorted(staging_cutouts.iterdir()):
            os.replace(staged, final_cutouts / staged.name)
        source_target = archive_fields.source_manifest_path()
        source_target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_source, source_target)
        os.replace(staged_manifest, archive_fields.manifest_path())

    state = archive_fields.availability()
    if not state.get("ready"):
        reasons = "; ".join(str(item) for item in state.get("reasons") or [])
        raise RuntimeError(f"synchronized archive collection is not ready: {reasons}")
    cap.tick(
        archive_fields.SAMPLE_COUNT + 2,
        archive_fields.SAMPLE_COUNT + 2,
        "multipoint archive collection ready",
    )
    cap.write(
        f"synchronized {state['sample_count']} four-band fields from "
        f"{state['parent_count']} independent pointings\n"
    )
    return state


def register(app) -> None:
    @app.get("/api/archive-fields")
    def api_archive_fields():
        return jsonify(archive_fields.availability())

    @app.post("/api/archive-fields/sync")
    def api_archive_fields_sync():
        if not _SYNC_LOCK.acquire(blocking=False):
            return jsonify({"ok": False, "error": "archive sync is already running"}), 409

        def run(cap):
            try:
                return _sync_archive_fields(cap)
            finally:
                _SYNC_LOCK.release()

        try:
            job_id = REGISTRY.spawn(
                label="archive fields: synchronize multipoint collection",
                target=run,
            )
        except Exception:
            _SYNC_LOCK.release()
            raise
        return jsonify({"ok": True, "job_id": job_id})


__all__ = ["register"]
