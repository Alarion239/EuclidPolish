"""Focused route contracts for synchronising multipoint archive fields."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from flask import Flask

from euclid_polish.web import remote
from euclid_polish.web.helpers import archive_fields as archive_provider
from euclid_polish.web.routes import archive_fields as routes


class _Capture:
    def __init__(self) -> None:
        self.progress: list[tuple[int, int, str]] = []
        self.messages: list[str] = []

    def tick(self, current: int, total: int, label: str) -> None:
        self.progress.append((current, total, label))

    def write(self, message: str) -> None:
        self.messages.append(message)


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _route_app() -> Flask:
    app = Flask(__name__)
    app.config.update(TESTING=True)
    routes.register(app)
    return app


def _remote_fixture(tmp_path: Path, *, bundle_sha: str | None = None) -> dict[str, Any]:
    """Create a one-sample remote snapshot suitable for route-only tests."""
    remote_source = tmp_path / "remote-source.json"
    remote_source.write_bytes(_json_bytes({
        "kind": "euclid_vis_noise_sampling",
        "version": 1,
        "source_release": "Q1_R1",
        "plan_fingerprint": "a" * 64,
        "samples": [],
    }))
    remote_bundle = tmp_path / "remote-field.fits"
    remote_bundle.write_bytes(b"four-band-fits-snapshot")
    expected_bundle_sha = bundle_sha or _sha256(remote_bundle)
    remote_root = "/remote/data/euclid_sky/archive_fields"
    sample = {
        "sample_id": 0,
        "status": "written",
        "error": None,
        "output_path": f"{remote_root}/cutouts/field_0000.fits",
        "bundle_sha256": expected_bundle_sha,
        "parent_id": "parent-000",
    }
    archive_manifest: dict[str, Any] = {
        "version": 1,
        "kind": "euclid_archive_fields",
        "source_release": "Q1_R1",
        "source": {
            "manifest_sha256": _sha256(remote_source),
            "plan_fingerprint": "a" * 64,
        },
        "plan": {"one_sample_test": True},
        "plan_fingerprint": "b" * 64,
        "samples": [sample],
    }
    archive_manifest["collection_fingerprint"] = (
        archive_provider.compute_collection_fingerprint(archive_manifest)
    )
    remote_archive = tmp_path / "remote-archive.json"
    remote_archive.write_bytes(_json_bytes(archive_manifest))
    return {
        "archive": remote_archive,
        "source": remote_source,
        "bundle": remote_bundle,
        "payload": archive_manifest,
        "remote_root": remote_root,
    }


def _install_sync_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    remote_files: dict[str, Any],
) -> tuple[Path, Path, list[tuple[str, dict[str, Any]]]]:
    """Redirect a one-sample sync to tmp_path and return its observed fetches."""
    local_root = tmp_path / "local" / "archive_fields"
    local_source = (
        tmp_path / "local" / "vis_noise_samples" / "vis_noise_sampling_manifest.json"
    )
    remote_root = str(remote_files["remote_root"])
    remote_paths = {
        f"{remote_root}/archive_fields_manifest.json": remote_files["archive"],
        "/remote/data/euclid_sky/vis_noise_samples/vis_noise_sampling_manifest.json": (
            remote_files["source"]
        ),
        f"{remote_root}/cutouts/field_0000.fits": remote_files["bundle"],
    }
    fetched: list[tuple[str, dict[str, Any]]] = []

    def fetch(remote_path: str, **kwargs: Any) -> SimpleNamespace:
        fetched.append((remote_path, {**kwargs, "protect_paths": set(
            kwargs.get("protect_paths") or (),
        )}))
        local = remote_paths.get(remote_path)
        if local is None:
            return SimpleNamespace(ok=False, local_path=None, error="unexpected path")
        return SimpleNamespace(
            ok=True,
            local_path=str(local),
            error=None,
            size_bytes=local.stat().st_size,
        )

    def load_manifest(path: Path | str) -> dict[str, Any]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["collection_fingerprint"] == (
            archive_provider.compute_collection_fingerprint(payload)
        )
        return payload

    monkeypatch.setattr(routes.fasrc_config, "load", lambda: SimpleNamespace(
        data_dir="/remote/data",
    ))
    monkeypatch.setattr(routes, "ensure_ssh_connected", lambda: None)
    monkeypatch.setattr(routes.fasrc_fetcher, "fetch_one_file", fetch)
    monkeypatch.setattr(routes.archive_fields, "SAMPLE_COUNT", 1)
    monkeypatch.setattr(routes.archive_fields, "collection_root", lambda: local_root)
    monkeypatch.setattr(
        routes.archive_fields,
        "manifest_path",
        lambda: local_root / "archive_fields_manifest.json",
    )
    monkeypatch.setattr(
        routes.archive_fields, "source_manifest_path", lambda: local_source,
    )
    monkeypatch.setattr(routes.archive_fields, "load_manifest", load_manifest)
    monkeypatch.setattr(routes.archive_fields, "availability", lambda: {
        "ready": True,
        "sample_count": 1,
        "parent_count": 1,
    })
    return local_root, local_source, fetched


def test_archive_fields_get_is_local_and_available_without_ssh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "available": False,
        "ready": False,
        "reasons": ["multipoint archive manifest is unavailable"],
    }
    monkeypatch.setattr(remote.STATE, "ssh", None)
    monkeypatch.setattr(routes.archive_fields, "availability", lambda: expected)
    monkeypatch.setattr(
        routes.fasrc_fetcher,
        "fetch_one_file",
        lambda *_args, **_kwargs: pytest.fail("GET must not contact FASRC"),
    )

    from euclid_polish.web.app import create_app

    response = create_app().test_client().get("/api/archive-fields")

    assert response.status_code == 200
    assert response.get_json() == expected


def test_archive_fields_post_delegates_once_until_job_releases_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = []
    monkeypatch.setattr(routes, "_SYNC_LOCK", threading.Lock())
    monkeypatch.setattr(
        routes,
        "_sync_archive_fields",
        lambda cap: {"ready": True, "capture": cap},
    )

    def spawn(*, label: str, target):
        assert label == "archive fields: synchronize multipoint collection"
        targets.append(target)
        return f"sync-job-{len(targets)}"

    monkeypatch.setattr(routes.REGISTRY, "spawn", spawn)
    client = _route_app().test_client()

    first = client.post("/api/archive-fields/sync")
    duplicate = client.post("/api/archive-fields/sync")

    assert first.status_code == 200
    assert first.get_json() == {"ok": True, "job_id": "sync-job-1"}
    assert duplicate.status_code == 409
    assert duplicate.get_json() == {
        "ok": False,
        "error": "archive sync is already running",
    }
    capture = _Capture()
    assert targets[0](capture) == {"ready": True, "capture": capture}

    after_completion = client.post("/api/archive-fields/sync")
    assert after_completion.status_code == 200
    assert after_completion.get_json() == {"ok": True, "job_id": "sync-job-2"}
    targets[1](capture)


@pytest.mark.parametrize(
    "remote_path",
    [
        "/remote/data/euclid_sky/archive_fields/cutouts/../other/field_0000.fits",
        "/remote/data/euclid_sky/archive_fields/cutouts/nested/field_0000.fits",
        "/remote/data/euclid_sky/archive_fields/cutouts/field_0001.fits",
        "/remote/data/euclid_sky/other/cutouts/field_0000.fits",
    ],
)
def test_remote_archive_sample_path_must_be_exact(remote_path: str) -> None:
    with pytest.raises(RuntimeError, match="outside the expected collection"):
        routes._remote_sample_path(
            {"sample_id": 0, "output_path": remote_path},
            "/remote/data/euclid_sky/archive_fields",
        )


def test_archive_sync_relocates_complete_snapshot_and_source_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_files = _remote_fixture(tmp_path)
    local_root, local_source, fetched = _install_sync_fakes(
        monkeypatch, tmp_path, remote_files,
    )

    result = routes._sync_archive_fields(_Capture())

    assert result == {"ready": True, "sample_count": 1, "parent_count": 1}
    installed_bundle = local_root / "cutouts" / "field_0000.fits"
    assert installed_bundle.read_bytes() == remote_files["bundle"].read_bytes()
    installed = json.loads(
        (local_root / "archive_fields_manifest.json").read_text(encoding="utf-8")
    )
    assert installed["samples"][0]["status"] == "cached"
    assert installed["samples"][0]["output_path"] == str(installed_bundle.resolve())
    assert installed["collection_fingerprint"] == (
        remote_files["payload"]["collection_fingerprint"]
    )
    assert archive_provider.compute_collection_fingerprint(installed) == (
        remote_files["payload"]["collection_fingerprint"]
    )
    synced_source = json.loads(local_source.read_text(encoding="utf-8"))
    assert synced_source["sync"]["remote_manifest_sha256"] == (
        _sha256(remote_files["source"])
    )
    assert local_source.read_bytes() != remote_files["source"].read_bytes()
    assert [item[0] for item in fetched] == [
        "/remote/data/euclid_sky/archive_fields/archive_fields_manifest.json",
        "/remote/data/euclid_sky/vis_noise_samples/vis_noise_sampling_manifest.json",
        "/remote/data/euclid_sky/archive_fields/cutouts/field_0000.fits",
    ]
    assert all(item[1]["force"] is True for item in fetched)


def test_archive_sync_hash_failure_preserves_previous_local_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_files = _remote_fixture(tmp_path, bundle_sha="0" * 64)
    local_root, local_source, _ = _install_sync_fakes(
        monkeypatch, tmp_path, remote_files,
    )
    old_bundle = local_root / "cutouts" / "field_0000.fits"
    old_bundle.parent.mkdir(parents=True)
    old_bundle.write_bytes(b"previous-bundle")
    old_manifest = local_root / "archive_fields_manifest.json"
    old_manifest.write_bytes(b"previous-manifest")
    local_source.parent.mkdir(parents=True)
    local_source.write_bytes(b"previous-source-manifest")

    with pytest.raises(RuntimeError, match="failed its SHA256 check"):
        routes._sync_archive_fields(_Capture())

    assert old_bundle.read_bytes() == b"previous-bundle"
    assert old_manifest.read_bytes() == b"previous-manifest"
    assert local_source.read_bytes() == b"previous-source-manifest"
    assert not any(path.name.startswith(".sync-") for path in local_root.iterdir())
