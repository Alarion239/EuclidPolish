"""The study store: immutable, whole-ensemble model comparisons for the paper.

A study is a directory ``<Config.TRACKING_DIR>/studies/<id>/``::

    study.json        the manifest (ensemble snapshot, gate + records identity,
                      numbers and field hashes, ``complete``)
    numbers/          members.csv, knee_psnr.json, integrated.csv,
                      training_curves.json, gate.json, real.json, thumbs/
    note.json         the note (mutable sidecar)
    selections.json   named member selections (mutable sidecar)

It lives outside every tracking campaign (none may be active, and saving a
campaign moves ``current/``), never references a live checkpoint, cube or
cache, and becomes immutable once ``complete``: :meth:`StudyStore.mark_complete`
refuses a study with missing numbers or fields not yet uploaded, then makes
the numbers read-only. Only the two sidecars change afterwards.

Attached fields live on holylabs only, under :func:`remote_field_root` —
``<parent of the remote tracking dir>/study_fields/<id>/<field>/``, a sibling
of the directory ``tracking.sync.push`` mirrors, so no tracking push (plain
``rsync -az``, no ``--delete``; and rsync only ever writes inside its
destination) can ever touch it. The numbers are mirrored explicitly to
``<remote tracking dir>/studies/<id>/`` (:func:`remote_numbers_dir`).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import posixpath
import re
import shlex
import shutil
import stat
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from euclid_polish.config import Config
from euclid_polish.web.helpers import atomic_files

SCHEMA = 1
MANIFEST = "study.json"
NUMBERS = "numbers"
NOTE = "note.json"
SELECTIONS = "selections.json"
#: At most this many fields may be attached to one study.
MAX_FIELDS = 10
#: The remote directory (beside the mirrored tracking dir) holding the fields.
FIELD_ROOT_NAME = "study_fields"
_STUDY_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-z0-9][a-z0-9-]{0,39}$")
#: An attached field: ``test-NNNNN`` / ``blackout-NNNNN`` (record index) or
#: ``real-<source>-<tile id>`` — safe as one path component.
_FIELD_ID = re.compile(r"^(?:(?:test|blackout)-[0-9]{5}|"
                       r"real-[a-z]{2,16}-[A-Za-z0-9][A-Za-z0-9._-]{0,199})$")
#: A numbers file (``thumbs/<field id>.jpg`` included: its name allows the
#: longest field id, 222 characters, plus the extension).
_NUMBER_NAME = re.compile(r"^(?:thumbs/)?[A-Za-z0-9][A-Za-z0-9._-]{0,239}$")
_LOCK = threading.RLock()


class StudyError(ValueError):
    """A refused study operation; ``code`` is the HTTP status to answer."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = int(code)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(name or "").lower()).strip("-")
    return (slug[:40].strip("-") or "study")


def new_study_id(name: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{slugify(name)}"


def check_study_id(study_id: str) -> str:
    if not _STUDY_ID.fullmatch(str(study_id or "")):
        raise StudyError(404, f"unknown study {study_id!r}")
    return str(study_id)


def check_field_id(fid: str) -> str:
    """A well-formed field id (:class:`ValueError` otherwise)."""
    text = str(fid or "")
    if not _FIELD_ID.fullmatch(text) or ".." in text:
        raise ValueError(f"bad field id {fid!r}")
    return text


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# remote locations (holylabs)
# ---------------------------------------------------------------------------

def remote_field_root(remote_tracking_dir: str) -> str:
    """``<parent of the remote tracking dir>/study_fields`` — beside, never
    inside, the directory the tracking mirror pushes into."""
    parent = posixpath.dirname(str(remote_tracking_dir).rstrip("/")) or "/"
    return posixpath.join(parent, FIELD_ROOT_NAME)


def remote_field_dir(remote_tracking_dir: str, study_id: str) -> str:
    return posixpath.join(remote_field_root(remote_tracking_dir), check_study_id(study_id))


def remote_numbers_dir(remote_tracking_dir: str, study_id: str) -> str:
    return posixpath.join(str(remote_tracking_dir).rstrip("/"), "studies",
                          check_study_id(study_id))


def safe_remote_study_dir(path: str, study_id: str, marker: str) -> bool:
    """The ``rm -rf`` guard: absolute, reasonably deep, ``/<marker>/`` in it,
    no ``..`` and the study id as its last component."""
    parts = path.split("/")
    return (path.startswith("/") and path.count("/") >= 4 and f"/{marker}/" in path
            and ".." not in parts and parts[-1] == study_id
            and bool(_STUDY_ID.fullmatch(study_id)))


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------

class StudyStore:
    """Studies under ``root`` (default ``<Config.TRACKING_DIR>/studies``)."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = os.path.abspath(os.fspath(root) if root is not None
                                    else os.path.join(Config.TRACKING_DIR, "studies"))

    # ----------------------------- paths ---------------------------------

    def path(self, study_id: str) -> Path:
        return Path(self.root) / check_study_id(study_id)

    def _existing(self, study_id: str) -> Path:
        directory = self.path(study_id)
        if not (directory / MANIFEST).is_file():
            raise StudyError(404, f"unknown study {study_id!r}")
        return directory

    def _number_path(self, study_id: str, name: str) -> Path:
        if not _NUMBER_NAME.fullmatch(str(name or "")) or ".." in str(name):
            raise StudyError(400, f"bad numbers file name {name!r}")
        return self._existing(study_id) / NUMBERS / name

    # ----------------------------- manifest ------------------------------

    def create(self, name: str, note: str, *, regime: str,
               extra: Mapping[str, Any] | None = None) -> str:
        """A new, incomplete study; returns its id."""
        name = str(name or "").strip()
        if not name:
            raise StudyError(400, "a study needs a name")
        with _LOCK:
            base = new_study_id(name)
            study_id, n = base, 2
            while self.path(study_id).exists():
                stamp, slug = base[:15], base[16:]
                study_id = f"{stamp}-{slug[:36].strip('-')}-{n}"
                n += 1
            directory = self.path(study_id)
            (directory / NUMBERS).mkdir(parents=True)
            manifest = {"schema": SCHEMA, "id": study_id, "name": name,
                        "note": str(note or ""), "created": _now(), "regime": regime,
                        "complete": False, "numbers": {}, "fields": [], **(extra or {})}
            atomic_files.write_json(directory / MANIFEST, manifest, indent=2,
                                    allow_nan=False)
            atomic_files.write_json(directory / NOTE, {"note": str(note or ""),
                                                       "updated": manifest["created"]},
                                    indent=2)
        return study_id

    def manifest(self, study_id: str) -> dict[str, Any]:
        path = self._existing(study_id) / MANIFEST
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StudyError(404, f"unreadable study {study_id!r}: {exc}") from exc
        if not isinstance(payload, dict):
            raise StudyError(404, f"unreadable study {study_id!r}")
        return payload

    def manifest_sha256(self, study_id: str) -> str:
        """sha256 of ``study.json`` as stored (the study's citable identity)."""
        return sha256_bytes((self._existing(study_id) / MANIFEST).read_bytes())

    def _require_open(self, study_id: str) -> dict[str, Any]:
        manifest = self.manifest(study_id)
        if manifest.get("complete"):
            raise StudyError(409, f"study {study_id} is complete and immutable")
        return manifest

    def write_manifest(self, study_id: str, manifest: Mapping[str, Any]) -> None:
        """Replace the manifest of an INCOMPLETE study (atomic)."""
        with _LOCK:
            current = self._require_open(study_id)
            payload = {**dict(manifest), "id": current["id"], "complete": False}
            atomic_files.write_json(self._existing(study_id) / MANIFEST, payload,
                                    indent=2, allow_nan=False)

    def update_manifest(self, study_id: str, **changes: Any) -> dict[str, Any]:
        with _LOCK:
            manifest = {**self._require_open(study_id), **changes}
            self.write_manifest(study_id, manifest)
            return manifest

    def write_number(self, study_id: str, name: str, data: bytes) -> dict[str, Any]:
        """Write one numbers file of an incomplete study; its hash goes into
        the manifest. Returns ``{sha256, bytes}``."""
        with _LOCK:
            manifest = self._require_open(study_id)
            path = self._number_path(study_id, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with atomic_files.temporary_sibling(path) as temporary:
                temporary.write_bytes(data)
                os.replace(temporary, path)
            info = {"sha256": sha256_bytes(data), "bytes": len(data)}
            manifest.setdefault("numbers", {})[name] = info
            self.write_manifest(study_id, manifest)
            return info

    def read_number(self, study_id: str, name: str) -> bytes:
        path = self._number_path(study_id, name)
        try:
            return path.read_bytes()
        except OSError as exc:
            raise StudyError(404, f"study {study_id} has no {name}") from exc

    def read_json(self, study_id: str, name: str) -> Any:
        try:
            return json.loads(self.read_number(study_id, name).decode("utf-8"))
        except ValueError as exc:
            raise StudyError(500, f"study {study_id}: {name} is not JSON") from exc

    def mark_complete(self, study_id: str) -> dict[str, Any]:
        """Seal a study: every numbers file hashed and present, every field
        uploaded. The numbers become read-only."""
        with _LOCK:
            manifest = self._require_open(study_id)
            numbers = manifest.get("numbers") or {}
            if not numbers:
                raise StudyError(409, f"study {study_id} has no numbers yet")
            for name, info in numbers.items():
                data = self.read_number(study_id, name)
                if sha256_bytes(data) != (info or {}).get("sha256"):
                    raise StudyError(409, f"study {study_id}: {name} does not match its hash")
            pending = [f.get("fid") for f in manifest.get("fields") or []
                       if f.get("state") != "uploaded"]
            if pending:
                raise StudyError(409, "fields not uploaded yet: " + ", ".join(map(str, pending)))
            manifest.pop("error", None)
            manifest["complete"] = True
            manifest["completed"] = _now()
            directory = self._existing(study_id)
            atomic_files.write_json(directory / MANIFEST, manifest, indent=2,
                                    allow_nan=False)
            for path in (directory / NUMBERS).rglob("*"):
                if path.is_file():
                    mode = path.stat().st_mode
                    os.chmod(path, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            return manifest

    # ----------------------------- listing -------------------------------

    def list(self) -> list[dict[str, Any]]:
        """Every study, newest first: complete ones and incomplete ones (with
        the reason they are not usable)."""
        root = Path(self.root)
        if not root.is_dir():
            return []
        rows = []
        for directory in sorted(root.iterdir(), reverse=True):
            if not _STUDY_ID.fullmatch(directory.name) or not directory.is_dir():
                continue
            try:
                manifest = self.manifest(directory.name)
            except StudyError as exc:
                rows.append({"id": directory.name, "state": "incomplete", "reason": str(exc)})
                continue
            rows.append(self.summary(manifest))
        return rows

    def summary(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        study_id = str(manifest.get("id"))
        complete = bool(manifest.get("complete"))
        fields = manifest.get("fields") or []
        reason = None
        if not complete:
            pending = [f.get("fid") for f in fields if f.get("state") != "uploaded"]
            reason = manifest.get("error") or (
                f"{len(pending)} field(s) not uploaded" if pending else
                "numbers not written" if not manifest.get("numbers") else "not sealed")
        ensemble = manifest.get("ensemble") or {}
        gate = manifest.get("gate") or {}
        return {
            "id": study_id, "name": manifest.get("name"), "created": manifest.get("created"),
            "completed": manifest.get("completed"), "regime": manifest.get("regime"),
            "members": len(ensemble.get("members") or []),
            "gate": gate.get("name") or gate.get("promoted_from"),
            "fields": len(fields), "field_ids": [f.get("fid") for f in fields],
            "note": self.note(study_id), "commit": manifest.get("commit"),
            "state": "complete" if complete else "incomplete", "reason": reason,
            "numbers_bytes": sum(int((v or {}).get("bytes") or 0)
                                 for v in (manifest.get("numbers") or {}).values()),
            "fields_bytes": sum(int(f.get("bytes") or 0) for f in fields),
        }

    # ----------------------------- sidecars ------------------------------

    def note(self, study_id: str) -> str:
        directory = self._existing(study_id)
        try:
            return str(json.loads((directory / NOTE).read_text(encoding="utf-8")).get("note", ""))
        except (OSError, ValueError, AttributeError):
            return str(self.manifest(study_id).get("note") or "")

    def set_note(self, study_id: str, note: str) -> str:
        directory = self._existing(study_id)
        text = str(note or "")[:4000]
        atomic_files.write_json(directory / NOTE, {"note": text, "updated": _now()}, indent=2)
        return text

    def selections(self, study_id: str) -> list[dict[str, Any]]:
        directory = self._existing(study_id)
        try:
            payload = json.loads((directory / SELECTIONS).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [dict(s) for s in payload.get("selections", []) if isinstance(s, dict)]

    def set_selections(self, study_id: str, selections: Sequence[Mapping[str, Any]], *,
                       members: Sequence[str] | None = None,
                       groups: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Replace the named selections: ``[{name, members?: [label],
        group?: recipe field, note?}]`` (names unique, ≤ 100). ``members`` /
        ``groups`` — the study's member labels and the allowed grouping
        fields — reject unknown ones (400)."""
        directory = self._existing(study_id)
        known = set(members) if members is not None else None
        out: list[dict[str, Any]] = []
        names: set[str] = set()
        for item in list(selections)[:100]:
            if not isinstance(item, Mapping):
                raise StudyError(400, "each selection must be an object")
            name = str(item.get("name") or "").strip()[:80]
            if not name or name in names:
                raise StudyError(400, "each selection needs a unique name")
            names.add(name)
            picked = item.get("members")
            if picked is not None and not isinstance(picked, list):
                raise StudyError(400, f"selection {name!r}: members must be a list of labels")
            labels = [str(m) for m in picked] if isinstance(picked, list) else None
            if known is not None and labels is not None:
                unknown = [label for label in labels if label not in known]
                if unknown or not labels:
                    raise StudyError(400, f"selection {name!r}: not members of this study: "
                                          + (", ".join(unknown) or "(empty)"))
            group = str(item["group"]) if item.get("group") else None
            if group is not None and groups is not None and group not in groups:
                raise StudyError(400, f"selection {name!r}: cannot group by {group!r}")
            out.append({"name": name, "members": labels, "group": group,
                        "note": str(item.get("note") or "")[:1000]})
        atomic_files.write_json(directory / SELECTIONS,
                                {"selections": out, "updated": _now()}, indent=2)
        return out

    # ----------------------------- delete --------------------------------

    def delete(self, study_id: str, ssh: Any, *, remote_tracking_dir: str,
               local_only: bool = False) -> dict[str, Any]:
        """Delete a study locally and its holylabs copies (field store and
        numbers mirror). A study with attached fields needs FASRC unless
        ``local_only``: then only the local study goes and the holylabs copies
        are always KEPT (connected or not) and reported."""
        manifest = self.manifest(study_id)
        has_fields = bool(manifest.get("fields"))
        connected = ssh is not None and bool(getattr(ssh, "is_connected", lambda: False)())
        if has_fields and not connected and not local_only:
            raise StudyError(409, (
                "connect FASRC to delete the study's holylabs field store, or resend "
                "with local_only=1 (the holylabs copy then stays)"))
        remote: list[str] = []
        targets = [(remote_field_dir(remote_tracking_dir, study_id), FIELD_ROOT_NAME),
                   (remote_numbers_dir(remote_tracking_dir, study_id), "studies")]
        for path, marker in targets:
            if local_only:
                remote.append(f"NOT deleted — kept on FASRC (local_only): {path}.")
            elif not safe_remote_study_dir(path, study_id, marker):
                remote.append(f"NOT deleted on FASRC (refused unsafe path {path!r}).")
            elif not connected or ssh is None:
                remote.append(f"NOT deleted on FASRC (not connected) — remove {path} there.")
            else:
                try:
                    rc, _out, err = ssh.run(f"rm -rf {shlex.quote(path)}", timeout=120)
                    remote.append(f"deleted on FASRC ({path})." if rc == 0 else
                                  f"FASRC delete failed (rc={rc}: {str(err).strip()[:200]}) "
                                  f"— remove {path} there.")
                except Exception as exc:  # noqa: BLE001 — remote cleanup is best-effort
                    remote.append(f"FASRC delete failed ({type(exc).__name__}: {exc}) — "
                                  f"remove {path} there.")
        with _LOCK:
            directory = self._existing(study_id)
            for path in directory.rglob("*"):
                with contextlib.suppress(OSError):
                    if path.is_file():
                        os.chmod(path, path.stat().st_mode | stat.S_IWUSR)
            shutil.rmtree(directory)
        return {"ok": True, "id": study_id, "remote": remote}


__all__ = [
    "FIELD_ROOT_NAME",
    "MAX_FIELDS",
    "SCHEMA",
    "StudyError",
    "StudyStore",
    "check_field_id",
    "check_study_id",
    "new_study_id",
    "remote_field_dir",
    "remote_field_root",
    "remote_numbers_dir",
    "safe_remote_study_dir",
    "sha256_bytes",
    "sha256_file",
    "slugify",
]
