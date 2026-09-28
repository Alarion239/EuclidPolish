"""The model-study store (``euclid_polish/studies/store.py``): ids, the
manifest, immutability once complete, the incomplete listing, the mutable
sidecars and the guarded delete of the holylabs copies. tmp dirs and a fake
SSH session only."""
from __future__ import annotations

import json
import os

import pytest

from euclid_polish.config import Config
from euclid_polish.studies import store as st


class FakeSSH:
    def __init__(self, connected=True):
        self.connected = connected
        self.commands: list[str] = []

    def is_connected(self):
        return self.connected

    def run(self, cmd, timeout=60):
        self.commands.append(cmd)
        return (0, "", "")


@pytest.fixture
def store(tmp_path):
    return st.StudyStore(tmp_path / "studies")


def _complete(store, sid, fields=()):
    store.write_number(sid, "members.csv", b"label\n1\n")
    manifest = store.manifest(sid)
    manifest["fields"] = [{"fid": f, "state": "uploaded", "bytes": 10} for f in fields]
    store.write_manifest(sid, manifest)
    store.mark_complete(sid)


def test_default_root_is_under_the_tracking_dir():
    assert st.StudyStore().root == os.path.abspath(os.path.join(Config.TRACKING_DIR, "studies"))


def test_study_id_is_stamp_plus_slug(store):
    sid = store.create("Loss × knee (paper)", "why l1", regime="starfull")
    assert st.check_study_id(sid) == sid
    assert sid.endswith("-loss-knee-paper")
    with pytest.raises(st.StudyError) as err:
        st.check_study_id("../etc")
    assert err.value.code == 404


def test_create_requires_a_name(store):
    with pytest.raises(st.StudyError) as err:
        store.create("  ", "", regime="starfull")
    assert err.value.code == 400


def test_manifest_roundtrip_and_numbers_hashes(store):
    sid = store.create("A", "note", regime="starfull")
    manifest = store.manifest(sid)
    assert manifest["id"] == sid and manifest["name"] == "A"
    assert manifest["complete"] is False and manifest["regime"] == "starfull"
    info = store.write_number(sid, "knee_psnr.json", json.dumps({"x": 1}).encode())
    assert info["bytes"] == len(b'{"x": 1}') and len(info["sha256"]) == 64
    assert store.manifest(sid)["numbers"]["knee_psnr.json"] == info
    assert store.read_json(sid, "knee_psnr.json") == {"x": 1}
    with pytest.raises(st.StudyError):
        store.write_number(sid, "../evil.json", b"{}")


def test_complete_study_is_immutable(store):
    sid = store.create("A", "", regime="starfull")
    _complete(store, sid)
    assert store.manifest(sid)["complete"] is True
    with pytest.raises(st.StudyError) as err:
        store.write_number(sid, "members.csv", b"changed")
    assert err.value.code == 409
    with pytest.raises(st.StudyError):
        store.write_manifest(sid, {**store.manifest(sid), "name": "B"})
    assert store.read_number(sid, "members.csv") == b"label\n1\n"
    # The frozen numbers are read-only on disk too.
    path = store.path(sid) / "numbers" / "members.csv"
    assert not os.access(path, os.W_OK)


def test_mark_complete_refuses_without_numbers_or_with_pending_fields(store):
    sid = store.create("A", "", regime="starfull")
    with pytest.raises(st.StudyError):
        store.mark_complete(sid)
    store.write_number(sid, "members.csv", b"x")
    manifest = store.manifest(sid)
    manifest["fields"] = [{"fid": "test-00001", "state": "pending"}]
    store.write_manifest(sid, manifest)
    with pytest.raises(st.StudyError, match="not uploaded"):
        store.mark_complete(sid)


def test_list_shows_complete_and_incomplete_with_reason(store):
    a = store.create("Done", "n", regime="starfull")
    _complete(store, a, fields=["test-00001"])
    b = store.create("Broken", "", regime="starfull")
    manifest = store.manifest(b)
    manifest["error"] = "sha mismatch on member_170"
    store.write_manifest(b, manifest)
    rows = {row["id"]: row for row in store.list()}
    assert rows[a]["state"] == "complete" and rows[a]["fields"] == 1
    assert rows[b]["state"] == "incomplete"
    assert "sha mismatch" in rows[b]["reason"]
    assert [row["id"] for row in store.list()] == sorted(rows, reverse=True)


def test_note_and_selections_are_the_mutable_sidecars(store):
    sid = store.create("A", "first", regime="starfull")
    _complete(store, sid)
    assert store.note(sid) == "first"
    store.set_note(sid, "second")
    assert store.note(sid) == "second"
    assert store.list()[0]["note"] == "second"
    assert store.manifest(sid)["note"] == "first"          # the frozen record keeps its own
    store.set_selections(sid, [{"name": "l1 only", "members": ["170·psnr"], "group": "loss"}])
    assert store.selections(sid)[0]["name"] == "l1 only"
    with pytest.raises(st.StudyError):
        store.set_selections(sid, [{"members": []}])         # a selection needs a name


def test_remote_roots_sit_beside_the_mirrored_tracking_dir():
    remote = "/n/holylabs/lab/me/EuclidPolish/tracking"
    root = st.remote_field_root(remote)
    assert root == "/n/holylabs/lab/me/EuclidPolish/study_fields"
    # Never inside the directory ``tracking.sync.push`` mirrors.
    assert not root.startswith(remote.rstrip("/") + "/")
    assert st.remote_numbers_dir(remote, "20260928-120000-a") == (
        "/n/holylabs/lab/me/EuclidPolish/tracking/studies/20260928-120000-a")


def test_delete_removes_local_and_guarded_remote_dirs(store):
    sid = store.create("A", "", regime="starfull")
    _complete(store, sid, fields=["test-00001"])
    ssh = FakeSSH()
    out = store.delete(sid, ssh, remote_tracking_dir="/n/holylabs/lab/me/EuclidPolish/tracking")
    assert not store.path(sid).exists()
    assert out["ok"] is True
    assert ssh.commands == [
        f"rm -rf /n/holylabs/lab/me/EuclidPolish/study_fields/{sid}",
        f"rm -rf /n/holylabs/lab/me/EuclidPolish/tracking/studies/{sid}",
    ]


def test_delete_refuses_unsafe_remote_paths(store):
    sid = store.create("A", "", regime="starfull")
    _complete(store, sid, fields=["test-00001"])
    ssh = FakeSSH()
    out = store.delete(sid, ssh, remote_tracking_dir="/tracking")
    assert ssh.commands == []
    assert all("refused" in line for line in out["remote"])


def test_delete_with_fields_offline_needs_local_only(store):
    sid = store.create("A", "", regime="starfull")
    _complete(store, sid, fields=["test-00001"])
    with pytest.raises(st.StudyError) as err:
        store.delete(sid, FakeSSH(connected=False), remote_tracking_dir="/n/a/b/c/tracking")
    assert err.value.code == 409
    assert store.path(sid).exists()
    out = store.delete(sid, None, remote_tracking_dir="/n/a/b/c/tracking", local_only=True)
    assert not store.path(sid).exists() and "NOT deleted" in " ".join(out["remote"])


def test_delete_without_fields_works_offline(store):
    sid = store.create("A", "", regime="starfull")
    _complete(store, sid)
    out = store.delete(sid, None, remote_tracking_dir="/n/a/b/c/tracking")
    assert out["ok"] and not store.path(sid).exists()


def test_unknown_study_is_404(store):
    with pytest.raises(st.StudyError) as err:
        store.manifest("20260101-000000-nope")
    assert err.value.code == 404


def test_the_longest_field_id_fits_a_thumbnail_name(store):
    fid = "real-poster-" + "x" * 200
    assert st.check_field_id(fid) == fid
    sid = store.create("Long", "", regime="starfull")
    info = store.write_number(sid, f"thumbs/{fid}.jpg", b"\xff\xd8\xff")
    assert info["bytes"] == 3
    assert store.read_number(sid, f"thumbs/{fid}.jpg") == b"\xff\xd8\xff"


def test_local_only_keeps_the_holylabs_copy_even_when_connected(store):
    sid = store.create("A", "", regime="starfull")
    _complete(store, sid, fields=["test-00001"])
    ssh = FakeSSH()
    out = store.delete(sid, ssh, remote_tracking_dir="/n/a/b/c/tracking", local_only=True)
    assert ssh.commands == [] and not store.path(sid).exists()
    assert all("kept on FASRC" in line for line in out["remote"])
