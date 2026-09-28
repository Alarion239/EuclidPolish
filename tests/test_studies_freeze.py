"""The freeze / resume job (``euclid_polish/studies/freeze.py``): numbers →
manifest (incomplete) → fields one product at a time → complete → mirror.
A failure leaves an incomplete study that resumes. Fake SSH only."""
from __future__ import annotations

import json
import shutil

import pytest

from euclid_polish import ensemble_registry as er
from euclid_polish.studies import fields, freeze
from euclid_polish.studies.store import StudyError, StudyStore
from euclid_polish.web.helpers import experiments
from tests._studies_fixtures import LABELS, FakeRemote, make_env

REMOTE_TRACKING = "/n/holylabs/lab/me/EuclidPolish/tracking"


@pytest.fixture
def env(tmp_path, monkeypatch):
    out = make_env(tmp_path, monkeypatch)
    monkeypatch.setattr(experiments, "free_bytes", lambda _path: 10 ** 15)
    out["store"] = StudyStore()
    out["ssh"] = FakeRemote(tmp_path / "remote")
    return out


def _freeze(env, fids=(), *, ssh=None, name="Loss study"):
    ssh = env["ssh"] if ssh is None else ssh
    sid = freeze.start(env["store"], name=name, note="why", starless=False, fids=list(fids),
                       connected=ssh.is_connected())
    return sid, freeze.run(env["store"], sid, ssh=ssh, remote_tracking_dir=REMOTE_TRACKING)


def test_freeze_without_fields_works_offline(env, tmp_path):
    offline = FakeRemote(tmp_path / "remote", connected=False)
    sid, result = _freeze(env, ssh=offline)
    manifest = env["store"].manifest(sid)
    assert manifest["complete"] is True and manifest["fields"] == []
    assert [m["label"] for m in manifest["ensemble"]["members"]] == LABELS
    assert set(manifest["numbers"]) >= {"members.csv", "knee_psnr.json", "integrated.csv",
                                        "training_curves.json", "gate.json", "real.json"}
    assert manifest["commit"] is None or "short" in manifest["commit"]
    assert result["mirror"]["ok"] is False                 # offline: the mirror waits
    assert offline.pushed == []


def test_the_study_survives_archiving_a_member(env):
    sid, _result = _freeze(env, ssh=FakeRemote(env["base"].parent / "r", connected=False))
    shutil.rmtree(env["base"] / "member_02")               # archived + deleted locally
    er.load_registry(str(env["base"]))
    manifest = env["store"].manifest(sid)
    assert "02·psnr" in [m["label"] for m in manifest["ensemble"]["members"]]
    knee = env["store"].read_json(sid, "knee_psnr.json")
    assert "02·psnr" in [m["id"] for m in knee["models"]]


def test_freeze_with_fields_uploads_verifies_and_mirrors(env):
    sid, result = _freeze(env, ["test-00001", "blackout-00002"])
    manifest = env["store"].manifest(sid)
    assert manifest["complete"] is True
    assert [f["fid"] for f in manifest["fields"]] == ["test-00001", "blackout-00002"]
    assert all(f["state"] == "uploaded" for f in manifest["fields"])
    field_root = f"/n/holylabs/lab/me/EuclidPolish/study_fields/{sid}"
    assert manifest["remote"]["field_root"] == field_root
    assert env["ssh"].local(f"{field_root}/test-00001/field.json").is_file()
    assert env["ssh"].local(f"{REMOTE_TRACKING}/studies/{sid}/study.json").is_file()
    assert result["mirror"]["ok"] is True
    assert env["store"].read_number(sid, "thumbs/test-00001.jpg")[:3] == b"\xff\xd8\xff"


def test_a_sha_mismatch_leaves_an_incomplete_study_that_resumes(env, tmp_path):
    bad = FakeRemote(tmp_path / "remote", corrupt={"mean.npz"})
    sid = freeze.start(env["store"], name="Broken", note="", starless=False,
                       fids=["test-00000"], connected=True)
    with pytest.raises(fields.UploadError, match="mean"):
        freeze.run(env["store"], sid, ssh=bad, remote_tracking_dir=REMOTE_TRACKING)
    manifest = env["store"].manifest(sid)
    assert manifest["complete"] is False and "mean" in manifest["error"]
    (row,) = [r for r in env["store"].list() if r["id"] == sid]
    assert row["state"] == "incomplete" and "mean" in row["reason"]
    assert manifest["numbers"]                              # the numbers were kept
    freeze.run(env["store"], sid, ssh=env["ssh"], remote_tracking_dir=REMOTE_TRACKING)
    assert env["store"].manifest(sid)["complete"] is True


def test_resume_refuses_when_the_ensemble_changed(env, tmp_path):
    bad = FakeRemote(tmp_path / "remote", corrupt={"gate.npz"})
    sid = freeze.start(env["store"], name="Changed", note="", starless=False,
                       fids=["test-00000"], connected=True)
    with pytest.raises(fields.UploadError):
        freeze.run(env["store"], sid, ssh=bad, remote_tracking_dir=REMOTE_TRACKING)
    (env["base"] / "member_01" / "checkpoint").write_text('model_checkpoint_path: "ckpt-9"\n')
    (env["base"] / "member_01" / "ckpt-9.index").write_bytes(b"new weights")
    with pytest.raises(StudyError) as err:
        freeze.run(env["store"], sid, ssh=env["ssh"], remote_tracking_dir=REMOTE_TRACKING)
    assert err.value.code == 409 and "changed" in str(err.value)


def test_start_validates_fields(env):
    store = env["store"]
    with pytest.raises(StudyError) as err:
        freeze.start(store, name="x", note="", starless=False,
                     fids=[f"test-{i:05d}" for i in range(11)], connected=True)
    assert err.value.code == 400 and "10" in str(err.value)
    with pytest.raises(StudyError) as err:
        freeze.start(store, name="x", note="", starless=False, fids=["test-00099"],
                     connected=True)
    assert err.value.code == 400
    with pytest.raises(StudyError) as err:
        freeze.start(store, name="x", note="", starless=False, fids=["test-00001"],
                     connected=False)
    assert err.value.code == 503
    with pytest.raises(StudyError) as err:
        freeze.start(store, name="", note="", starless=False, fids=[], connected=True)
    assert err.value.code == 400
    assert store.list() == []                              # nothing was written


def test_start_refuses_unavailable_fields_and_stale_cubes(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch, blackout_labels=LABELS[:2])
    store = StudyStore()
    with pytest.raises(StudyError) as err:
        freeze.start(store, name="x", note="", starless=False, fids=["blackout-00000"],
                     connected=True)
    assert err.value.code == 409 and "member(s) 03" in str(err.value)


def test_start_refuses_to_break_the_disk_margin(env, monkeypatch):
    monkeypatch.setattr(experiments, "free_bytes", lambda _path: experiments.MIN_FREE_BYTES)
    with pytest.raises(experiments.DiskSpaceError):
        freeze.start(env["store"], name="x", note="", starless=False, fids=[], connected=True)


def test_start_records_the_plan_in_an_incomplete_manifest(env):
    sid = freeze.start(env["store"], name="Plan", note="n", starless=False,
                       fids=["test-00002"], connected=True)
    manifest = env["store"].manifest(sid)
    assert manifest["complete"] is False and manifest["regime"] == "starfull"
    assert manifest["fields"][0]["fid"] == "test-00002"
    assert manifest["fields"][0]["state"] == "pending"
    assert manifest["identity"]["labels"] == LABELS
    assert json.loads(json.dumps(manifest)) == manifest


def test_a_failed_thumbnail_never_causes_a_reupload(env, monkeypatch):
    sid = freeze.start(env["store"], name="Thumb", note="", starless=False,
                       fids=["test-00001"], connected=True)

    def broken(_cube, side=160):
        raise RuntimeError("thumbnail renderer broke")

    real = freeze.candidates.render_thumbnail
    monkeypatch.setattr(freeze.candidates, "render_thumbnail", broken)
    with pytest.raises(RuntimeError, match="thumbnail"):
        freeze.run(env["store"], sid, ssh=env["ssh"], remote_tracking_dir=REMOTE_TRACKING)
    assert env["store"].manifest(sid)["fields"][0]["state"] == "uploaded"
    pushed = len(env["ssh"].pushed)
    monkeypatch.setattr(freeze.candidates, "render_thumbnail", real)
    assert env["store"].manifest(sid)["complete"] is False
    freeze.run(env["store"], sid, ssh=env["ssh"], remote_tracking_dir=REMOTE_TRACKING)
    field_pushes = [p for p in env["ssh"].pushed[pushed:] if "/study_fields/" in "/" + p]
    assert field_pushes == [] and env["store"].manifest(sid)["complete"] is True


def test_a_missing_lr_refuses_before_anything_is_uploaded(env):
    (env["cubes"] / "lr_00001.npy").unlink()
    (env["records"] / "dirty_test.tfrecord").unlink()
    sid = freeze.start(env["store"], name="No LR", note="", starless=False,
                       fids=["test-00001"], connected=True)
    with pytest.raises(RuntimeError, match="no LR input"):
        freeze.run(env["store"], sid, ssh=env["ssh"], remote_tracking_dir=REMOTE_TRACKING)
    assert not [p for p in env["ssh"].pushed if "study_fields" in p]
