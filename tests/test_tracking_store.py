"""Tests for the experiment-tracking store (euclid_polish.tracking.store)."""

from __future__ import annotations

import json
import os

import pytest

from euclid_polish.tracking.store import (
    TrackingError,
    TrackingStore,
    _PROJECT_ROOT,
    git_commit_info,
)


@pytest.fixture
def store(tmp_path):
    """A store rooted in tmp, with a NON-git repo_root so commit stamps are
    deterministically ``None`` (no dependency on the working tree's state)."""
    return TrackingStore(root=str(tmp_path / "tracking"),
                         repo_root=str(tmp_path / "not-a-repo"))


def _make_fits(path, data=b"SIMPLE  = T" + b" " * 100):
    with open(path, "wb") as fp:
        fp.write(data)
    return path


# --------------------------------------------------------------------------
# campaign lifecycle
# --------------------------------------------------------------------------

def test_create_campaign_seeds_layout(store):
    meta = store.create_campaign("My First Run", description="trying anchors")
    assert store.has_current()
    assert meta["title"] == "My First Run"
    assert meta["slug"] == "my-first-run"
    assert meta["status"] == "active"
    assert meta["saved_at"] is None
    assert meta["created_commit"] is None  # repo_root is not a git repo
    # subdirs + seed files
    for sub in ("models", "fits", "images"):
        assert os.path.isdir(os.path.join(store.current_dir, sub))
    log = store.read_log()
    assert "# My First Run" in log
    assert "trying anchors" in log
    assert os.path.isfile(os.path.join(store.current_dir, "fasrc_jobs.jsonl"))


def test_create_twice_raises(store):
    store.create_campaign("one")
    with pytest.raises(TrackingError):
        store.create_campaign("two")


def test_backup_without_campaign_raises(store, tmp_path):
    src = _make_fits(tmp_path / "x.fits")
    with pytest.raises(TrackingError):
        store.backup_fits(str(src), comment="nope")


def test_save_campaign_moves_to_archive(store):
    store.create_campaign("Experiment Alpha")
    res = store.save_campaign()
    assert not store.has_current()                       # current consumed
    assert os.path.basename(res["archive_path"]) == "experiment-alpha"
    assert os.path.isdir(res["archive_path"])
    saved_meta = json.load(
        open(os.path.join(res["archive_path"], "metadata.json")))
    assert saved_meta["status"] == "saved"
    assert saved_meta["saved_at"] is not None
    listing = store.list_campaigns()
    assert listing["active"] is None
    assert len(listing["archived"]) == 1
    assert listing["archived"][0]["title"] == "Experiment Alpha"


def test_save_then_new_campaign_allowed(store):
    store.create_campaign("first")
    store.save_campaign()
    meta = store.create_campaign("second")           # no longer blocked
    assert meta["title"] == "second"
    assert store.list_campaigns()["active"]["title"] == "second"


def test_archive_name_collision_suffixes(store):
    store.create_campaign("dup")
    store.save_campaign()
    store.create_campaign("dup")
    store.save_campaign()
    names = {c["_dir"] for c in store.list_campaigns()["archived"]}
    assert names == {"dup", "dup-2"}


# --------------------------------------------------------------------------
# backups
# --------------------------------------------------------------------------

def test_backup_fits_copies_with_sidecar(store, tmp_path):
    store.create_campaign("c")
    src = _make_fits(tmp_path / "result.fits")
    rec = store.backup_fits(str(src), comment="the SR output")
    stored = os.path.join(store.current_dir, "fits", rec["name"])
    assert os.path.isfile(stored)
    side = json.load(open(stored + ".meta.json"))
    assert side["kind"] == "fits"
    assert side["comment"] == "the SR output"
    assert side["source_path"] == str(src)
    assert side["size_bytes"] == os.path.getsize(stored)


def test_backup_image_uses_images_dir(store, tmp_path):
    store.create_campaign("c")
    src = tmp_path / "plot.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 50)
    rec = store.backup_image(str(src), comment="training curve")
    assert os.path.isfile(os.path.join(store.current_dir, "images", rec["name"]))
    assert rec["kind"] == "image"


def test_duplicate_filenames_get_unique_names(store, tmp_path):
    store.create_campaign("c")
    os.makedirs(tmp_path / "d1", exist_ok=True)
    os.makedirs(tmp_path / "d2", exist_ok=True)
    s1 = _make_fits(tmp_path / "d1" / "same.fits")
    s2 = _make_fits(tmp_path / "d2" / "same.fits")
    r1 = store.backup_fits(str(s1))
    r2 = store.backup_fits(str(s2))
    assert r1["name"] != r2["name"]
    assert {r1["name"], r2["name"]} == {"same.fits", "same-2.fits"}


def test_backup_model_copies_restorable_set(store, tmp_path):
    store.create_campaign("c")
    ck = tmp_path / "ckpt"
    ck.mkdir()
    (ck / "checkpoint").write_text('model_checkpoint_path: "ckpt-5"\n')
    (ck / "ckpt-5.index").write_bytes(b"idx")
    (ck / "ckpt-5.data-00000-of-00001").write_bytes(b"weights")
    (ck / "training_log.csv").write_text("step,loss\n1,0.5\n")
    (ck / "training_log.20260101.bak").write_text("old\n")   # skipped
    lb = ck / "loss_best"
    lb.mkdir()
    (lb / "checkpoint").write_text('model_checkpoint_path: "ckpt-13"\n')
    (lb / "ckpt-13.index").write_bytes(b"idx2")

    rec = store.backup_model(str(ck), comment="best so far", name="anchors-on")
    dest = os.path.join(store.current_dir, "models", "anchors-on")
    assert os.path.isfile(os.path.join(dest, "checkpoint"))
    assert os.path.isfile(os.path.join(dest, "ckpt-5.data-00000-of-00001"))
    assert os.path.isfile(os.path.join(dest, "training_log.csv"))
    assert os.path.isfile(os.path.join(dest, "loss_best", "ckpt-13.index"))
    # .bak rotations are not part of a restorable snapshot
    assert not os.path.exists(os.path.join(dest, "training_log.20260101.bak"))
    assert rec["kind"] == "model"
    assert "loss_best/ckpt-13.index" in rec["files"]
    assert rec["comment"] == "best so far"


def test_backup_model_missing_dir_raises(store, tmp_path):
    store.create_campaign("c")
    with pytest.raises(TrackingError):
        store.backup_model(str(tmp_path / "does-not-exist"))


def test_list_backups_enumerates_all_kinds(store, tmp_path):
    store.create_campaign("c")
    store.backup_fits(str(_make_fits(tmp_path / "r.fits")), comment="f")
    png = tmp_path / "p.png"
    png.write_bytes(b"\x89PNG")
    store.backup_image(str(png), comment="i")
    out = store.list_backups()
    assert len(out["fits"]) == 1
    assert len(out["images"]) == 1
    assert out["fits"][0]["comment"] == "f"


# --------------------------------------------------------------------------
# log + fasrc jobs
# --------------------------------------------------------------------------

def test_write_and_append_log(store):
    store.create_campaign("c")
    store.write_log("# Fresh\n")
    assert store.read_log() == "# Fresh\n"
    store.append_log("ran a thing")
    log = store.read_log()
    assert "# Fresh" in log
    assert "ran a thing" in log


def test_log_fasrc_job_appends_to_active(store):
    store.create_campaign("c")
    path = store.log_fasrc_job({"jobid": "12345", "label": "train",
                                "params": {"steps": 400000}})
    assert path.endswith("fasrc_jobs.jsonl")
    jobs = store.read_fasrc_jobs()
    assert len(jobs) == 1
    assert jobs[0]["jobid"] == "12345"
    assert "logged_at" in jobs[0]


def test_log_fasrc_job_without_campaign_falls_back(store):
    path = store.log_fasrc_job({"jobid": "999"})
    assert path.endswith("unassigned_fasrc_jobs.jsonl")
    assert os.path.isfile(path)
    # read_fasrc_jobs only reads the active campaign's log → empty here
    assert store.read_fasrc_jobs() == []


# --------------------------------------------------------------------------
# git commit stamping
# --------------------------------------------------------------------------

def test_git_commit_info_on_real_repo_has_keys():
    info = git_commit_info(_PROJECT_ROOT)
    # The project itself is a git repo, so this should resolve.
    assert info is not None
    assert set(info) == {"hash", "short", "branch", "dirty"}
    assert isinstance(info["dirty"], bool)
    assert len(info["hash"]) >= 7


def test_git_commit_info_non_repo_returns_none(tmp_path):
    assert git_commit_info(str(tmp_path / "nope")) is None
