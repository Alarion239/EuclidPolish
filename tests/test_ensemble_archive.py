"""job_archive_member: zip → tracking, tombstone, delete, rebuild cache."""
from __future__ import annotations

import json
import os
import zipfile

import pytest


class _Cap:
    def tick(self, *a):
        pass

    def write(self, *a):
        pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    from euclid_polish.config import Config
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR",
                        str(tmp_path / "ckpt/wdsr"))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    monkeypatch.setattr(Config, "TRACKING_DIR", str(tmp_path / "tracking"))
    from euclid_polish.tracking import default_store
    store = default_store()
    store.create_campaign("archive test")
    return tmp_path


def _mk_members(base, *idxs):
    for i in idxs:
        d = os.path.join(base, f"member_{i:02d}")
        os.makedirs(d)
        with open(os.path.join(d, "checkpoint"), "w") as f:
            f.write("weights")


def test_archive_member_full_flow(env, monkeypatch):
    from euclid_polish import ensemble_registry as er
    from euclid_polish.tracking import default_store
    from euclid_polish.web.helpers import ensemble_viz as ev
    from euclid_polish.web.remote import STATE

    # Genuinely disconnected (the conftest session stub PRETENDS connected
    # and swallows commands with rc=0, which would read as a remote delete).
    monkeypatch.setattr(STATE, "ssh", None)

    base = ev.ensemble_dir()
    _mk_members(base, 0, 1)
    cubes = ev._ensemble_cubes_dir(starless=False)
    os.makedirs(cubes)
    with open(os.path.join(cubes, "viz_index.json"), "w") as f:
        json.dump({"member_labels": ["00·psnr", "01·psnr"], "indices": []}, f)

    out = ev.job_archive_member(_Cap(), name="member_01")

    assert out["member"] == "member_01" and out["zip"].endswith(".zip")
    assert not os.path.isdir(os.path.join(base, "member_01"))   # dir gone
    # The detached regime cache is rebuilt in place, retaining only the active
    # member.  This preserves reusable inference instead of forcing a rerun.
    with open(os.path.join(cubes, "viz_index.json")) as f:
        assert json.load(f)["member_labels"] == ["00·psnr"]
    reg = er.load_registry(base)
    assert reg["active"] == ["member_00"]
    tomb = reg["archived"][0]
    assert tomb["name"] == "member_01" and tomb["zip"].endswith(".zip")
    zpath = os.path.join(default_store().current_dir, "models", out["zip"])
    with zipfile.ZipFile(zpath) as z:
        assert "checkpoint" in z.namelist()
    log = default_store().read_log()
    assert "member_01" in log                                   # campaign note
    # No SSH in the test env → the remote copy is flagged, not silently kept.
    assert "NOT deleted on FASRC" in out["remote"]
    assert "FASRC copy:" in log


def test_archive_member_deletes_fasrc_copy_when_connected(env, monkeypatch):
    from euclid_polish.web.helpers import ensemble_viz as ev
    from euclid_polish.web.remote import STATE

    base = ev.ensemble_dir()
    _mk_members(base, 0, 1)

    ran = {}

    class _FakeSSH:
        def is_connected(self):
            return True

        def run(self, cmd, timeout=0):
            ran["cmd"] = cmd
            return 0, "", ""

    class _Cfg:
        ckpt_dir = "/n/netscratch/lab/user/EuclidPolish/ckpt/wdsr"

    monkeypatch.setattr(STATE, "ssh", _FakeSSH())
    monkeypatch.setattr("euclid_polish.web.helpers.ensemble_viz.fasrc_config.load",
                        lambda: _Cfg())

    out = ev.job_archive_member(_Cap(), name="member_01")
    assert "deleted on FASRC" in out["remote"]
    assert ran["cmd"] == ("rm -rf /n/netscratch/lab/user/EuclidPolish/ckpt/"
                          "ensemble/member_01")


def test_remote_delete_refuses_unsafe_path(env, monkeypatch):
    from euclid_polish.web.helpers import ensemble_viz as ev
    from euclid_polish.web.remote import STATE

    class _FakeSSH:
        def is_connected(self):
            return True

        def run(self, cmd, timeout=0):  # pragma: no cover — must not be hit
            raise AssertionError("rm must not run on an unsafe path")

    class _Cfg:
        ckpt_dir = "ckpt/wdsr"          # relative → unsafe remote path

    monkeypatch.setattr(STATE, "ssh", _FakeSSH())
    monkeypatch.setattr("euclid_polish.web.helpers.ensemble_viz.fasrc_config.load",
                        lambda: _Cfg())
    assert "refused unsafe path" in ev._delete_remote_member("member_01")


def test_archive_member_rejects_bad_names(env):
    from euclid_polish.web.helpers import ensemble_viz as ev
    base = ev.ensemble_dir()
    _mk_members(base, 0)
    with pytest.raises(RuntimeError, match="invalid member name"):
        ev.job_archive_member(_Cap(), name="../../etc")
    with pytest.raises(RuntimeError, match="not an active"):
        ev.job_archive_member(_Cap(), name="member_09")


def test_archive_member_requires_campaign(env, monkeypatch):
    from euclid_polish.config import Config
    from euclid_polish.web.helpers import ensemble_viz as ev
    # Point tracking at a fresh root with NO campaign.
    monkeypatch.setattr(Config, "TRACKING_DIR", str(env / "tracking2"))
    base = ev.ensemble_dir()
    _mk_members(base, 3)
    with pytest.raises(RuntimeError, match="tracking campaign"):
        ev.job_archive_member(_Cap(), name="member_03")
