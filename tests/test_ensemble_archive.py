"""job_archive_member: zip → tracking, tombstone, delete member dir, purge cache."""
from __future__ import annotations

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
    from euclid_polish.config import Config
    from euclid_polish.tracking import default_store
    from euclid_polish.web.helpers import ensemble_viz as ev

    base = ev.ensemble_dir()
    _mk_members(base, 0, 1)
    cubes = os.path.join(Config.VIS_DIR, "ensemble", "cubes")
    os.makedirs(cubes)
    with open(os.path.join(cubes, "viz_index.json"), "w") as f:
        f.write("{}")

    out = ev.job_archive_member(_Cap(), name="member_01")

    assert out["member"] == "member_01" and out["zip"].endswith(".zip")
    assert not os.path.isdir(os.path.join(base, "member_01"))   # dir gone
    assert not os.path.isdir(cubes)                             # eager purge
    reg = er.load_registry(base)
    assert reg["active"] == ["member_00"]
    tomb = reg["archived"][0]
    assert tomb["name"] == "member_01" and tomb["zip"].endswith(".zip")
    zpath = os.path.join(default_store().current_dir, "models", out["zip"])
    with zipfile.ZipFile(zpath) as z:
        assert "checkpoint" in z.namelist()
    assert "member_01" in default_store().read_log()            # campaign note


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
