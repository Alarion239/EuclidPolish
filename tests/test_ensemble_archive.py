"""job_archive_member: zip → tracking, tombstone, delete, mark stale; a
restore before the next evaluation takes the member off that queue."""
from __future__ import annotations

import json
import os
import zipfile

import pytest

from euclid_polish.web.helpers import purge_requests


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
    # Archiving is deliberately cheap: it leaves cached cubes untouched and
    # queues their rebuild for the next evaluation of the affected regime.
    with open(os.path.join(cubes, "viz_index.json")) as f:
        assert json.load(f)["member_labels"] == ["00·psnr", "01·psnr"]
    assert ev._pending_archived_members(False) == ["member_01"]
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
    assert "marked stale" in log
    # The cubes themselves go in the follow-up purge job.
    assert purge_requests.read_pending()["reasons"] == ["archived member_01"]


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


def test_next_evaluation_consumes_queued_archive_from_cached_cubes(env, monkeypatch):
    """The next Evaluate, not Archive, performs the cached-cube rebuild."""
    from euclid_polish.web.helpers import ensemble_viz as ev

    records = env / "records"
    records.mkdir()
    (records / "dirty_test").write_text("")
    (records / "hr_test").write_text("")
    monkeypatch.setattr(ev, "_sky_records_local_dir", lambda: str(records))
    monkeypatch.setattr(ev, "eval_subset", lambda _rdir: "test")
    monkeypatch.setattr(ev, "tfrecord_path",
                        lambda root, name: os.path.join(root, name))
    monkeypatch.setattr(ev, "_pending_archived_members",
                        lambda starless: ["member_01"])
    rebuilt = []
    monkeypatch.setattr(ev, "_rebuild_pending_archive_caches",
                        lambda starless: rebuilt.append(starless) or True)
    # The rebuilt cache is complete and current for the remaining members.
    monkeypatch.setattr(ev, "_bucket_current", lambda *a: True)
    monkeypatch.setattr(ev, "_reevaluate_from_cached_cubes",
                        lambda starless, num_images: {
                            "regime": "starfull", "recomputed_from_cubes": True})
    cleared = []
    monkeypatch.setattr(ev, "_clear_archive_stale",
                        lambda starless: cleared.append(starless))
    monkeypatch.setattr(ev, "evaluate_on_records",
                        lambda *a, **k: pytest.fail("must reuse cached cubes"))

    out = ev.job_ensemble_evaluate(_Cap(), num_images=100, starless=False)

    assert rebuilt == [False]
    assert cleared == [False]
    assert out["recomputed_from_archives"] == ["member_01"]


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


def test_restore_takes_the_member_off_the_pending_archive_queue(env, monkeypatch):
    """A member restored before the next Evaluate is no longer queued as
    archived; members still archived stay queued."""
    from euclid_polish.web.helpers import ensemble_viz as ev
    from euclid_polish.web.remote import STATE

    monkeypatch.setattr(STATE, "ssh", None)
    base = ev.ensemble_dir()
    _mk_members(base, 0, 1, 2)
    ev.job_archive_member(_Cap(), name="member_01")
    ev.job_archive_member(_Cap(), name="member_02")
    assert ev._pending_archived_members(False) == ["member_01", "member_02"]

    ev.job_restore_member(_Cap(), name="member_02")

    assert ev._pending_archived_members(False) == ["member_01"]


@pytest.fixture
def cube_env(tmp_path, monkeypatch):
    from euclid_polish.tracking import default_store
    from euclid_polish.web.remote import STATE
    from tests._ensemble_cube_cache_fixtures import make_env

    env = make_env(tmp_path, monkeypatch)
    default_store().create_campaign("archive test")
    monkeypatch.setattr(STATE, "ssh", None)
    return env


def test_evaluate_after_archive_then_restore_keeps_the_member_cached(cube_env):
    """Archive → restore → Evaluate: the restored member's cached cubes stay in
    both buckets (the queued archive must not drop them)."""
    from euclid_polish.config import Config
    from euclid_polish.eval.ensemble_cube_cache import member_cube_path
    from euclid_polish.web.helpers import ensemble_viz as ev
    from tests._ensemble_cube_cache_fixtures import N_FIELDS, Cap

    ev.job_ensemble_evaluate(Cap(), num_images=N_FIELDS, starless=False)
    ev._prepare_validate_cubes(Cap(), starless=False, num_images=N_FIELDS,
                               target_fwhm=Config.TARGET_PSF_FWHM_ARCSEC)

    ev.job_archive_member(Cap(), name="member_03")
    ev.job_restore_member(Cap(), name="member_03")
    out = ev.job_ensemble_evaluate(Cap(), num_images=N_FIELDS, starless=False)

    assert "recomputed_from_archives" not in out
    assert out["member_labels"] == cube_env["labels"]
    with open(cube_env["validate"] / "viz_index.json") as f:
        assert json.load(f)["member_labels"] == cube_env["labels"]
    assert all(os.path.isfile(member_cube_path(str(cube_env["validate"]), "03·psnr", rec))
               for rec in range(N_FIELDS))
