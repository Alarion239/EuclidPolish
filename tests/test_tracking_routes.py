"""Route-level tests for the tracking routes behind the Notebook workspace (euclid_polish.web.app)."""

from __future__ import annotations

import json
import os

import pytest

from euclid_polish.config import Config
from euclid_polish.tracking import default_store
from euclid_polish.tracking import timetravel as tt
from euclid_polish.web import remote
from euclid_polish.web.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_tracking_page_renders_when_empty(client):
    r = client.get("/notebook/log")
    assert r.status_code == 200
    assert b'id="root"' in r.data


def test_tracking_page_reachable_without_ssh(client, monkeypatch):
    # The Notebook log is local-first: it serves with SSH down.
    monkeypatch.setattr(remote.STATE, "ssh", None)
    r = client.get("/notebook/log")
    assert r.status_code == 200


def test_new_then_state_then_save(client):
    r = client.post("/api/tracking/new",
                    data={"title": "Route Run", "description": "via http"})
    assert r.status_code == 200 and r.get_json()["ok"]
    meta = r.get_json()["metadata"]
    assert meta["title"] == "Route Run"

    # Second create is rejected while one is active.
    r2 = client.post("/api/tracking/new", data={"title": "second"})
    assert r2.status_code == 400 and not r2.get_json()["ok"]

    # State reflects the active campaign.
    st = client.get("/api/tracking/state").get_json()
    assert st["active"]["title"] == "Route Run"

    # The React page reads the updated state through its JSON endpoint.
    assert client.get("/notebook/log").status_code == 200

    # Save → archived, no active.
    rs = client.post("/api/tracking/save")
    assert rs.status_code == 200 and rs.get_json()["ok"]
    st2 = client.get("/api/tracking/state").get_json()
    assert st2["active"] is None
    assert any(c["title"] == "Route Run" for c in st2["archived"])


def test_new_requires_title(client):
    r = client.post("/api/tracking/new", data={"title": "   "})
    assert r.status_code == 400
    assert "title" in r.get_json()["error"]


def test_log_append_and_replace(client):
    client.post("/api/tracking/new", data={"title": "notes"})
    r = client.post("/api/tracking/log",
                    data={"text": "first observation", "mode": "append"})
    assert r.status_code == 200
    assert "first observation" in r.get_json()["log_md"]

    r = client.post("/api/tracking/log",
                    data={"text": "# wiped\n", "mode": "replace"})
    assert r.get_json()["log_md"] == "# wiped\n"

    # empty append is rejected
    r = client.post("/api/tracking/log", data={"text": "  ", "mode": "append"})
    assert r.status_code == 400


def test_backup_fits_route(client, tmp_path, monkeypatch):
    # Make tmp_path an allowed root, then back up a file living under it.
    monkeypatch.setattr(Config, "DEFAULT_OUTPUT_DIR", str(tmp_path))
    src = tmp_path / "result.fits"
    src.write_bytes(b"SIMPLE = T" + b" " * 80)

    client.post("/api/tracking/new", data={"title": "bk"})
    r = client.post("/api/tracking/backup",
                    data={"kind": "fits", "path": str(src),
                          "comment": "the SR output", "name": "sr"})
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    assert j["ok"]
    assert j["record"]["kind"] == "fits"
    assert j["record"]["comment"] == "the SR output"
    # the synthetic null-SSH stub makes sync a no-op success
    assert "sync" in j

    st = client.get("/api/tracking/state").get_json()
    assert len(st["backups"]["fits"]) == 1


def test_backup_rejects_path_outside_roots(client, tmp_path, monkeypatch):
    # Allowed root is a *different* dir, so a file elsewhere is 403.
    monkeypatch.setattr(Config, "DEFAULT_OUTPUT_DIR", str(tmp_path / "allowed"))
    os.makedirs(Config.DEFAULT_OUTPUT_DIR, exist_ok=True)
    outside = tmp_path / "outside.fits"
    outside.write_bytes(b"x")
    client.post("/api/tracking/new", data={"title": "bk"})
    r = client.post("/api/tracking/backup",
                    data={"kind": "fits", "path": str(outside)})
    assert r.status_code == 403


def test_backup_unknown_kind(client):
    client.post("/api/tracking/new", data={"title": "bk"})
    r = client.post("/api/tracking/backup", data={"kind": "bogus"})
    assert r.status_code == 400


# --------------------------------------------------------------------------
# time-travel routes (heavy git/worktree/server bits stubbed)
# --------------------------------------------------------------------------

def test_timetravel_restore_route(client, monkeypatch):
    store = default_store()
    store.create_campaign("tt")
    mdir = os.path.join(store.current_dir, "models", "m1")
    os.makedirs(mdir)
    json.dump({"name": "m1", "kind": "model",
               "commit": {"hash": "abc123", "short": "abc123",
                          "branch": "main", "dirty": False}},
              open(os.path.join(mdir, "meta.json"), "w"))

    monkeypatch.setattr(tt, "prepare_local_sandbox",
                        lambda commit, **k: {"short": "abc123", "home": "/tmp/x",
                                             "root": "/tmp/x"})
    monkeypatch.setattr(tt, "write_home_fasrc_config",
                        lambda short, cfg: "/tmp/x/fasrc.json")
    monkeypatch.setattr(tt, "spawn_server",
                        lambda short, **k: {"ok": True, "port": 8766, "pid": 1,
                                            "url": "http://127.0.0.1:8766/"})

    r = client.post("/api/tracking/timetravel/restore",
                    data={"campaign": "current", "model": "m1"})
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    assert j["ok"] and j["short"] == "abc123"
    assert j["url"].endswith("8766/")
    assert j["warning"] is None        # commit was clean


def test_timetravel_restore_unknown_campaign(client):
    r = client.post("/api/tracking/timetravel/restore",
                    data={"campaign": "does-not-exist"})
    assert r.status_code == 400


def test_timetravel_stop_unknown(client):
    r = client.post("/api/tracking/timetravel/stop", data={"short": "zzz"})
    assert r.status_code == 400


def test_backup_model_bundles_training_log_plot(client, tmp_path, monkeypatch):
    # A tmp checkpoint dir (under its own ckpt root) with a plottable CSV.
    ckpt = tmp_path / "ckpt" / "wdsr"
    ckpt.mkdir(parents=True)
    (ckpt / "checkpoint").write_text('model_checkpoint_path: "ckpt-5"\n')
    (ckpt / "ckpt-5.index").write_bytes(b"i")
    (ckpt / "ckpt-5.data-00000-of-00001").write_bytes(b"w")
    header = ("step,wall_time,loss,"
              "psnr_stretched,psnr_raw,gnorm_avg,gnorm_max,clip_norm,"
              "duration_s,combined_loss,is_baseline")
    rows = "\n".join(
        f"{s},178051{s},0.04,46.{s},39.{s},1.4,160.0,5.0,135.0,0.003,"
        for s in (1000, 2000, 3000))
    (ckpt / "training_log.csv").write_text(header + "\n" + rows + "\n")
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(ckpt))

    client.post("/api/tracking/new", data={"title": "plots"})
    r = client.post("/api/tracking/backup",
                    data={"kind": "model", "comment": "with plot",
                          "name": "m1"})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()["ok"]

    bdir = default_store().model_backup_dir("current", "m1")
    assert os.path.isfile(os.path.join(bdir, "training_log.csv"))
    # The rendered plot is bundled alongside the checkpoint.
    assert os.path.isfile(os.path.join(bdir, "training_log.png"))


def test_backup_model_accepts_vis_only_sibling(client, tmp_path, monkeypatch):
    """A model backup must use the sibling ``-vis`` dir when the request
    passes it as ``ckpt_dir`` (the Notebook's backup dialog sends the chosen
    checkpoint dir this way; the VIS-only model and its toggle are retired)."""
    base = tmp_path / "ckpt" / "wdsr"
    visd = tmp_path / "ckpt" / "wdsr-vis"
    base.mkdir(parents=True)
    visd.mkdir(parents=True)
    # Only the -vis dir has checkpoint files — we're backing that one up.
    (visd / "checkpoint").write_text('model_checkpoint_path: "ckpt-7"\n')
    (visd / "ckpt-7.index").write_bytes(b"i")
    (visd / "ckpt-7.data-00000-of-00001").write_bytes(b"w")
    (visd / "training_log.csv").write_text("step\n7\n")
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(base))

    client.post("/api/tracking/new", data={"title": "visbk"})
    r = client.post("/api/tracking/backup",
                    data={"kind": "model", "comment": "vis model",
                          "name": "mvis", "ckpt_dir": str(visd)})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()["ok"]

    bdir = default_store().model_backup_dir("current", "mvis")
    # The -vis checkpoint files made it into the backup.
    assert os.path.isfile(os.path.join(bdir, "ckpt-7.index"))
    assert os.path.isfile(os.path.join(bdir, "training_log.csv"))


# --------------------------------------------------------------------------
# W-Ops: slim state, paginated compact jobs, archived-campaign detail
# --------------------------------------------------------------------------

_BLOB = "{" + "p" * 6000 + "}"


def _log_jobs(store, n):
    for i in range(n):
        store.log_fasrc_job({"jobid": str(100 + i), "label": f"job {i}", "step_id": "synthetic_generate",
                             "params": {"n_train": "10", "_star_prior_json": _BLOB}})


def test_state_no_longer_embeds_the_job_records(client):
    store = default_store()
    store.create_campaign("slim")
    _log_jobs(store, 3)
    st = client.get("/api/tracking/state").get_json()
    assert "jobs" not in st
    assert st["jobs_count"] == 3
    assert st["unassigned_count"] == 0


def test_jobs_endpoint_pages_and_strips_payload_blobs(client):
    store = default_store()
    store.create_campaign("paged")
    _log_jobs(store, 5)
    r = client.get("/api/tracking/jobs?offset=1&limit=2")
    assert r.status_code == 200
    body = r.get_json()
    assert body["total"] == 5 and body["offset"] == 1 and body["limit"] == 2
    assert [j["jobid"] for j in body["jobs"]] == ["103", "102"]      # newest first
    job = body["jobs"][0]
    assert job["params"] == {"n_train": "10"}
    assert job["params_omitted"] == {"_star_prior_json": len(_BLOB)}
    assert _BLOB not in r.get_data(as_text=True)


def test_jobs_endpoint_filters_and_reads_unassigned(client):
    store = default_store()
    store.log_fasrc_job({"jobid": "7", "label": "orphan eval"})
    body = client.get("/api/tracking/jobs?campaign=unassigned").get_json()
    assert [j["jobid"] for j in body["jobs"]] == ["7"]
    store.create_campaign("filter")
    _log_jobs(store, 3)
    body = client.get("/api/tracking/jobs?q=job%202").get_json()
    assert [j["jobid"] for j in body["jobs"]] == ["102"] and body["total"] == 1
    assert client.get("/api/tracking/jobs?campaign=nope").status_code == 404


def test_archived_campaign_detail(client):
    store = default_store()
    store.create_campaign("Detail me", "desc")
    store.append_log("a result worth keeping")
    _log_jobs(store, 2)
    res = store.save_campaign()
    name = os.path.basename(res["archive_path"])
    r = client.get(f"/api/tracking/campaign/{name}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["metadata"]["title"] == "Detail me"
    assert "a result worth keeping" in body["log_md"]
    assert body["jobs_count"] == 2
    assert set(body["backups"]) == {"models", "fits", "images"}
    assert body["dir"] == name
    assert client.get("/api/tracking/campaign/missing").status_code == 404


def test_timetravel_restore_from_a_zipped_model_uses_its_commit(client, monkeypatch):
    store = default_store()
    store.create_campaign("zips")
    models = os.path.join(store.current_dir, "models")
    with open(os.path.join(models, "ensemble-member-01.zip"), "wb") as fp:
        fp.write(b"PK")
    json.dump({"name": "ensemble-member-01.zip", "kind": "model-zip",
               "commit": {"hash": "def456", "short": "def456", "branch": "main", "dirty": True}},
              open(os.path.join(models, "ensemble-member-01.zip.meta.json"), "w"))
    seen = {}

    def fake_prepare(commit, **k):
        seen.update(k, commit=commit)
        return {"short": "def456", "home": "/tmp/x", "root": "/tmp/x"}

    monkeypatch.setattr(tt, "prepare_local_sandbox", fake_prepare)
    monkeypatch.setattr(tt, "write_home_fasrc_config", lambda short, cfg: "/tmp/x/fasrc.json")
    monkeypatch.setattr(tt, "spawn_server", lambda short, **k: {"ok": True, "url": "http://127.0.0.1:8766/"})
    r = client.post("/api/tracking/timetravel/restore",
                    data={"campaign": "current", "model": "ensemble-member-01.zip"})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert seen["commit"] == "def456"
    assert seen["seed_ckpt_dir"] is None          # a zip is not a live checkpoint
    assert r.get_json()["warning"]                # dirty commit → reproducibility warning


@pytest.mark.parametrize("action", ["open", "stop", "remove"])
@pytest.mark.parametrize("short", ["", ".", "..", "../x", "abc/def", "abc1234/..",
                                   "ABCD\x00", "beef00d"])
def test_timetravel_refuses_bad_or_unknown_sandbox_ids(client, monkeypatch, action, short):
    """An empty or '.' id used to name the time-travel ROOT (remove → rmtree of
    every sandbox); only an id of an existing sandbox reaches timetravel."""
    calls = []
    monkeypatch.setattr(tt, "list_sandboxes", lambda: [{"short": "abc1234"}])
    for name in ("spawn_server", "stop_server", "remove_sandbox"):
        monkeypatch.setattr(tt, name, lambda s, **k: calls.append(s) or {"ok": True})
    r = client.post(f"/api/tracking/timetravel/{action}", data={"short": short})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False and r.get_json()["error"]
    assert calls == []


@pytest.mark.parametrize("action,target", [("open", "spawn_server"), ("stop", "stop_server"),
                                           ("remove", "remove_sandbox")])
def test_timetravel_acts_on_a_listed_sandbox(client, monkeypatch, action, target):
    calls = []
    monkeypatch.setattr(tt, "list_sandboxes", lambda: [{"short": "abc1234"}])
    monkeypatch.setattr(tt, target, lambda s, **k: calls.append(s) or {"ok": True})
    r = client.post(f"/api/tracking/timetravel/{action}", data={"short": "abc1234"})
    assert r.status_code == 200 and calls == ["abc1234"]


def test_jobs_endpoint_lists_every_job_id_of_a_campaign(client):
    """``?ids=1``: the whole campaign's job ids (unpaged, deduped, newest
    first) for Runs › History's campaign filter, without the records."""
    store = default_store()
    store.create_campaign("ids")
    _log_jobs(store, 3)
    store.log_fasrc_job({"jobid": "101", "label": "logged twice"})
    body = client.get("/api/tracking/jobs?ids=1&limit=1").get_json()
    assert body["ok"] and body["campaign"] == "current"
    assert body["jobids"] == ["101", "102", "100"]
    assert body["total"] == 3
    assert "jobs" not in body and _BLOB not in json.dumps(body)
    assert client.get("/api/tracking/jobs?ids=1&campaign=nope").status_code == 404


def _stub_timetravel(monkeypatch, seen):
    def fake_prepare(commit, **k):
        seen.update(k, commit=commit)
        return {"short": commit[:7], "home": "/tmp/x", "root": "/tmp/x"}

    monkeypatch.setattr(tt, "prepare_local_sandbox", fake_prepare)
    monkeypatch.setattr(tt, "write_home_fasrc_config", lambda short, cfg: "/tmp/x/fasrc.json")
    monkeypatch.setattr(tt, "spawn_server", lambda short, **k: {"ok": True, "url": "http://127.0.0.1:8766/"})


def test_timetravel_restore_from_a_fits_backup_uses_its_commit(client, tmp_path, monkeypatch):
    """A FITS (or image) backup time-travels to the commit it was saved at."""
    monkeypatch.setattr(Config, "DEFAULT_OUTPUT_DIR", str(tmp_path))
    src = tmp_path / "result.fits"
    src.write_bytes(b"SIMPLE = T" + b" " * 80)
    client.post("/api/tracking/new", data={"title": "fits tt"})
    rec = client.post("/api/tracking/backup", data={"kind": "fits", "path": str(src), "name": "sr"}).get_json()["record"]
    seen: dict = {}
    _stub_timetravel(monkeypatch, seen)
    commit = (rec.get("commit") or {}).get("hash")
    r = client.post("/api/tracking/timetravel/restore",
                    data={"campaign": "current", "backup": rec["name"], "kind": "fits"})
    if not commit:      # a checkout without git: the route says so
        assert r.status_code == 400 and "no git commit" in r.get_json()["error"]
        return
    assert r.status_code == 200, r.get_data(as_text=True)
    assert seen["commit"] == commit
    assert seen["seed_ckpt_dir"] is None
    assert seen["source"] == {"campaign": "current", "model": None, "backup": rec["name"], "kind": "fits"}


def test_timetravel_restore_refuses_an_unknown_backup(client):
    client.post("/api/tracking/new", data={"title": "nobackup"})
    r = client.post("/api/tracking/timetravel/restore",
                    data={"campaign": "current", "backup": "missing", "kind": "image"})
    assert r.status_code == 400
    assert "no image backup" in r.get_json()["error"]
    r = client.post("/api/tracking/timetravel/restore",
                    data={"campaign": "current", "backup": "x", "kind": "model"})
    assert r.status_code == 400
