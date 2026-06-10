"""Route-level tests for the Tracking tab (euclid_polish.web.app)."""

from __future__ import annotations

import os

import pytest

from euclid_polish.config import Config
from euclid_polish.web.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_tracking_page_renders_when_empty(client):
    r = client.get("/tracking")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Start a campaign" in body


def test_tracking_page_reachable_without_ssh(client, monkeypatch):
    # The gate must NOT redirect /tracking even when SSH is down.
    from euclid_polish.web import remote
    monkeypatch.setattr(remote.STATE, "ssh", None)
    r = client.get("/tracking")
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

    # The page now shows the title + the Save button.
    body = client.get("/tracking").get_data(as_text=True)
    assert "Route Run" in body and "Save &amp; start new" in body

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
    import json
    import os
    from euclid_polish.tracking import default_store
    from euclid_polish.tracking import timetravel as tt

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
    header = ("step,wall_time,loss,loss_syn,loss_hst,loss_anchor,"
              "psnr_stretched,psnr_raw,gnorm_avg,gnorm_max,clip_norm,"
              "duration_s,psnr_stretched_hst,psnr_raw_hst,anchor_val_psnr,"
              "save_best_score,combined_loss,is_baseline")
    rows = "\n".join(
        f"{s},178051{s},0.04,0.04,,,46.{s},39.{s},1.4,160.0,5.0,135.0,,,,"
        f"46.{s},0.003,"
        for s in (1000, 2000, 3000))
    (ckpt / "training_log.csv").write_text(header + "\n" + rows + "\n")
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(ckpt))

    client.post("/api/tracking/new", data={"title": "plots"})
    r = client.post("/api/tracking/backup",
                    data={"kind": "model", "comment": "with plot",
                          "name": "m1"})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()["ok"]

    from euclid_polish.tracking import default_store
    bdir = default_store().model_backup_dir("current", "m1")
    assert os.path.isfile(os.path.join(bdir, "training_log.csv"))
    # The rendered plot is bundled alongside the checkpoint.
    assert os.path.isfile(os.path.join(bdir, "training_log.png"))


def test_backup_model_accepts_vis_only_sibling(client, tmp_path, monkeypatch):
    """Tracking the VIS-only model must back up the sibling ``-vis`` dir when
    the request passes it as ``ckpt_dir`` (the training page sends this when
    the VIS-only toggle is on)."""
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

    from euclid_polish.tracking import default_store
    bdir = default_store().model_backup_dir("current", "mvis")
    # The -vis checkpoint files made it into the backup.
    assert os.path.isfile(os.path.join(bdir, "ckpt-7.index"))
    assert os.path.isfile(os.path.join(bdir, "training_log.csv"))
