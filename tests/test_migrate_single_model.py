"""migrate_single_model: zip legacy single-model ckpts into tracking, delete."""
from __future__ import annotations

import os


def test_migrate_zips_and_deletes(tmp_path, monkeypatch):
    from euclid_polish.config import Config
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR",
                        str(tmp_path / "ckpt/wdsr"))
    monkeypatch.setattr(Config, "TRACKING_DIR", str(tmp_path / "tracking"))
    for d in ("ckpt/wdsr", "ckpt/wdsr-vis"):
        p = tmp_path / d
        p.mkdir(parents=True)
        (p / "checkpoint").write_text("x")

    from scripts.migrate_single_model import migrate
    out = migrate()                       # creates a campaign if none active
    assert not (tmp_path / "ckpt/wdsr").exists()
    assert not (tmp_path / "ckpt/wdsr-vis").exists()
    assert len(out["archived"]) == 2
    from euclid_polish.tracking import default_store
    store = default_store()
    names = {m["name"] for m in store.list_backups()["models"]}
    assert out["archived"][0] in names
    zpath = os.path.join(store.current_dir, "models", out["archived"][0])
    assert os.path.isfile(zpath) and zpath.endswith(".zip")
    assert "FASRC" in store.read_log()    # remote-cleanup reminder logged

    out2 = migrate()                      # idempotent
    assert out2["archived"] == []
