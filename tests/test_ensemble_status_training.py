"""ensemble_status: per-member step (log tail) + origin.json provenance."""
from __future__ import annotations

import json
import os


def _member(base, name):
    d = os.path.join(base, name)
    os.makedirs(d)
    with open(os.path.join(d, "checkpoint"), "w") as f:
        f.write("x")
    return d


def test_status_reports_step_and_origin(tmp_path, monkeypatch):
    from euclid_polish.config import Config
    from euclid_polish.web.helpers import ensemble_viz as ev
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR",
                        str(tmp_path / "ckpt/wdsr"))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    d = _member(ev.ensemble_dir(), "member_00")
    header = ("step,wall_time,loss,psnr_stretched,psnr_raw,"
              "save_best_score,combined_loss,is_baseline\n")
    with open(os.path.join(d, "training_log.csv"), "w") as f:
        f.write(header + "1000,1,0.1,40,33,40,0.01,\n"
                       + "2000,2,0.09,41,34,41,0.009,\n")
    with open(os.path.join(d, "origin.json"), "w") as f:
        json.dump({"op": "fork", "forked_from": "member_09·psnr"}, f)

    st = ev.ensemble_status()
    (m,) = st["members"]
    assert m["step"] == 2000
    assert m["origin"]["forked_from"] == "member_09·psnr"


def test_status_step_blank_when_no_log(tmp_path, monkeypatch):
    from euclid_polish.config import Config
    from euclid_polish.web.helpers import ensemble_viz as ev
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR",
                        str(tmp_path / "ckpt/wdsr"))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    _member(ev.ensemble_dir(), "member_00")
    st = ev.ensemble_status()
    assert st["members"][0]["step"] is None
    assert st["members"][0]["origin"] is None
