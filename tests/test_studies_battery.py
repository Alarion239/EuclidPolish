"""Battery studies (2026-11-02): a finished model-battery run frozen as a
study — its results, report and figures copied, hashed and sealed; smoke runs,
runs without figures and missing files refused with nothing left behind; the
list and the study endpoint answer the battery's models and numbers."""
from __future__ import annotations

import json

import pytest

from euclid_polish.config import Config
from euclid_polish.studies import battery
from euclid_polish.studies.store import StudyError, StudyStore
from euclid_polish.web import app as web_app

MODELS = [{"spec": "milestone:member_259@200000", "name": "259@200k", "kind": "milestone",
           "member": "member_259", "step": 200000, "fingerprint": "ckpt-200000:1:2",
           "member_dir": "data/x"},
          {"spec": "universal:member_257", "name": "257", "kind": "universal",
           "member": "member_257", "step": None, "fingerprint": "ckpt-116:1:3",
           "member_dir": "ckpt/ensemble/member_257"}]


def _run(tmp_path, **meta):
    out = tmp_path / "battery"
    (out / "figures").mkdir(parents=True)
    (out / "figures" / "kron_vis.png").write_bytes(b"\x89PNG kron")
    (out / "figures" / "curve_of_growth.png").write_bytes(b"\x89PNG growth")
    (out / "battery.md").write_text("# battery\n", encoding="utf-8")
    payload = {"meta": {"created": "2026-11-02 13:25", "models": MODELS, "tests": ["synthetic", "figures"],
                        "limits": "", "records_fp": "dirty:1|hr:2", "records_dir": "/records/v6",
                        "figures": ["figures/kron_vis.png", "figures/curve_of_growth.png"], **meta},
               "table": [["Synthetic score", "∫PSNR, mean of 4 bands (dB)", ["59.98", "60.61"]]],
               "paired": [["Grid", "real sky", ["+0.9 [+0.5, +1.4]"]]]}
    (out / "battery.json").write_text(json.dumps(payload), encoding="utf-8")
    return out


def test_freeze_copies_hashes_and_seals_the_run(tmp_path):
    store = StudyStore(tmp_path / "studies")
    study_id = battery.freeze(str(_run(tmp_path)), name="Steps on v6", note="259 vs 257", store=store)
    manifest = store.manifest(study_id)
    assert manifest["complete"] and manifest["kind"] == "battery"
    assert [m["name"] for m in manifest["battery"]["models"]] == ["259@200k", "257"]
    assert "member_dir" not in manifest["battery"]["models"][0]
    assert manifest["battery"]["records_fp"] == "dirty:1|hr:2"
    assert manifest["battery"]["figures"] == ["figure-kron_vis.png", "figure-curve_of_growth.png"]
    assert set(manifest["numbers"]) == {"battery.json", "battery.md", "figure-kron_vis.png",
                                        "figure-curve_of_growth.png"}
    assert store.read_number(study_id, "figure-kron_vis.png") == b"\x89PNG kron"
    summary = store.summary(manifest)
    assert summary["kind"] == "battery" and summary["members"] == 2 and summary["models"] == ["259@200k", "257"]
    with pytest.raises(StudyError):
        store.write_number(study_id, "battery.md", b"changed")


@pytest.mark.parametrize("meta, missing", [({"limits": "smoke"}, None), ({"figures": []}, None),
                                           ({}, "figures/kron_vis.png"), ({"models": []}, None)])
def test_freeze_refuses_and_leaves_nothing(tmp_path, meta, missing):
    run = _run(tmp_path, **meta)
    if missing:
        (run / missing).unlink()
    store = StudyStore(tmp_path / "studies")
    with pytest.raises(StudyError):
        battery.freeze(str(run), name="x", store=store)
    assert store.list() == []


def test_the_study_endpoint_answers_the_battery(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "TRACKING_DIR", str(tmp_path / "tracking"))
    study_id = battery.freeze(str(_run(tmp_path)), name="Steps on v6")
    app = web_app.create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        listed = client.get("/api/studies").get_json()["studies"]
        assert [(s["id"], s["kind"], s["members"]) for s in listed] == [(study_id, "battery", 2)]
        detail = client.get(f"/api/studies/{study_id}").get_json()
        assert detail["charts"] == []
        assert detail["numbers"]["battery"]["table"][0][2] == ["59.98", "60.61"]
        figure = client.get(f"/api/studies/{study_id}/numbers/figure-kron_vis.png")
        assert figure.status_code == 200 and figure.mimetype == "image/png"
        report = client.get(f"/api/studies/{study_id}/numbers/battery.md")
        assert report.mimetype == "text/markdown" and report.data == b"# battery\n"
