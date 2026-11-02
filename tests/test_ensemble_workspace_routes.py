"""HTTP contract of the Models workspace routes (routes/ensemble.py): the
new read endpoints, the job endpoints' validation and the knobs they hand
their jobs (spawn stubbed — nothing runs)."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from euclid_polish import ensemble_registry
from euclid_polish.config import Config
from euclid_polish.observability import JobLog, JobRecord
from euclid_polish.web import fasrc_jobs
from euclid_polish.web.app import create_app
from euclid_polish.web.helpers import model_catalog
from euclid_polish.web.routes import ensemble as routes


class _Cap:
    def tick(self, *_a, **_k):
        pass


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(tmp_path / "ckpt/wdsr"))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def spawned(monkeypatch):
    """Capture spawned jobs: run their target against a recording stub."""
    calls: list[dict] = []

    def fake_spawn(label, target, kind=None):
        calls.append({"label": label, "target": target, "kind": kind})
        return f"job{len(calls)}"

    monkeypatch.setattr(routes.REGISTRY, "spawn", fake_spawn)
    return calls


def _capture(monkeypatch, name):
    seen: dict = {}

    def fake(_cap, **kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(routes, name, fake)
    return seen


def test_read_endpoints_answer_on_an_empty_ensemble(client):
    for url in ("/ensemble/members.json", "/ensemble/training-jobs.json",
                "/ensemble/training-curves.json", "/ensemble/universal.json",
                "/ensemble/universal/spectra.json", "/ensemble/universal/milestones.json"):
        r = client.get(url)
        assert r.status_code == 200, url
    assert client.get("/ensemble/members.json").get_json()["members"] == []
    assert client.get("/ensemble/members.json").get_json()["universal"] == []
    universal = client.get("/ensemble/universal.json").get_json()
    assert universal["available"] is False and universal["models"] == []
    spectra = client.get("/ensemble/universal/spectra.json").get_json()
    assert spectra["models"] == [] and spectra["missing"] == []
    assert universal["fasrc"] == [] and "combiners" not in universal


@pytest.mark.parametrize(("method", "url"), [
    ("GET", "/ensemble/status.json"), ("GET", "/ensemble/overview.json"),
    ("POST", "/ensemble/evaluate"), ("GET", "/ensemble/combiners.json"),
    ("POST", "/ensemble/combiners/compare"), ("GET", "/ensemble/combiners/compare.json"),
    ("POST", "/ensemble/combiners/fit"), ("POST", "/ensemble/combiners/promote"),
    ("POST", "/ensemble/combiners/archive"), ("POST", "/ensemble/combiners/restore"),
    ("POST", "/ensemble/knee-psnr"), ("GET", "/ensemble/knee-psnr.json"),
    ("POST", "/ensemble/member-psnr"), ("GET", "/ensemble/evals.json"),
    ("POST", "/ensemble/evals/bands"), ("GET", "/ensemble/pixel-trace.json"),
    ("POST", "/ensemble/pull"), ("GET", "/api/inference/field.json"),
    ("GET", "/api/inference/diagnostics.json"), ("POST", "/inference/refresh-combiners"),
    ("GET", "/api/system/production"),
])
def test_the_removed_endpoints_are_gone(client, spawned, method, url):
    r = client.open(url, method=method, headers={"Sec-Fetch-Site": "same-origin"})
    assert r.status_code in (404, 405), (method, url, r.status_code)
    assert spawned == []


def test_member_json_404_and_400_are_json(client):
    r = client.get("/ensemble/member/member_07.json")
    assert r.status_code == 404 and "member_07" in r.get_json()["error"]
    r = client.get("/ensemble/member/bogus.json")
    assert r.status_code == 400 and r.get_json()["ok"] is False


def test_errors_under_ensemble_are_json(client):
    r = client.get("/ensemble/nothing-here.json")
    assert r.status_code == 404 and "error" in r.get_json()


def test_restore_validates_member_names(client, spawned, monkeypatch):
    seen_restore = _capture(monkeypatch, "job_restore_member")
    assert client.post("/ensemble/restore-member", data={"member": "x"}).status_code == 400
    assert client.post("/ensemble/restore-member", data={"member": "7"}).status_code == 200
    spawned[0]["target"](_Cap())
    assert seen_restore == {"name": "member_07"}


def _ledger_universal(tmp_path, monkeypatch, *names):
    monkeypatch.setattr(fasrc_jobs, "JOBLOG", JobLog(str(tmp_path / "joblog.csv")))
    spec = {"asinh_knees": [0.1, 1, 10], "output_knee": 10, "learn_output_knee": True}
    for i, name in enumerate(names):
        fasrc_jobs.JOBLOG.record_submission(JobRecord(
            jobid=str(100 + i), step_id="ensemble_train", label="Train",
            submitted_at="2026-10-12T00:00:00Z", state="COMPLETED",
            params_json=json.dumps({"mode": "add", "member_names": name,
                                    "member_spec": json.dumps([spec])})))


def test_training_curves_count_registered_universal_models_as_local(client, monkeypatch,
                                                                     tmp_path):
    base = str(tmp_path / "ckpt" / "ensemble")
    ensemble_registry.add_universal_entry(base, "216")
    _ledger_universal(tmp_path, monkeypatch, "member_216", "member_217")
    seen: dict = {}

    def fake_remote(local, archived, *, universal_names=()):
        seen.update(local=list(local), archived=list(archived),
                    universal=sorted(universal_names))
        return []

    monkeypatch.setattr(routes, "remote_training_curves", fake_remote)
    assert client.get("/ensemble/training-curves.json").status_code == 200
    assert seen == {"local": ["member_216"], "archived": [],
                    "universal": ["member_216", "member_217"]}


def test_train_preview_answers_names_or_the_refusal(client, tmp_path):
    base = tmp_path / "ckpt" / "ensemble" / "member_03"
    base.mkdir(parents=True)
    (base / "checkpoint").write_text("x")
    r = client.post("/ensemble/train/preview", data={"mode": "add", "count": "1", "steps": "1000"})
    assert r.status_code == 200 and r.get_json()["member_names"] == ["member_04"]
    r = client.post("/ensemble/train/preview", data={"mode": "continue"})
    assert r.status_code == 400 and r.get_json()["ok"] is False
    assert os.path.isdir(base)


def test_archive_member_validates_the_name_before_spawning(client, spawned, monkeypatch):
    """A destructive job: a bad or inactive name is a 400 up front (not a
    job id followed by a failed job), like restore-member."""
    monkeypatch.setattr(routes, "_active_member_names", lambda: {"member_01"})
    for bad in ("", "../../etc", "member_1/../x", "01·loss"):
        r = client.post("/ensemble/archive-member", data={"member": bad})
        assert r.status_code == 400 and r.get_json()["ok"] is False, bad
    inactive = client.post("/ensemble/archive-member", data={"member": "member_09"})
    assert inactive.status_code == 400 and "not an active" in inactive.get_json()["error"]
    assert spawned == []
    seen = _capture(monkeypatch, "job_archive_member")
    ok = client.post("/ensemble/archive-member", data={"member": "1·psnr"})
    assert ok.status_code == 200 and ok.get_json() == {"ok": True, "job_id": "job1"}
    spawned[0]["target"](_Cap())
    assert seen == {"name": "member_01"}
    # a registered universal model archives too
    monkeypatch.setattr(routes, "_universal_member_names", lambda: {"member_09"})
    universal = client.post("/ensemble/archive-member", data={"member": "9"})
    assert universal.status_code == 200
    spawned[-1]["target"](_Cap())
    assert seen == {"name": "member_09"}


@pytest.fixture
def exclusive(monkeypatch):
    """Capture single-run jobs (``REGISTRY.spawn_exclusive``)."""
    calls: list[dict] = []

    def fake(label, target, *, kind, key=None, per_key=False):
        calls.append({"label": label, "target": target, "kind": kind})
        return SimpleNamespace(job_id=f"job{len(calls)}"), True

    monkeypatch.setattr(routes.REGISTRY, "spawn_exclusive", fake)
    return calls


def _universal_member(tmp_path, n, origin):
    d = tmp_path / "ckpt" / "ensemble" / f"member_{n:02d}"
    d.mkdir(parents=True)
    (d / "checkpoint").write_text('model_checkpoint_path: "ckpt-1"\n')
    (d / "ckpt-1.index").write_bytes(b"x")
    (d / "origin.json").write_text(json.dumps(origin))
    return d


_UNIVERSAL = {"asinh_knees": [0.1, 1.0, 10.0], "learned_output_knee": {"init_e": 10.0}}


def test_universal_score_validates_the_models_and_passes_knobs(client, exclusive, monkeypatch,
                                                              tmp_path):
    monkeypatch.setattr(model_catalog, "PANEL_FIRST_MODEL", 0)     # 211 predates the panel's 257
    base = str(tmp_path / "ckpt" / "ensemble")
    ensemble_registry.add_universal_entry(base, "211")
    _universal_member(tmp_path, 211, _UNIVERSAL)
    _universal_member(tmp_path, 170, {"asinh_knee": 10.0})
    seen = _capture(monkeypatch, "job_universal_score")
    assert client.post("/ensemble/universal/score", data={"members": "bad!"}).status_code == 400
    r = client.post("/ensemble/universal/score", data={"members": "170"})
    assert r.status_code == 400 and "member_170" in r.get_json()["error"]
    r = client.post("/ensemble/universal/score", data={"members": "211·psnr", "force": "1"})
    assert r.get_json() == {"ok": True, "job_id": "job1", "already_running": False}
    exclusive[0]["target"](_Cap())
    assert seen == {"members": ["member_211"], "force": True, "spectra": False}
    assert exclusive[0]["kind"] == "universal-score"
    seen.clear()
    assert client.post("/ensemble/universal/score").status_code == 200
    exclusive[1]["target"](_Cap())
    assert seen == {"members": None, "force": False, "spectra": False}
    seen.clear()
    assert client.post("/ensemble/universal/score", data={"spectra": "1"}).status_code == 200
    exclusive[2]["target"](_Cap())
    assert seen == {"members": None, "force": False, "spectra": True}
    assert "spectra" in exclusive[2]["label"]


def test_universal_score_refuses_a_model_before_the_first_model(client, exclusive, monkeypatch,
                                                                tmp_path):
    """A registered universal model numbered below PANEL_FIRST_MODEL (257)
    is no score target: 400 before spawning; 257 is scored."""
    base = str(tmp_path / "ckpt" / "ensemble")
    for n in (211, 257):
        ensemble_registry.add_universal_entry(base, str(n))
        _universal_member(tmp_path, n, _UNIVERSAL)
    _capture(monkeypatch, "job_universal_score")
    r = client.post("/ensemble/universal/score", data={"members": "211"})
    assert r.status_code == 400 and "member_211" in r.get_json()["error"]
    assert exclusive == []
    assert client.post("/ensemble/universal/score", data={"members": "257"}).status_code == 200
    names = [m["name"] for m in client.get("/ensemble/universal.json").get_json()["models"]]
    assert names == ["member_257"]


def test_universal_spectra_map_404s_are_json(client, tmp_path):
    for url in ("/ensemble/universal/spectra/bogus!.json",
                "/ensemble/universal/spectra/member_07.json"):
        r = client.get(url)
        assert r.status_code == 404 and r.get_json()["ok"] is False, url
    assert "member_07" in client.get("/ensemble/universal/spectra/member_07.json").get_json()["error"]


def test_universal_pull_needs_names_and_refuses_ensemble_members(client, exclusive, monkeypatch,
                                                                 tmp_path):
    base = str(tmp_path / "ckpt" / "ensemble")
    _universal_member(tmp_path, 196, _UNIVERSAL)
    _universal_member(tmp_path, 150, {"asinh_knee": 10.0})
    ensemble_registry.load_registry(base)
    ensemble_registry.archive_member_entry(base, "member_150", zip_path="models/z.zip",
                                           commit=None)
    seen = _capture(monkeypatch, "job_universal_pull")
    gate = tmp_path / "vis" / "ensemble" / "starfull" / "spatial_gate_combiner"
    gate.mkdir(parents=True)
    (gate / "combiner.json").write_text(json.dumps({"member_labels": ["195·psnr"]}))
    for data, needle in (({}, "members"), ({"members": "bad!"}, "bad!"),
                         ({"members": "196"}, "active"), ({"members": "150"}, "archived"),
                         ({"members": "195"}, "gate")):
        r = client.post("/ensemble/universal/pull", data=data)
        assert r.status_code == 400 and needle in r.get_json()["error"], data
    assert exclusive == []
    r = client.post("/ensemble/universal/pull", data={"members": "211,member_212,211"})
    assert r.status_code == 200 and r.get_json()["job_id"] == "job1"
    exclusive[0]["target"](_Cap())
    assert seen == {"members": ["member_211", "member_212"]}
    assert exclusive[0]["kind"] == "universal-pull"
