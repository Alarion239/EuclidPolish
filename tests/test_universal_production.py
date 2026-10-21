"""Production as one universal model: the registry's production pointer,
the model catalogue's ``production`` spec, the evaluators' model and its
identity, the checks that name it, the Universal tab's reference, and the
archive / restore of every spatial gate once production is no gate."""
from __future__ import annotations

import json
import os
import shutil
import zipfile

import numpy as np
import pytest

from euclid_polish import ensemble_registry as er
from euclid_polish.config import Config
from euclid_polish.ensemble import member_fingerprint
from euclid_polish.eval import catalog_runner, ensemble_infer
from euclid_polish.tracking import TrackingError, default_store
from euclid_polish.tracking.store import TrackingStore
from euclid_polish.web.app import create_app
from euclid_polish.web.helpers import ensemble_viz as ev
from euclid_polish.web.helpers import model_catalog as mc
from euclid_polish.web.helpers import sky_records, system_alerts
from euclid_polish.web.helpers import universal_models as um
from euclid_polish.web.routes import system

UNIVERSAL = {"asinh_knees": [0.1, 1.0, 10.0], "knee_loss": "balanced",
             "learned_output_knee": {"init_e": 10.0, "min_e": 0.1, "max_e": 1e4},
             "num_res_blocks": 16, "isotropy_penalty": 0.5}
SINGLE = {"asinh_knee": 10.0, "num_res_blocks": 32}


class _Cap:
    def tick(self, *_a, **_k):
        pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR", str(tmp_path / "ckpt/wdsr"))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    monkeypatch.setattr(Config, "TRACKING_DIR", str(tmp_path / "tracking"))
    monkeypatch.setattr(ev, "_sky_records_local_dir", lambda: None)
    default_store().create_campaign("production test")
    base = ev.ensemble_dir()
    _member(base, 1, SINGLE)
    er.add_universal_entry(base, "238")
    _member(base, 238, UNIVERSAL)
    return base


def _member(base, n, origin, ckpt="ckpt-5"):
    d = os.path.join(base, f"member_{n:02d}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "checkpoint"), "w") as f:
        f.write(f'model_checkpoint_path: "{ckpt}"\n')
    with open(os.path.join(d, f"{ckpt}.index"), "wb") as f:
        f.write(b"idx")
    with open(os.path.join(d, "origin.json"), "w") as f:
        json.dump(origin, f)
    return d


# ---- the registry pointer ---------------------------------------------------- #

def test_the_pointer_survives_bootstrap_writes_and_guards_the_model(env):
    base = env
    assert er.production_name(base) is None
    assert er.set_production(base, "238") == "member_238"
    _member(base, 2, SINGLE)                          # a new dir → a bootstrap write
    reg = er.load_registry(base)
    assert "member_02" in reg["active"] and reg["production"]["name"] == "member_238"
    with open(er.registry_path(base)) as f:
        assert json.load(f)["production"]["name"] == "member_238"
    with pytest.raises(ValueError, match="production model"):
        er.archive_member_entry(base, "member_238", zip_path="models/x.zip", commit=None)
    with pytest.raises(ValueError, match="production model"):
        er.remove_universal_entry(base, "238")
    with pytest.raises(ValueError, match="not a registered universal model"):
        er.set_production(base, "1")
    assert er.set_production(base, None) is None and er.production_name(base) is None


# ---- the catalogue ------------------------------------------------------------ #

class _Members:
    def get(self, label):
        assert label == "238·psnr"
        return np.full((4, 4, 4), 7.0, np.float32)


def test_production_runs_the_universal_model_the_registry_names(env):
    er.set_production(env, "238")
    spec = mc.resolve_spec("production")
    assert spec.available and spec.reads == ("238·psnr",) and spec.combiner_dir is None
    assert spec.label == "Production · Universal 238" and mc.is_single_model(spec)
    assert spec.fingerprint == mc.resolve_spec("universal:member_238").fingerprint
    assert spec.details["universal"] is True
    assert spec.details["model_spec"] == "universal:member_238"
    payload = mc.catalog_payload()
    assert payload["production_kind"] == "universal"
    assert payload["production"] == {"kind": "universal", "spec": "universal:member_238",
                                     "name": "member_238", "label": "Universal 238",
                                     "available": True, "reason": None}
    out = mc.predict(spec, np.zeros((2, 2, 4), np.float32), _Members())
    assert np.all(out == 7.0)
    shutil.rmtree(os.path.join(env, "member_238"))
    gone = mc.resolve_spec("production")
    assert not gone.available and gone.reason.startswith("production model member_238:")


def test_without_a_pointer_production_stays_the_gate(env):
    assert mc.production_summary()["kind"] == "gate"
    spec = mc.resolve_spec("production")
    assert spec.reason == "spatial_gate_combiner is not fitted"


# ---- the evaluators' model ------------------------------------------------------ #

class _FakeEnsemble:
    made: list[dict] = []

    def __init__(self, base, **kwargs):
        self.made.append(kwargs)
        self.member_labels = list(kwargs.get("labels") or ["01·psnr"])
        self.n_members = len(self.member_labels)

    def member_arrays(self, lr, *_a, **_k):
        return np.ones((self.n_members, 4, 4, 4), np.float32)


def test_the_eval_model_is_the_universal_production(env, monkeypatch):
    er.set_production(env, "238")
    _FakeEnsemble.made = []
    monkeypatch.setattr(ensemble_infer, "EnsembleModel", _FakeEnsemble)
    logs: list[str] = []
    model = ensemble_infer.load_eval_ensemble(env, log=logs.append)
    assert _FakeEnsemble.made[-1]["labels"] == ["238·psnr"]
    assert _FakeEnsemble.made[-1]["include_universal"] is True
    assert model.combiner_kind == "universal" and model.member_labels == ["238·psnr"]
    assert logs[-1] == "using universal model 238 (production)"
    identity = catalog_runner.eval_model_identity(model)
    assert identity["combiner_kind"] == "universal"
    assert identity["combiner_fingerprint"] == member_fingerprint(os.path.join(env, "member_238"))
    assert catalog_runner.current_eval_identity(env) == identity
    needs = catalog_runner.reuse_requirements(identity)
    assert needs["member_labels"] == ["238·psnr"] and needs["identity"] is identity


def test_a_production_model_that_does_not_load_falls_back_to_the_mean_loudly(env, monkeypatch):
    er.set_production(env, "238")

    class Failing(_FakeEnsemble):
        def __init__(self, base, **kwargs):
            if kwargs.get("include_universal"):
                raise ValueError("no checkpoint")
            super().__init__(base, **kwargs)

    monkeypatch.setattr(ensemble_infer, "EnsembleModel", Failing)
    logs: list[str] = []
    model = ensemble_infer.load_eval_ensemble(env, log=logs.append)
    assert model.combiner_kind is None
    assert "falls back to the plain member mean" in logs[-1]
    assert "production model 238·psnr does not load" in logs[-1]


def test_an_sr_made_by_the_gate_reads_production_changed(monkeypatch):
    recorded = {"member_labels": ["1·psnr", "2·psnr"], "combiner_kind": "spatial_gate",
                "combiner_fingerprint": "a"}
    current = {"member_labels": ["238·psnr"], "combiner_kind": "universal",
               "combiner_fingerprint": "b"}
    monkeypatch.setattr(sky_records, "sr_count", lambda subset: 3)
    monkeypatch.setattr(sky_records, "read_sr_manifest", lambda subset: {"identity": recorded})
    state = sky_records.sr_state("test", current, None)
    assert state["state"] == "stale"
    assert state["reasons"] == [
        "production changed (the spatial_gate combiner → universal model 238)"]


# ---- checks and payloads ------------------------------------------------------------ #

def test_the_checks_name_the_universal_production(env):
    er.set_production(env, "238")
    check = system.check_combiner()
    assert check["state"] == "ok" and check["title"] == "Production is Universal 238"
    stage = system_alerts.gate_stage({"checks": [{**check, "id": "combiner"}]})
    assert stage["state"] == "current" and stage["reason"] == "production is Universal 238"
    overview = ev.ensemble_overview(False)
    ids = {c["id"]: c for c in overview["checks"]}
    assert ids["production"]["ok"] and "gate" not in ids
    assert overview["production_model"]["name"] == "member_238"


def test_the_universal_tab_compares_with_the_production_models_score(env):
    er.set_production(env, "238")
    er.add_universal_entry(env, "211")              # registered before its dir lands
    _member(env, 211, UNIVERSAL)
    store = ev.universal_scores_dir()
    os.makedirs(store, exist_ok=True)
    scores = {"schema": ev.UNIVERSAL_SCORES_SCHEMA, "identity": ev.universal_score_identity(),
              "models": {name: {"fingerprint": member_fingerprint(os.path.join(env, name)),
                                "integrated": [mean] * 4, "mean": mean}
                         for name, mean in (("member_238", 62.0), ("member_211", 61.0))}}
    with open(os.path.join(store, ev.UNIVERSAL_SCORES_INDEX), "w") as f:
        json.dump(scores, f)
    payload = um.universal_payload()
    assert payload["production_model"] == "member_238"
    assert payload["production"] == {"integrated": [62.0] * 4, "mean": 62.0, "stale": False,
                                     "model": "member_238"}
    rows = {m["name"]: m for m in payload["models"]}
    assert rows["member_211"]["vs_production_db"] == pytest.approx(-1.0)
    assert rows["member_238"]["vs_production_db"] == pytest.approx(0.0)


# ---- archive / restore every gate ------------------------------------------------------ #

def _gate_world():
    regime = ev._ensemble_regime_dir(False)
    for name in ("spatial_gate_combiner", "spatial_gate_v1", "spatial_gate_backup_20261002-204649"):
        os.makedirs(os.path.join(regime, name))
        with open(os.path.join(regime, name, "combiner.json"), "w") as f:
            json.dump({"kind": "spatial_gate", "name": name}, f)
    os.makedirs(os.path.join(regime, "spatial_gate_comparisons"))
    for rel in ("spatial_gate_combiner_evals.json", "spatial_gate_comparison.json",
                "spatial_gate_comparisons/20261013.json", "ensemble_evals.json"):
        with open(os.path.join(regime, rel), "w") as f:
            json.dump({"file": rel}, f)
    cubes = os.path.join(regime, "cubes")
    os.makedirs(cubes)
    np.save(os.path.join(cubes, "comb_spatial_gate_00001.npy"), np.zeros(2))
    with open(os.path.join(cubes, "viz_index.json"), "w") as f:
        json.dump({"member_labels": [], "indices": [1], "has_combiner_spatial_gate": True}, f)
    return regime


def test_every_gate_archives_into_one_zip_and_restores(env):
    regime = _gate_world()
    with pytest.raises(RuntimeError, match="still the spatial gate"):
        ev.job_archive_gates(_Cap())
    er.set_production(env, "238")
    out = ev.job_archive_gates(_Cap())
    assert out["items"] == ["spatial_gate_backup_20261002-204649", "spatial_gate_combiner",
                            "spatial_gate_combiner_evals.json", "spatial_gate_comparison.json",
                            "spatial_gate_comparisons", "spatial_gate_v1"]
    assert not [n for n in os.listdir(regime) if n.startswith("spatial_gate_")]
    assert os.path.isfile(os.path.join(regime, "ensemble_evals.json"))     # not a gate file
    assert not os.path.exists(os.path.join(regime, "cubes", "comb_spatial_gate_00001.npy"))
    with open(os.path.join(regime, "cubes", "viz_index.json")) as f:
        assert json.load(f)["has_combiner_spatial_gate"] is False
    zpath = os.path.join(default_store().current_dir, "models", out["zip"])
    with zipfile.ZipFile(zpath) as zf:
        names = set(zf.namelist())
        assert json.loads(zf.read("ARCHIVE.json"))["production"] == "member_238"
    assert "spatial_gate_combiner/combiner.json" in names
    assert "spatial_gate_comparisons/20261013.json" in names
    assert out["zip"] in default_store().read_log()
    archives = ev.gate_archives()
    assert [a["name"] for a in archives] == [out["zip"]] and archives[0]["items"] == out["items"]
    assert ev.combiner_variants(False)["archives"][0]["n_items"] == 6
    assert mc.resolve_spec("production").available                     # 238 still runs

    restored = ev.job_restore_gates(_Cap())
    assert restored["restored"] == out["items"] and restored["production"] == "member_238"
    with open(os.path.join(regime, "spatial_gate_v1", "combiner.json")) as f:
        assert json.load(f)["name"] == "spatial_gate_v1"
    with pytest.raises(RuntimeError, match="already in the regime dir"):
        ev.job_restore_gates(_Cap())


def test_a_failed_zip_puts_every_gate_back(env, monkeypatch):
    regime = _gate_world()
    er.set_production(env, "238")

    def boom(self, *_a, **_k):
        raise TrackingError("disk full")

    monkeypatch.setattr(TrackingStore, "archive_model_zip", boom)
    with pytest.raises(TrackingError):
        ev.job_archive_gates(_Cap())
    assert sorted(n for n in os.listdir(regime) if n.startswith("spatial_gate_")) == [
        "spatial_gate_backup_20261002-204649", "spatial_gate_combiner",
        "spatial_gate_combiner_evals.json", "spatial_gate_comparison.json",
        "spatial_gate_comparisons", "spatial_gate_v1"]
    assert not [n for n in os.listdir(regime) if n.startswith(".gate-archive")]


def test_the_production_model_cannot_be_archived(env):
    er.set_production(env, "238")
    with pytest.raises(RuntimeError, match="production model"):
        ev.job_archive_member(_Cap(), name="member_238")


# ---- routes ------------------------------------------------------------------------------ #

def test_routes_set_production_and_guard_the_gate_archive(env):
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        r = client.post("/ensemble/combiners/archive")
        assert r.status_code == 400 and "still the spatial gate" in r.get_json()["error"]
        assert client.post("/ensemble/combiners/restore").status_code == 400
        r = client.post("/ensemble/production", data={"model": "universal:member_238"})
        assert r.get_json() == {"ok": True, "production": "member_238",
                                "spec": "universal:member_238"}
        assert client.get("/api/models").get_json()["production"]["name"] == "member_238"
        r = client.post("/ensemble/archive-member", data={"member": "238"})
        assert r.status_code == 400 and "production model" in r.get_json()["error"]
        assert client.post("/ensemble/production", data={"model": "1"}).status_code == 400
        assert client.post("/ensemble/production", data={}).status_code == 400
        r = client.post("/ensemble/production", data={"model": "gate"})
        assert r.get_json()["production"] is None
