"""The resource-advisor routes on a temp job ledger: the dashboard list, one
step's usage + recommendation, and the read-only recommend POST (JSON and
form bodies). Offline, never gated, never spawns a job."""

from __future__ import annotations

import csv
import json
import os

import pytest

from euclid_polish.observability import JobLog
from euclid_polish.web import fasrc_jobs, job_config, remote
from euclid_polish.web.app import create_app
from euclid_polish.web.fasrc_pipeline import REGISTRY as STEP_REGISTRY
from euclid_polish.web.jobs import REGISTRY
from euclid_polish.web.routes import resources as resource_routes

_VT = {"regenerate_splits": "validate,test", "n_train": "6400", "n_valid": "100",
       "n_test": "100", "image_size": "510"}
_ENS = {"mode": "add", "steps": "70000", "batch_size": "4", "hr_crop_size": "256",
        "member_spec": json.dumps([{"num_res_blocks": 32}])}

_SUMMARY_KEYS = {
    "step_id", "label", "needs_gpu", "registered", "runs", "states", "success_rate",
    "last_submitted_at",
    "cpu_efficiency", "gpu_util", "mem_ratio", "time_ratio", "peak_mem_p90_mb",
    "cpu_hours_alloc", "cpu_hours_used", "gpu_hours_alloc", "gpu_hours_used",
    "mem_gb_hours_alloc", "mem_gb_hours_used"}
_RECOMMENDATION_KEYS = {"ok", "step_id", "available", "confidence", "resources", "current",
                        "changes", "basis", "notes", "warnings"}


def _cells(jobid, step, *, day, state="COMPLETED", cpus=10, gpus=0, memory="20G",
           time_limit="1:00:00", elapsed=600, rss=15_000, eff=0.6, gpu_util="", params=None):
    return {
        "jobid": str(jobid), "submitted_at": f"2026-09-{day:02d}T00:00:00Z", "step_id": step,
        "label": f"{step} #{jobid}", "partition": "gpu" if gpus else "shared",
        "req_cpus": str(cpus), "req_gpus": str(gpus), "req_memory": memory,
        "req_time_limit": time_limit, "params_json": json.dumps(params or {}),
        "state": state, "elapsed_seconds": str(elapsed), "cpu_efficiency": str(eff),
        "max_rss_mb": str(rss), "alloc_cpus": str(cpus), "alloc_gpus": str(gpus),
        "jobstats_gpu_util": str(gpu_util),
    }


def _write(log: JobLog, rows: list[dict[str, str]]) -> None:
    """Write the whole ledger at once (as ``JobLog`` would, header and all)."""
    with open(log.csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=JobLog.COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in JobLog.COLUMNS})


def _ledger() -> list[dict[str, str]]:
    rows = [_cells(100 + i, "synthetic_generate", day=1 + i, params=_VT, memory="17G",
                   time_limit="20:00") for i in range(5)]
    rows.append(_cells(200, "synthetic_generate", day=8, params=_VT, state="OUT_OF_MEMORY",
                       memory="15G", elapsed=300))
    rows.append(_cells(201, "synthetic_generate", day=9, params=_VT, state="FAILED",
                       elapsed=20))
    rows += [_cells(300 + i, "ensemble_train", day=2 + i, params=_ENS, cpus=16, gpus=1,
                    memory="32G", time_limit="2:00:00", elapsed=4600, rss=17_000, eff=0.4,
                    gpu_util=71) for i in range(4)]
    rows.append(_cells(400, "train", day=3, params={"steps": "5000"}, gpus=1, cpus=4))
    rows.append(_cells(500, "euclid_query", day=20, cpus=1, memory="4G"))
    return rows


@pytest.fixture
def log(tmp_path, monkeypatch):
    ledger = JobLog(str(tmp_path / "fasrc_job_log.csv"))
    monkeypatch.setattr(fasrc_jobs, "JOBLOG", ledger)
    monkeypatch.setattr(remote.STATE, "ssh", None)         # offline
    # /config (the recommend POST merges it in, as a submit does): the
    # defaults (6400 / 100 / 100 scenes at 510 px), never the user's file.
    monkeypatch.setattr(job_config, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(job_config, "CONFIG_PATH", str(tmp_path / "job_config.json"))
    _write(ledger, _ledger())
    return ledger


def _card_params(step_id: str, **values: str) -> dict[str, str]:
    """What a step card's advice POST carries (``submitParams``): every task
    param, blank unless set, and none of the /config-injected knobs."""
    params = {p.name: "" for p in STEP_REGISTRY.by_id[step_id].task_params}
    params.update(values)
    return params


@pytest.fixture
def client(log):
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


# --------------------------------------------------------------------------- list

def test_list_orders_the_main_steps_first(client):
    body = client.get("/api/fasrc/resources").get_json()
    assert body["ok"] is True
    order = [s["step_id"] for s in body["steps"]]
    assert order == ["ensemble_train", "synthetic_generate", "euclid_query", "train"]
    for summary in body["steps"]:
        assert set(summary) == _SUMMARY_KEYS
    gen = body["steps"][1]
    assert gen["label"] == "Generate synthetic training pairs (CPU)"   # registry title
    assert gen["registered"] is True
    assert gen["states"] == {"completed": 5, "oom": 1, "timeout": 0, "failed": 1,
                             "cancelled": 0, "running": 0}
    assert gen["needs_gpu"] is False and body["steps"][0]["needs_gpu"] is True
    # A historical step (no longer registered) is labelled by its id, and says so.
    assert body["steps"][3]["label"] == "train" and body["steps"][3]["registered"] is False


def test_an_empty_ledger_lists_nothing(client, log):
    _write(log, [])
    assert client.get("/api/fasrc/resources").get_json() == {"ok": True, "steps": []}


# --------------------------------------------------------------------------- one step

def test_step_detail_recommends_the_next_run_like_the_last(client):
    body = client.get("/api/fasrc/resources/synthetic_generate").get_json()
    assert body["ok"] is True and body["step_id"] == "synthetic_generate"
    assert set(body["summary"]) == _SUMMARY_KEYS
    ids = [r["jobid"] for r in body["runs"]]
    assert ids == ["201", "200", "104", "103", "102", "101", "100"]      # newest first
    assert body["runs"][0]["label"] == "synthetic_generate #201"
    assert body["runs"][2]["units"] == 200 and body["runs"][2]["units_label"] == "images"
    rec = body["recommendation"]
    assert set(rec) == _RECOMMENDATION_KEYS
    assert rec["available"] is True
    # Latest counted run = the OOM at 15G (its params and resources).
    assert rec["current"]["memory"] == "15G" and rec["current"]["n_cpus"] == "10"
    assert rec["basis"]["level"] == "exact" and rec["basis"]["n_runs"] == 6
    memory = next(c for c in rec["changes"] if c["field"] == "memory")
    assert memory["recommended"] == "20G" and "OOM" in memory["reason"]


def test_runs_are_capped_at_two_hundred(client, log):
    _write(log, [_cells(i, "euclid_query", day=1 + i % 28, cpus=1) for i in range(205)])
    body = client.get("/api/fasrc/resources/euclid_query").get_json()
    assert len(body["runs"]) == 200 and body["summary"]["runs"] == 205


def test_a_historical_step_with_rows_still_answers(client):
    body = client.get("/api/fasrc/resources/train").get_json()
    assert body["ok"] is True and body["summary"]["label"] == "train"
    assert body["summary"]["registered"] is False
    assert body["recommendation"]["available"] is True
    assert body["recommendation"]["basis"]["units_label"] == "steps"


def test_a_registered_step_without_rows_has_no_recommendation(client):
    body = client.get("/api/fasrc/resources/tng_grid").get_json()
    assert body["ok"] is True and body["runs"] == [] and body["summary"]["runs"] == 0
    rec = body["recommendation"]
    assert rec["available"] is False and rec["changes"] == []
    assert set(rec["resources"]) == {"n_cpus", "n_gpus", "memory", "time_limit"}


def test_an_unknown_step_is_404(client):
    for response in (client.get("/api/fasrc/resources/no_such_step"),
                     client.post("/api/fasrc/resources/no_such_step/recommend", json={})):
        assert response.status_code == 404
        body = response.get_json()
        assert body["ok"] is False and "no_such_step" in body["error"]


# --------------------------------------------------------------------------- recommend

def test_recommend_from_a_json_body(client):
    response = client.post("/api/fasrc/resources/ensemble_train/recommend", json={
        "params": _ENS,
        "resources": {"n_cpus": "16", "n_gpus": "1", "memory": "32G", "time_limit": "2:00:00"},
    })
    assert response.status_code == 200
    rec = response.get_json()
    assert set(rec) == _RECOMMENDATION_KEYS and rec["ok"] is True
    assert rec["resources"]["n_cpus"] == "16" and rec["resources"]["n_gpus"] == "1"
    assert rec["resources"]["memory"] == "20G"                  # 17 000 MB × 1.2 → 20G
    assert rec["basis"]["units"] == 70_000 and rec["basis"]["level"] == "similar"
    assert {c["field"] for c in rec["changes"]} == {"memory", "time_limit"}
    assert any("TensorFlow" in n for n in rec["notes"])


def test_recommend_from_form_fields_splits_resources_from_params(client):
    form = {**_VT, "n_cpus": "10", "n_gpus": "0", "memory": "17G", "time_limit": "20:00",
            "partition": "shared", "confirm": "yes"}
    rec = client.post("/api/fasrc/resources/synthetic_generate/recommend", data=form).get_json()
    assert rec["current"] == {"n_cpus": "10", "n_gpus": "0", "memory": "17G",
                              "time_limit": "20:00"}
    assert rec["basis"]["units"] == 200 and rec["basis"]["level"] == "exact"


def test_a_flat_json_body_is_accepted(client):
    rec = client.post("/api/fasrc/resources/synthetic_generate/recommend",
                      json={**_VT, "n_cpus": 10, "memory": "17G"}).get_json()
    assert rec["current"]["n_cpus"] == "10" and rec["basis"]["units"] == 200


def test_different_params_change_the_basis(client):
    rec = client.post("/api/fasrc/resources/synthetic_generate/recommend", json={
        "params": {**_VT, "regenerate_splits": "train"},
        "resources": {"n_cpus": "10", "n_gpus": "0", "memory": "17G", "time_limit": "20:00"},
    }).get_json()
    assert rec["basis"]["level"] == "step" and rec["basis"]["units"] == 6400


_FORM = {"n_cpus": "10", "n_gpus": "0", "memory": "17G", "time_limit": "20:00"}


def test_the_step_cards_params_are_completed_like_a_submit(client):
    """The synthetic step card posts only its task params (no scene counts,
    no image size: /config injects those at submit, so every ledger row has
    them). The advisor must see the same completed params to match history."""
    params = _card_params("synthetic_generate", regenerate_splits="validate,test")
    assert not {"n_train", "n_valid", "n_test", "image_size"} & set(params)
    rec = client.post("/api/fasrc/resources/synthetic_generate/recommend",
                      json={"params": params, "resources": _FORM}).get_json()
    assert rec["basis"]["level"] == "exact" and rec["basis"]["n_runs"] == 6
    assert rec["basis"]["units"] == 200 and rec["basis"]["rate_s_per_unit"] is not None
    assert not any("Units unknown" in n for n in rec["notes"])


def test_the_plan_uses_the_current_config(client):
    job_config.update({"n_valid": "300"})
    params = _card_params("synthetic_generate", regenerate_splits="validate,test")
    rec = client.post("/api/fasrc/resources/synthetic_generate/recommend",
                      json={"params": params, "resources": _FORM}).get_json()
    assert rec["basis"]["units"] == 400                            # 300 + 100 from /config
    # /config wins over a posted value on this step, as at submit.
    rec = client.post("/api/fasrc/resources/synthetic_generate/recommend",
                      json={"params": {**params, "n_valid": "5"}, "resources": _FORM}).get_json()
    assert rec["basis"]["units"] == 400


def test_blank_task_params_take_the_schema_defaults(client):
    """A blank ``steps`` trains the schema default, as the submit fills it."""
    params = _card_params("ensemble_train", batch_size="4", hr_crop_size="256",
                          member_spec=_ENS["member_spec"])
    rec = client.post("/api/fasrc/resources/ensemble_train/recommend", json={
        "params": params,
        "resources": {"n_cpus": "16", "n_gpus": "1", "memory": "32G", "time_limit": "2:00:00"},
    }).get_json()
    default_steps = next(p.default for p in STEP_REGISTRY.by_id["ensemble_train"].task_params
                         if p.name == "steps")
    assert rec["basis"]["units"] == default_steps
    assert rec["basis"]["level"] == "similar" and rec["basis"]["rate_s_per_unit"] is not None


def test_a_half_typed_param_still_gets_the_config(client):
    params = _card_params("synthetic_generate", regenerate_splits="validate,test",
                          onthefly_train="mayb")                   # the schema refuses it
    response = client.post("/api/fasrc/resources/synthetic_generate/recommend",
                           json={"params": params, "resources": _FORM})
    assert response.status_code == 200
    rec = response.get_json()
    assert rec["basis"]["level"] == "exact" and rec["basis"]["units"] == 200


@pytest.mark.parametrize("body", ["[1, 2]", '"text"', "{broken"])
def test_a_non_object_json_body_is_400(client, body):
    response = client.post("/api/fasrc/resources/synthetic_generate/recommend", data=body,
                           content_type="application/json")
    assert response.status_code == 400 and response.get_json()["ok"] is False


def test_params_and_resources_must_be_objects(client):
    response = client.post("/api/fasrc/resources/synthetic_generate/recommend",
                           json={"params": [1], "resources": {}})
    assert response.status_code == 400


def test_recommend_is_read_only_and_spawns_nothing(client, log, monkeypatch):
    def forbidden(*_a, **_kw):
        raise AssertionError("the advisor must never submit")

    monkeypatch.setattr(fasrc_jobs, "submit_sbatch_script", forbidden)
    before_jobs = [j["job_id"] for j in REGISTRY.list(summary=True)]
    with open(log.csv_path, encoding="utf-8") as handle:
        before_ledger = handle.read()
    same_origin = {"Sec-Fetch-Site": "same-origin"}
    assert client.post("/api/fasrc/resources/ensemble_train/recommend", json={"params": _ENS},
                       headers=same_origin).status_code == 200
    assert client.get("/api/fasrc/resources/ensemble_train").status_code == 200
    assert client.get("/api/fasrc/resources").status_code == 200
    assert [j["job_id"] for j in REGISTRY.list(summary=True)] == before_jobs
    with open(log.csv_path, encoding="utf-8") as handle:
        assert handle.read() == before_ledger


def test_a_cross_site_recommend_is_refused(client):
    response = client.post("/api/fasrc/resources/ensemble_train/recommend", json={},
                           headers={"Sec-Fetch-Site": "cross-site"})
    assert response.status_code == 403


# --------------------------------------------------------------------------- cache

def test_the_ledger_is_parsed_once_per_change(client, log, monkeypatch):
    reads = []
    original = log.list_all

    def counting():
        reads.append(1)
        return original()

    monkeypatch.setattr(log, "list_all", counting)
    client.get("/api/fasrc/resources")
    client.get("/api/fasrc/resources/ensemble_train")
    assert len(reads) == 1
    reads.clear()
    rows = _ledger() + [_cells(999, "tng_grid", day=25, cpus=2)]
    _write(log, rows)
    body = client.get("/api/fasrc/resources").get_json()
    assert len(reads) == 1
    assert "tng_grid" in [s["step_id"] for s in body["steps"]]


def test_a_missing_ledger_reads_as_empty(client, log):
    os.remove(log.csv_path)
    assert client.get("/api/fasrc/resources").get_json()["steps"] == []
    assert resource_routes._ledger_runs() == ()
