"""Ops › Provenance: the lineage index (helpers/provenance_index.py) and its
routes (routes/provenance.py), on synthetic sidecars in tmp dirs."""

from __future__ import annotations

import json
import os

import pytest

from euclid_polish.config import Config
from euclid_polish.web.app import create_app
from euclid_polish.web.helpers import provenance_index as pi

GEN = "11111111"
TRAIN = "22222222"
MODEL_OLD = "33333333"
MODEL_NEW = "44444444"
RUN_OLD = "55555555"
RUN_NEW = "66666666"
SR_OLD = "77777777"
SR_NEW = "88888888"
SR_LEGACY = "99999999"


def _write(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh)


@pytest.fixture
def layout(tmp_path):
    """data/_prov with runs, co-located SR sidecars, two member checkpoints."""
    data = tmp_path / "data"
    prov = data / "_prov"
    ckpt = tmp_path / "ckpt"
    _write(str(prov / f"{GEN}.generationrun.json"), {
        "kind": "generationrun", "id": GEN, "created_at": "2026-01-01T00:00:00+00:00",
        "config": {"config_type": "SkySimulatorConfig", "fields": {}}, "parents": [], "inputs": [],
        "git": {"short": "abc1234", "dirty": False}, "status": "ok", "seed": 7})
    _write(str(prov / f"{RUN_OLD}.inferencerun.json"), {
        "kind": "inferencerun", "id": RUN_OLD, "created_at": "2026-02-01T00:00:00+00:00",
        "inputs": [MODEL_OLD], "parents": [], "status": "ok"})
    _write(str(prov / f"{RUN_NEW}.inferencerun.json"), {
        "kind": "inferencerun", "id": RUN_NEW, "created_at": "2026-03-01T00:00:00+00:00",
        "inputs": [MODEL_NEW], "parents": [], "status": "ok"})
    (prov / "not-a-sidecar.json").write_text("{}")
    tile = data / "eval_results" / "gal_1"
    _write(str(tile / f"{SR_OLD}.srcutoutartifact.json"), {
        "kind": "srcutoutartifact", "id": SR_OLD, "created_at": "2026-02-01T00:00:01+00:00",
        "produced_by": RUN_OLD, "parents": [MODEL_OLD], "path": "./data/eval_results/gal_1/SR.fits",
        "format": "fits", "descriptors": {"ra": 266.8, "dec": 67.4}})
    _write(str(tile / f"{SR_NEW}.srcutoutartifact.json"), {
        "kind": "srcutoutartifact", "id": SR_NEW, "created_at": "2026-03-01T00:00:01+00:00",
        "produced_by": RUN_NEW, "parents": [], "path": "./data/eval_results/gal_1/SR.fits"})
    _write(str(tile / f"{SR_LEGACY}.srcutoutartifact.json"), {
        "kind": "srcutoutartifact", "id": SR_LEGACY, "created_at": "2025-12-01T00:00:00+00:00",
        "produced_by": "00000000", "parents": []})
    _write(str(ckpt / "ensemble" / "member_01" / "provenance.json"),
           {"id": MODEL_OLD, "produced_by": TRAIN, "parents": [], "schema_version": 3})
    _write(str(ckpt / "ensemble" / "member_02" / "provenance.json"),
           {"id": MODEL_NEW, "produced_by": TRAIN, "parents": [GEN], "schema_version": 3})
    current = [{"id": MODEL_NEW, "member": "member_02", "regime": "starfull", "dir": "ckpt/ensemble/member_02"}]
    return {"data": str(data), "prov": str(prov), "ckpt": str(ckpt), "current": current,
            "tmp": tmp_path}


def _index(layout, **kw):
    return pi.build_index(prov_dir=layout["prov"], data_dirs=[layout["data"]],
                          ckpt_root=layout["ckpt"], current_models=layout["current"], **kw)


def test_index_reads_prov_sidecars_and_checkpoint_stamps(layout):
    idx = _index(layout)
    assert set(idx.entries) == {GEN, RUN_OLD, RUN_NEW, SR_OLD, SR_NEW, SR_LEGACY, MODEL_OLD, MODEL_NEW}
    assert idx.get(GEN)["source"] == "prov"
    assert idx.get(SR_OLD)["source"] == "sidecar"
    model = idx.get(MODEL_OLD)
    assert model["source"] == "checkpoint" and model["kind"] == "checkpointartifact"
    assert model["member"] == "member_01"
    assert idx.get(SR_OLD)["ra"] == 266.8
    roles = {r["role"]: r["records"] for r in idx.roots}
    assert roles == {"index": 3, "data": 3, "checkpoints": 2}     # _prov not walked twice


def test_upstream_downstream_and_transitive_walks(layout):
    idx = _index(layout)
    up = {(r["role"], r["id"]) for r in idx.upstream(SR_OLD)}
    assert up == {("parent", MODEL_OLD), ("produced_by", RUN_OLD)}
    down = {r["id"] for r in idx.downstream(MODEL_OLD)}
    assert down == {SR_OLD, RUN_OLD}
    anc, total = idx.walk(SR_OLD, "ancestors")
    assert total == 3                          # model, run, and the model's training run
    missing = next(a for a in anc if a["id"] == TRAIN)
    assert missing["exists"] is False and missing["depth"] == 2
    desc, total = idx.walk(GEN, "descendants")
    assert {d["id"] for d in desc} == {MODEL_NEW, RUN_NEW, SR_NEW} and total == 3


def test_verdicts_compare_the_model_with_the_active_members(layout):
    idx = _index(layout)
    assert idx.verdict(SR_OLD) == "stale"       # made by an inactive member
    assert idx.verdict(SR_NEW) == "current"     # via its inference run's input
    assert idx.verdict(SR_LEGACY) == "unknown"  # sentinel produced_by, no model
    assert idx.verdict(RUN_NEW) == "current"
    assert idx.verdict(GEN) is None             # not a model product
    counts = idx.counts()
    assert counts["verdicts"] == {"current": 2, "stale": 2, "unknown": 1}
    assert counts["kinds"]["srcutoutartifact"] == 3


def test_search_filters_and_sorts_newest_first(layout):
    idx = _index(layout)
    assert [e["id"] for e in idx.search(kind="srcutoutartifact")] == [SR_NEW, SR_OLD, SR_LEGACY]
    assert [e["id"] for e in idx.search(q="skysimulator")] == [GEN]
    assert [e["id"] for e in idx.search(q="member_01")] == [MODEL_OLD]
    assert [e["id"] for e in idx.search(verdict="stale", kind="srcutoutartifact")] == [SR_OLD]
    assert [e["id"] for e in idx.search(q="7777")] == [SR_OLD]


def test_walk_cap_is_respected(layout):
    idx = _index(layout)
    items, total = idx.walk(GEN, "descendants", cap=1)
    assert len(items) == 1 and total == 3


def test_max_files_truncates_the_walk(layout):
    idx = _index(layout, max_files=1)
    assert idx.truncated is True


# --------------------------------------------------------------------------- routes

@pytest.fixture
def client(layout, monkeypatch):
    monkeypatch.setattr(Config, "PROV_DIR", layout["prov"], raising=False)
    monkeypatch.setattr(Config, "DATA_DIR", layout["data"], raising=False)
    monkeypatch.setattr(Config, "DEFAULT_CHECKPOINT_DIR",
                        os.path.join(layout["ckpt"], "wdsr"), raising=False)
    monkeypatch.setattr(pi, "current_model_ids", lambda ensemble_dir=None: layout["current"])
    monkeypatch.chdir(layout["tmp"])
    pi.reset_cache()
    app = create_app()
    app.config["TESTING"] = True
    yield app.test_client()
    pi.reset_cache()


def test_summary_route(client):
    body = client.get("/api/provenance/summary").get_json()
    assert body["ok"] is True
    assert body["total"] == 8
    assert body["counts"]["verdicts"]["stale"] == 2
    assert body["current_models"][0]["member"] == "member_02"
    assert {r["role"] for r in body["roots"]} == {"index", "data", "checkpoints"}


def test_records_route_pages_and_filters(client):
    body = client.get("/api/provenance/records?kind=srcutoutartifact&limit=2").get_json()
    assert body["total"] == 3 and len(body["records"]) == 2
    assert body["records"][0]["id"] == SR_NEW and body["records"][0]["verdict"] == "current"
    body = client.get("/api/provenance/records?verdict=stale&offset=0").get_json()
    assert {r["id"] for r in body["records"]} == {SR_OLD, RUN_OLD}
    assert client.get("/api/provenance/records?verdict=bogus").status_code == 400


def test_record_route_detail(client):
    r = client.get(f"/api/provenance/record/{SR_OLD}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["record"]["produced_by"] == RUN_OLD          # the stored JSON
    assert body["entry"]["verdict"] == "stale"
    assert {u["id"] for u in body["upstream"]} == {MODEL_OLD, RUN_OLD}
    assert body["ancestors"]["total"] == 3
    assert body["models"][0]["member"] == "member_01"
    assert body["inspect_path"] == "data/eval_results/gal_1/SR.fits"
    assert client.get("/api/provenance/record/zzzz").status_code == 400
    assert client.get("/api/provenance/record/abcdef01").status_code == 404


def test_rebuild_route_picks_up_new_sidecars(client, layout):
    assert client.get("/api/provenance/summary").get_json()["total"] == 8
    _write(os.path.join(layout["data"], "eval_results", "gal_2", "aaaaaaaa.srcutoutartifact.json"),
           {"kind": "srcutoutartifact", "id": "aaaaaaaa", "created_at": "2026-04-01T00:00:00+00:00"})
    r = client.post("/api/provenance/rebuild")
    assert r.status_code == 200 and r.get_json()["total"] == 9
