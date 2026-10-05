"""What a freeze would capture now (``euclid_polish/studies/candidates.py``):
the ensemble summary with a status per numbers block, and every field with
SR for all members (disabled ones carry the reason). Read-only."""
from __future__ import annotations

import json
import os

import numpy as np

from euclid_polish.studies import candidates
from euclid_polish.web import remote
from euclid_polish.web.helpers import experiments, model_catalog, real_tiles
from tests import _real_fixtures as rf
from tests._studies_fixtures import LABELS, make_env


def _snapshot(root):
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)
            out[path] = os.stat(path).st_mtime_ns
    return out


def _by_fid(payload):
    return {f["fid"]: f for f in payload["fields"]}


def test_candidates_summarise_the_ensemble_and_list_fields(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    monkeypatch.setattr(remote.STATE, "ssh", None)
    payload = candidates.candidates(False)
    ens = payload["ensemble"]
    assert ens["n_members"] == 3 and ens["members"] == LABELS
    assert ens["gate"]["available"] and ens["gate"]["name"] == "spatial_gate_p2"
    blocks = {b["id"]: b for b in ens["blocks"]}
    assert blocks["test_cubes"]["state"] == "current"
    assert blocks["compare"]["state"] == "missing"       # no compare report: gate.json says so
    assert payload["can_freeze"] is True and payload["max_fields"] == 10
    assert ens["numbers_bytes"] > 0
    assert payload["fasrc_connected"] is False and "FASRC" in payload["fields_note"]
    fields = _by_fid(payload)
    assert [fid for fid in fields if fid.startswith("test-")] == [
        "test-00000", "test-00001", "test-00002"]
    assert all(f["available"] for fid, f in fields.items() if fid.startswith("test-"))
    assert fields["test-00000"]["kind"] == "test" and fields["test-00000"]["bytes"] > 0
    assert fields["test-00000"]["thumb_url"].startswith("/api/studies/candidates/thumb/test-00000")
    assert fields["blackout-00000"]["available"] and fields["blackout-00002"]["available"]
    assert env["labels"] == LABELS


def test_missing_compare_report_points_at_the_combiner_tab(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch)
    monkeypatch.setattr(remote.STATE, "ssh", None)
    blocks = {b["id"]: b for b in candidates.candidates(False)["ensemble"]["blocks"]}
    detail = blocks["compare"]["detail"]
    # The tab is labelled "Combiner" (nav.ts), not "Combiners".
    assert "Models › Combiner)" in detail
    assert "Combiners" not in detail


def test_candidates_are_read_only(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch)
    before = _snapshot(tmp_path)
    candidates.candidates(False)
    assert _snapshot(tmp_path) == before


def test_stale_cubes_block_the_freeze_with_the_reason(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch, labels=LABELS[:2])
    # A third member joined after the evaluation.
    payload = candidates.candidates(False)
    blocks = {b["id"]: b for b in payload["ensemble"]["blocks"]}
    assert blocks["test_cubes"]["state"] == "stale"
    assert "predates members 03" in blocks["test_cubes"]["detail"]
    assert payload["can_freeze"] is False and "re-evaluate" in payload["blocking"].lower()
    fields = _by_fid(payload)
    assert not fields["test-00000"]["available"]
    assert "predates members 03" in fields["test-00000"]["reason"]
    assert env["labels"] == LABELS[:2]


def test_blackout_fields_need_every_member(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch, blackout_labels=LABELS[:2])
    fields = _by_fid(candidates.candidates(False))
    assert not fields["blackout-00000"]["available"]
    assert "member(s) 03" in fields["blackout-00000"]["reason"]
    # A member file missing from one natural test field disables that field only.
    os.remove(tmp_path / "vis/ensemble/starfull/cubes/member2_00001.npy")
    fields = _by_fid(candidates.candidates(False))
    assert fields["test-00000"]["available"] and not fields["test-00001"]["available"]
    assert "member(s) 03" in fields["test-00001"]["reason"]


def test_real_tiles_need_cached_sr_for_every_member(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch)
    rf.point_store(tmp_path, monkeypatch)
    rf.make_poster(tmp_path / "poster")
    (entry,) = real_tiles.list_entries("poster")
    fps = {label: f"ckpt-{i}:1:1" for i, label in enumerate(LABELS)}
    monkeypatch.setattr(model_catalog, "active_member_labels", lambda: list(LABELS))
    monkeypatch.setattr(model_catalog, "member_fingerprints",
                        lambda labels: {lb: fps.get(lb) for lb in labels})
    tile = real_tiles.get_tile("poster", entry.id)
    lr = np.where(np.isfinite(tile.lr_e), tile.lr_e, 0.0).astype(np.float32)
    cache = experiments.CachedTileMembers("poster", entry.id, lr,
                                          lr_sha=model_catalog.array_sha(lr),
                                          runner=rf.StubRunner(), fingerprints=fps)
    for label in LABELS[:2]:
        cache.get(label)
    fid = f"real-poster-{entry.id}"
    field = _by_fid(candidates.candidates(False))[fid]
    assert not field["available"] and "member(s) 03" in field["reason"]
    cache.get(LABELS[2])
    field = _by_fid(candidates.candidates(False))[fid]
    assert field["available"] and field["kind"] == "real" and field["bytes"] > 0
    # A retrained member invalidates its cached SR.
    fps[LABELS[0]] = "ckpt-9:1:1"
    field = _by_fid(candidates.candidates(False))[fid]
    assert not field["available"] and "member(s) 01" in field["reason"]


def test_real_tiles_are_starfull_only(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch)
    payload = candidates.candidates(True)
    assert not [f for f in payload["fields"] if f["kind"] == "real"]
    assert payload["regime"] == "starless"


def test_field_ids_roundtrip():
    assert candidates.parse_field_id("test-00042") == ("test", "00042")
    assert candidates.parse_field_id("blackout-00007") == ("blackout", "00007")
    assert candidates.parse_field_id("real-poster-target_181255_new4") == (
        "real", "poster/target_181255_new4")
    assert candidates.field_id("real", "nexus/f200w-0214") == "real-nexus-f200w-0214"
    for bad in ("test-1", "real-nope-x", "../x", "test-00001/..", ""):
        try:
            candidates.parse_field_id(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_thumbnail_is_a_jpeg(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch)
    for fid in ("test-00001", "blackout-00002"):
        body = candidates.thumbnail(fid, False, 64)
        assert body[:3] == b"\xff\xd8\xff"


def test_ensemble_snapshot_counts_matching_experiments(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch)
    rf.point_store(tmp_path, monkeypatch)
    monkeypatch.setattr(model_catalog, "current_fingerprints", lambda specs=None: {"mean": "m1"})
    root = experiments.records_root()
    root.mkdir(parents=True)
    (root / "20260927-180342-e3237d.json").write_text(json.dumps({
        "id": "20260927-180342-e3237d", "fingerprints": {"mean": "m1"}, "tiles": [],
        "models": ["mean"], "summary": {}}))
    (root / "20260926-180342-aaaaaa.json").write_text(json.dumps({
        "id": "20260926-180342-aaaaaa", "fingerprints": {"mean": "old"}, "tiles": [],
        "models": ["mean"], "summary": {}}))
    blocks = {b["id"]: b for b in candidates.candidates(False)["ensemble"]["blocks"]}
    assert blocks["real"]["state"] == "current" and "1 " in blocks["real"]["detail"]


def test_fields_whose_core_cannot_be_fetched_are_refused(tmp_path, monkeypatch):
    make_env(tmp_path, monkeypatch)
    before = _by_fid(candidates.candidates(False))["test-00000"]
    assert before["bytes_upper_bound"] is True
    assert before["core_bytes"] < before["bytes"]
    assert before["largest_product_bytes"] <= before["core_bytes"]
    # The whole field may exceed the cache: only the core set must fit.
    monkeypatch.setattr(candidates, "FIELD_CACHE_BUDGET_BYTES", before["core_bytes"])
    field = _by_fid(candidates.candidates(False))["test-00000"]
    assert field["available"] and field["bytes"] > candidates.FIELD_CACHE_BUDGET_BYTES
    monkeypatch.setattr(candidates, "FIELD_CACHE_BUDGET_BYTES", before["core_bytes"] - 1)
    field = _by_fid(candidates.candidates(False))["test-00000"]
    assert not field["available"] and "too large to fetch back" in field["reason"]


def _blocks(payload):
    return {b["id"]: b for b in payload["ensemble"]["blocks"]}


def test_a_retrained_member_makes_the_cubes_stale(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    member = env["base"] / "member_01"
    (member / "checkpoint").write_text('model_checkpoint_path: "ckpt-9"\n')
    (member / "ckpt-9.index").write_bytes(b"retrained")
    payload = candidates.candidates(False)
    block = _blocks(payload)["test_cubes"]
    assert block["state"] == "stale"
    assert "member 01's checkpoint changed since the evaluation" in block["detail"].lower()
    assert payload["can_freeze"] is False


def test_a_refitted_gate_makes_the_cubes_stale(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    npz = env["regime"] / "spatial_gate_combiner" / "combiner.npz"
    npz.write_bytes(npz.read_bytes() + b"refit")
    block = _blocks(candidates.candidates(False))["test_cubes"]
    assert block["state"] == "stale"
    assert "the production gate changed since the evaluation" in block["detail"].lower()


def test_an_evaluation_without_fingerprints_is_stale(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    (env["regime"] / "eval_summary.json").write_text(json.dumps({
        "member_labels": LABELS, "eval_identity": {"records_fp": "fp"}}))
    block = _blocks(candidates.candidates(False))["test_cubes"]
    assert block["state"] == "stale" and "checkpoint fingerprints" in block["detail"].lower()


def test_a_member_newer_than_its_blackout_cube_disables_the_field(tmp_path, monkeypatch):
    env = make_env(tmp_path, monkeypatch)
    cube = env["blackout"] / "member1_00002.npy"
    index = env["base"] / "member_02" / "ckpt-5.index"
    # The cube was built before this checkpoint reached the machine. (rsync
    # keeps a pulled checkpoint's FASRC mtime, so its ctime — the pull — counts.)
    past = os.stat(index).st_ctime - 3600
    os.utime(cube, (past, past))
    fields = _by_fid(candidates.candidates(False))
    assert fields["test-00002"]["available"]
    assert not fields["blackout-00002"]["available"]
    reason = fields["blackout-00002"]["reason"]
    assert "member 02's checkpoint is newer than its blackout cube" in reason
    assert "delete cubes_blackout/blackout_index.json and run a combiner comparison" in reason
