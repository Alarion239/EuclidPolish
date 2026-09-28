"""Attached study fields (``euclid_polish/studies/fields.py`` + ``cache.py``):
one product packed at a time, uploaded, verified by the remote sha256 and
deleted locally; a sha mismatch fails; fetch into the bounded LRU cache with
the disk margin. A fake SSH whose "remote" is a tmp directory."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

from euclid_polish.studies import cache as study_cache
from euclid_polish.studies import fields
from euclid_polish.web.helpers import experiments, model_catalog, real_tiles
from tests import _real_fixtures as rf
from tests._studies_fixtures import LABELS, FakeRemote, _truth, make_env

REMOTE = "/n/holylabs/lab/me/EuclidPolish/study_fields/20260928-120000-a"


@pytest.fixture
def env(tmp_path, monkeypatch):
    out = make_env(tmp_path, monkeypatch)
    monkeypatch.setattr(experiments, "free_bytes", lambda _path: 10 ** 15)
    out["ssh"] = FakeRemote(tmp_path / "remote")
    return out


def _pack(env, fid, **kwargs):
    source = fields.open_field(fid, starless=False, labels=LABELS)
    return fields.pack_field(env["ssh"], source, f"{REMOTE}/{fid}",
                             staging=study_cache.staging_root(), **kwargs)


def _remote_npz(env, fid, name):
    with np.load(env["ssh"].local(f"{REMOTE}/{fid}/{name}.npz")) as handle:
        return handle["data"]


def test_a_test_field_uploads_every_product_verified(env):
    seen = []

    def on_product(name, done, total):
        seen.append(name)
        # Local disk never holds more than the one product being uploaded.
        assert len([p for p in study_cache.staging_root().rglob("*") if p.is_file()]) <= 1

    record = _pack(env, "test-00001", on_product=on_product)
    names = ["member_01", "member_02", "member_03", "mean", "gate", "lr", "hr"]
    assert list(record["products"]) == [*names, "truth.json", "field.json"]
    assert seen[:len(names)] == names
    assert record["state"] == "uploaded" and record["bytes"] > 0
    assert not [p for p in study_cache.staging_root().rglob("*") if p.is_file()]
    np.testing.assert_array_equal(_remote_npz(env, "test-00001", "hr"), _truth(1))
    np.testing.assert_array_equal(
        _remote_npz(env, "test-00001", "gate"),
        np.load(env["cubes"] / "comb_spatial_gate_00001.npy"))
    field_json = json.loads(env["ssh"].local(f"{REMOTE}/test-00001/field.json").read_text())
    assert field_json["labels"] == LABELS and field_json["kind"] == "test"
    assert field_json["products"]["hr"]["sha256"] == record["products"]["hr"]["sha256"]
    assert field_json["target"]["fwhm_arcsec"] == 0.066
    truth = json.loads(env["ssh"].local(f"{REMOTE}/test-00001/truth.json").read_text())
    assert truth["sources"][0]["type"] == "star"
    assert any(c.startswith("sha256sum ") for c in env["ssh"].commands)


def test_a_blackout_field_gets_mean_gate_and_hole_mask(env):
    record = _pack(env, "blackout-00002")
    assert {"mean", "gate", "lr", "hr", "mask"} <= set(record["products"])
    members = [np.load(env["blackout"] / f"member{i}_00002.npy") for i in range(3)]
    np.testing.assert_allclose(_remote_npz(env, "blackout-00002", "mean"),
                               np.mean(members, axis=0), rtol=1e-6)
    mask = _remote_npz(env, "blackout-00002", "mask")
    assert mask.dtype == np.uint8 and mask.shape == (16, 16, 4) and mask.any()
    # The uniform production gate of the fixture = the plain mean in linear space.
    np.testing.assert_allclose(_remote_npz(env, "blackout-00002", "gate"),
                               np.mean(members, axis=0), rtol=1e-4, atol=1e-3)


def test_a_sha_mismatch_fails_the_upload(env, tmp_path):
    env["ssh"] = FakeRemote(tmp_path / "remote2", corrupt={"member_02.npz"})
    with pytest.raises(fields.UploadError, match="member_02"):
        _pack(env, "test-00000")
    assert not [p for p in study_cache.staging_root().rglob("*") if p.is_file()]


def test_the_disk_margin_is_checked_before_each_product(env, monkeypatch):
    calls = []

    def free(_path):
        # Plenty of room for the first product, then the disk fills up.
        calls.append(1)
        return 10 ** 15 if len(calls) == 1 else experiments.MIN_FREE_BYTES

    monkeypatch.setattr(experiments, "free_bytes", free)
    with pytest.raises(experiments.DiskSpaceError):
        _pack(env, "test-00000")
    assert len(env["ssh"].pushed) == 1 and len(calls) == 2


SID = "20260928-120000-a"
CORE = ["field.json", "truth.json", "lr", "hr", "mean", "gate"]


def _fetch(env, record, cache, products=None):
    return fields.fetch_field(env["ssh"], SID, record, f"{REMOTE}/{record['fid']}", cache,
                              products=products)


def test_fetch_pulls_the_core_products_verified(env, tmp_path):
    record = _pack(env, "test-00001")
    cache = study_cache.FieldCache(tmp_path / "cache")
    out = _fetch(env, record, cache)
    assert sorted(out["fetched"]) == sorted(CORE) and out["skipped"] == []
    assert cache.is_cached(SID, "test-00001", fields.core_products(record), record["products"])
    assert not cache.is_cached(SID, "test-00001", ["member_01"], record["products"])
    np.testing.assert_array_equal(cache.load(SID, "test-00001", "hr"), _truth(1))
    # Every pull is one product file, never the whole field directory.
    assert all(p.rsplit("/", 1)[-1].endswith((".npz", ".json")) for p in env["ssh"].pulled)
    # A second fetch is a cache hit (no transfer).
    before = list(env["ssh"].pulled)
    assert _fetch(env, record, cache)["fetched"] == []
    assert env["ssh"].pulled == before


def test_members_are_fetched_explicitly(env, tmp_path):
    record = _pack(env, "test-00001")
    cache = study_cache.FieldCache(tmp_path / "cache")
    _fetch(env, record, cache)
    out = _fetch(env, record, cache, products=["member_02"])
    assert out["fetched"] == ["member_02"]
    np.testing.assert_array_equal(cache.load(SID, "test-00001", "member_02"),
                                  np.load(env["cubes"] / "member1_00001.npy"))
    with pytest.raises(ValueError):
        _fetch(env, record, cache, products=["member_99"])


def test_fetch_all_members_that_fit(env, tmp_path):
    record = _pack(env, "test-00001")
    member = record["products"]["member_01"]["bytes"]
    core = sum(record["products"][name]["bytes"] for name in CORE)
    cache = study_cache.FieldCache(tmp_path / "cache", budget=core + 2 * member + member // 2)
    _fetch(env, record, cache)
    out = _fetch(env, record, cache, products="members")
    assert out["fetched"] == ["member_01", "member_02"] and out["skipped"] == ["member_03"]
    assert cache.is_cached(SID, "test-00001", fields.core_products(record), record["products"])


def test_fetch_rejects_a_corrupted_remote_copy(env, tmp_path):
    record = _pack(env, "test-00001")
    remote_file = env["ssh"].local(f"{REMOTE}/test-00001/mean.npz")
    remote_file.write_bytes(remote_file.read_bytes()[:-3] + b"xyz")
    cache = study_cache.FieldCache(tmp_path / "cache")
    with pytest.raises(fields.UploadError, match="mean"):
        _fetch(env, record, cache)
    assert not cache.is_cached(SID, "test-00001", ["mean"], record["products"])
    assert not (cache.field_dir(SID, "test-00001") / "mean.npz").exists()


def test_the_budget_evicts_least_recently_used_products_of_other_fields(env, tmp_path):
    a = _pack(env, "test-00000")
    b = _pack(env, "test-00001")
    core = sum(a["products"][name]["bytes"] for name in CORE)
    cache = study_cache.FieldCache(tmp_path / "cache", budget=int(core * 1.6))
    _fetch(env, a, cache)
    _fetch(env, b, cache)
    assert cache.is_cached(SID, "test-00001", fields.core_products(b), b["products"])
    assert not cache.is_cached(SID, "test-00000", fields.core_products(a), a["products"])
    assert cache.total() <= cache.budget


def test_a_product_larger_than_the_budget_is_refused(env, tmp_path):
    record = _pack(env, "test-00000")
    cache = study_cache.FieldCache(tmp_path / "cache", budget=100)
    with pytest.raises(experiments.DiskSpaceError):
        _fetch(env, record, cache)


def test_fetch_refuses_to_break_the_disk_margin(env, tmp_path, monkeypatch):
    record = _pack(env, "test-00000")
    cache = study_cache.FieldCache(tmp_path / "cache")
    monkeypatch.setattr(experiments, "free_bytes",
                        lambda _path: experiments.MIN_FREE_BYTES + 100)
    with pytest.raises(experiments.DiskSpaceError):
        _fetch(env, record, cache)


def test_uploads_stage_one_file_at_a_time(env):
    _pack(env, "blackout-00002")
    assert env["ssh"].staged_counts and set(env["ssh"].staged_counts) == {1}


def test_a_real_tile_packs_member_sr_mean_and_gate(env, tmp_path, monkeypatch):
    rf.point_store(tmp_path, monkeypatch)
    rf.make_poster(tmp_path / "poster")
    (entry,) = real_tiles.list_entries("poster")
    tile = real_tiles.get_tile("poster", entry.id)
    lr = np.where(np.isfinite(tile.lr_e), tile.lr_e, 0.0).astype(np.float32)
    fps = model_catalog.member_fingerprints(LABELS)
    members = experiments.CachedTileMembers("poster", entry.id, lr,
                                            lr_sha=model_catalog.array_sha(lr),
                                            runner=rf.StubRunner(), fingerprints=fps)
    for label in LABELS:
        members.get(label)
    fid = f"real-poster-{entry.id}"
    record = _pack(env, fid)
    assert "hr" not in record["products"] and "mask" not in record["products"]
    stack = [members.get(label) for label in LABELS]
    np.testing.assert_allclose(_remote_npz(env, fid, "mean"), np.mean(stack, axis=0), rtol=1e-6)
    assert _remote_npz(env, fid, "gate").shape == stack[0].shape
    field_json = json.loads(env["ssh"].local(f"{REMOTE}/{fid}/field.json").read_text())
    assert field_json["wcs"]["lr"] is not None and field_json["wcs"]["sr"] is not None


def test_the_lru_follows_local_use_not_upload_time(env, tmp_path):
    b = _pack(env, "test-00001")                             # uploaded first …
    old = 1_000_000_000
    for path in env["ssh"].local(f"{REMOTE}/test-00001").iterdir():
        os.utime(path, (old, old))                           # … long ago
    a = _pack(env, "test-00000")
    c = _pack(env, "test-00002")
    core = sum(a["products"][name]["bytes"] for name in CORE)
    cache = study_cache.FieldCache(tmp_path / "cache", budget=int(core * 2.5))
    _fetch(env, a, cache)                                    # used first locally
    time.sleep(0.01)
    _fetch(env, b, cache)
    _fetch(env, c, cache)
    assert fields.core_products(b) and cache.is_cached(SID, "test-00001",
                                                        fields.core_products(b), b["products"])
    assert not cache.is_cached(SID, "test-00000", fields.core_products(a), a["products"])


def test_reservations_keep_two_jobs_inside_the_margin(env, tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "free_bytes",
                        lambda _path: experiments.MIN_FREE_BYTES + 1000)
    with study_cache.claim(tmp_path, 800):
        assert study_cache.reserved_bytes() == 800
        with pytest.raises(experiments.DiskSpaceError), study_cache.claim(tmp_path, 800):
            pass                                             # 800 + 800 > 1000 spare
        with study_cache.claim(tmp_path, 150):
            assert study_cache.reserved_bytes() == 950
    assert study_cache.reserved_bytes() == 0
