"""Experiments on real tiles (contract C9, plan WP-B2 T5): member SRs computed
once per tile and cached per checkpoint fingerprint, specs applied into the
output store, real-data metrics per (tile, spec), persisted records."""

from __future__ import annotations

import json

import numpy as np
import pytest
from astropy.io import fits

from euclid_polish.web.helpers import experiments, model_catalog, real_tiles
from euclid_polish.web.jobs import JobCancelled
from tests import _real_fixtures as fx


@pytest.fixture
def world(tmp_path, monkeypatch):
    store = fx.point_store(tmp_path, monkeypatch)
    regime = fx.stub_regime(tmp_path, monkeypatch)
    fx.make_nexus_field(n_tiles=2)
    return {**store, **regime}


def test_plan_validates_tiles_and_specs(world):
    entries, runnable, skipped = experiments.plan(
        [("nexus", "f200w-0000")], ["mean", "gate:two", "rbf"])
    assert [e.ref for e in entries] == ["nexus/f200w-0000"]
    assert [s.spec for s in runnable] == ["mean", "gate:two"]
    assert "not fitted" in skipped["rbf"]
    with pytest.raises(ValueError, match="unknown model spec"):
        experiments.plan([("nexus", "f200w-0000")], ["member:member_9"])
    with pytest.raises(ValueError, match="no runnable model"):
        experiments.plan([("nexus", "f200w-0000")], ["rbf"])
    with pytest.raises(ValueError, match="at least one"):
        experiments.plan([], ["mean"])
    with pytest.raises(real_tiles.RealTileError):
        experiments.plan([("nexus", "f200w-0042")], ["mean"])


def test_experiment_computes_members_once_and_scores_every_spec(world):
    runner = fx.StubRunner()
    ticks: list[tuple[int, int, str]] = []
    record = experiments.run_experiment(
        [("nexus", "f200w-0000")], ["mean", "member:member_1", "production", "gate:two", "rbf"],
        runner=runner, progress=lambda *a: ticks.append(a), label="unit")
    assert record["status"] == "done"
    assert sorted(runner.calls) == sorted(fx.LABELS)        # each member exactly once
    assert "rbf" in record["skipped"]
    results = record["results"]["nexus/f200w-0000"]
    assert set(results) == {"mean", "member:member_1", "production", "gate:two"}
    vis = {spec: results[spec]["metrics"]["per_band"]["VIS"] for spec in results}
    # member k is the flux-conserving SR × k; mean of 1,2,3 = 2; gate:two = 1.5
    assert vis["member:member_1"]["flux_ratio"] == pytest.approx(1.0, rel=1e-5)
    assert vis["member:member_1"]["hole_pct"] == 0.0
    assert vis["member:member_1"]["median_R"] == pytest.approx(1.0, abs=1e-4)
    assert vis["mean"]["flux_ratio"] == pytest.approx(2.0, rel=1e-5)
    assert vis["production"]["flux_ratio"] == pytest.approx(2.0, rel=1e-4)
    assert vis["gate:two"]["flux_ratio"] == pytest.approx(1.5, rel=1e-4)
    # gate specs report which members carry the bright-pixel weight
    core = results["production"]["metrics"]["gate_core_weights"]["VIS"]
    assert {row[0] for row in core} <= set(fx.LABELS)
    assert sum(row[1] for row in core) == pytest.approx(1.0, abs=1e-3)
    assert record["summary"]["mean"]["n_tiles"] == 1
    assert record["counts"]["members_computed"] == 3
    assert ticks and ticks[-1][0] == ticks[-1][1]
    # outputs + record persisted
    outputs = model_catalog.list_outputs("nexus", "f200w-0000")
    assert set(outputs) == set(results)
    assert outputs["mean"]["experiment_id"] == record["id"]
    stored = experiments.get_experiment(record["id"])
    assert stored["status"] == "done" and stored["label"] == "unit"
    json.dumps(stored, allow_nan=False)
    assert experiments.list_experiments()[0]["id"] == record["id"]
    assert experiments.experiments_for_tile("nexus", "f200w-0000") == [record["id"]]


def test_rerun_reuses_outputs_and_retraining_a_member_recomputes_only_it(world):
    runner = fx.StubRunner()
    experiments.run_experiment([("nexus", "f200w-0000")], ["mean", "member:member_1"],
                               runner=runner)
    assert len(runner.calls) == 3
    again = experiments.run_experiment([("nexus", "f200w-0000")], ["mean", "member:member_1"],
                                       runner=runner)
    assert len(runner.calls) == 3                            # nothing recomputed
    assert again["counts"]["outputs_reused"] == 2
    assert {r["state"] for r in again["results"]["nexus/f200w-0000"].values()} == {"reused"}
    world["fingerprints"]["2·psnr"] = "ckpt-9:11:200"         # member 2 retrained
    third = experiments.run_experiment([("nexus", "f200w-0000")], ["mean", "member:member_1"],
                                       runner=runner)
    assert runner.calls[3:] == ["2·psnr"]                    # only member 2 re-ran
    states = {spec: r["state"] for spec, r in third["results"]["nexus/f200w-0000"].items()}
    assert states == {"mean": "computed", "member:member_1": "reused"}


def test_a_changed_lr_invalidates_the_member_cache(world):
    runner = fx.StubRunner()
    experiments.run_experiment([("nexus", "f200w-0000")], ["member:member_1"], runner=runner)
    directory = real_tiles.list_entries("nexus")[0]
    path = (real_tiles.jwst_euclid.nexus_field_root() / directory.extras["field_id"]
            / directory.extras["lr_file"])
    with fits.open(path, mode="update") as hdul:
        hdul[0].data = hdul[0].data * 2
    experiments.run_experiment([("nexus", "f200w-0000")], ["member:member_1"], runner=runner)
    assert runner.calls == ["1·psnr", "1·psnr"]


def test_cancel_marks_the_record(world):
    runner = fx.StubRunner()
    calls = {"n": 0}

    def check():
        calls["n"] += 1
        if calls["n"] > 2:
            raise JobCancelled()

    with pytest.raises(JobCancelled):
        experiments.run_experiment(
            [("nexus", "f200w-0000"), ("nexus", "f200w-0001")], ["mean"],
            runner=runner, check_cancelled=check, experiment_id="20260926-000000-abcdef")
    assert experiments.get_experiment("20260926-000000-abcdef")["status"] == "cancelled"


def test_delete_tile_outputs_removes_outputs_and_cache(world):
    experiments.run_experiment([("nexus", "f200w-0000")], ["mean"], runner=fx.StubRunner())
    assert experiments.member_cache_dir("nexus", "f200w-0000").is_dir()
    result = experiments.delete_tile_outputs("nexus", "f200w-0000")
    assert result["removed_count"] >= 2 + 6 and result["cache_bytes_freed"] > 0
    assert model_catalog.list_outputs("nexus", "f200w-0000") == {}
    assert not experiments.member_cache_dir("nexus", "f200w-0000").exists()


def test_unknown_experiment_ids(world):
    for bad in ("nope", "../../x", "20260926-000000-zzzzzz"):
        with pytest.raises(KeyError):
            experiments.get_experiment(bad)


def test_gate_core_weights_ranks_members():
    weights = np.zeros((4, 4, 3, 1), np.float32)
    weights[..., 0, 0], weights[..., 1, 0], weights[..., 2, 0] = 0.7, 0.2, 0.1
    lr = np.arange(4, dtype=np.float32).reshape(2, 2, 1) + 1.0
    out = experiments.gate_core_weights(weights, lr, ["a", "b", "c"], ["VIS"])
    assert out["VIS"][0] == ["a", 0.7] and [row[0] for row in out["VIS"]] == ["a", "b", "c"]


def test_unknown_spec_errors_keep_their_quotes(world):
    for spec in ("gate:doesnotexist", "member:member_9"):
        with pytest.raises(ValueError) as info:
            experiments.plan([("nexus", "f200w-0000")], [spec])
        assert str(info.value) == f"unknown model spec {model_catalog.canonical_spec(spec)!r}"


def _cache_files():
    return sorted(p.name for p in model_catalog.member_cache_root().rglob("*.npy"))


def test_member_cache_is_bounded_and_evicts_least_recently_used(world, monkeypatch):
    one = 80 * 80 * 4 * 4 + 128                    # one member SR (.npy) of a 40² tile
    monkeypatch.setattr(experiments, "MEMBER_CACHE_BUDGET_BYTES", 2 * one + one // 2)
    runner = fx.StubRunner()
    first = experiments.run_experiment([("nexus", "f200w-0000")], ["mean"], runner=runner)
    assert first["status"] == "done"
    assert first["counts"]["members_computed"] == 3 and first["counts"]["members_evicted"] == 1
    assert _cache_files() == ["member_2.npy", "member_3.npy"]      # member 1 was oldest
    experiments.run_experiment([("nexus", "f200w-0000")], ["member:member_3"], runner=runner)
    assert runner.calls == fx.LABELS                               # member 3 reused (touched)
    second = experiments.run_experiment([("nexus", "f200w-0001")], ["member:member_1"],
                                        runner=runner)
    assert second["counts"]["members_evicted"] == 1
    usage = sum(p.stat().st_size for p in model_catalog.member_cache_root().rglob("*.npy"))
    assert usage <= experiments.MEMBER_CACHE_BUDGET_BYTES
    # the least recently USED entry went: member 2 (member 3 was just reused)
    remaining = {(p.parent.name, p.name) for p in model_catalog.member_cache_root().rglob("*.npy")}
    assert remaining == {("f200w-0000", "member_3.npy"), ("f200w-0001", "member_1.npy")}


def test_member_cache_is_skipped_when_disk_space_is_low(world, monkeypatch):
    # the outputs fit above the free-space margin, the member cache does not
    monkeypatch.setattr(experiments, "free_bytes",
                        lambda _path: experiments.MIN_FREE_BYTES + 10 ** 8)
    record = experiments.run_experiment([("nexus", "f200w-0000")], ["mean"],
                                        runner=fx.StubRunner())
    assert record["status"] == "done"
    assert record["counts"]["members_not_cached"] == 3 and _cache_files() == []
    assert model_catalog.list_outputs("nexus", "f200w-0000")["mean"]


def test_plan_refuses_outputs_that_do_not_fit_on_disk(world, monkeypatch):
    monkeypatch.setattr(experiments, "free_bytes", lambda _path: experiments.MIN_FREE_BYTES)
    with pytest.raises(experiments.DiskSpaceError) as info:
        experiments.plan([("nexus", "f200w-0000"), ("nexus", "f200w-0001")], ["mean", "production"])
    assert info.value.needed >= 4 * 80 * 80 * 4 * 4 and "free" in str(info.value)


def test_default_runner_is_told_which_members_the_job_needs(world, monkeypatch):
    made: list[dict] = []

    def factory(*args, **kwargs):
        made.append(kwargs)
        return fx.StubRunner()

    monkeypatch.setattr(model_catalog, "EnsembleMemberRunner", factory)
    experiments.run_experiment([("nexus", "f200w-0000")], ["gate:two", "member:member_1"])
    assert made == [{"labels": ["1·psnr", "2·psnr"]}]


def test_default_runner_skips_members_of_specs_already_current(world, monkeypatch):
    """A spec whose output is already current on every tile needs no member:
    the runner is only told about the members still to run."""
    experiments.run_experiment([("nexus", "f200w-0000")], ["mean"], runner=fx.StubRunner())
    made: list[dict] = []

    def factory(*args, **kwargs):
        made.append(kwargs)
        return fx.StubRunner()

    monkeypatch.setattr(model_catalog, "EnsembleMemberRunner", factory)
    experiments.run_experiment([("nexus", "f200w-0000")], ["mean", "member:member_3"])
    assert made == [{"labels": ["3·psnr"]}]


def test_member_runner_widens_when_asked_for_a_member_past_its_prefix(monkeypatch):
    active = ["1·psnr", "2·psnr", "3·psnr"]
    monkeypatch.setattr(model_catalog, "active_member_labels", lambda: list(active))
    built: list[dict] = []

    class FakeEnsemble:
        def __init__(self, base_dir, **kwargs):
            built.append(kwargs)
            self.member_labels = active[:kwargs.get("n_members") or len(active)]

        def member_arrays(self, lr, indices):
            return np.stack([lr * (i + 1) for i in indices])

    runner = model_catalog.EnsembleMemberRunner("/nowhere", factory=FakeEnsemble,
                                                labels=["1·psnr"])
    lr = np.ones((2, 2, 4), np.float32)
    np.testing.assert_allclose(runner.predict(lr, "1·psnr"), 1.0)
    np.testing.assert_allclose(runner.predict(lr, "3·psnr"), 3.0)   # rebuilt uncapped
    assert built == [{"starless": False, "n_members": 1}, {"starless": False}]
    with pytest.raises(KeyError):
        runner.predict(lr, "9·psnr")
