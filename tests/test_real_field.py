"""Tests for the cached real-field diagnostics payload."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from euclid_polish.config import Config
from euclid_polish.eval.spatial_gate import MIX_LINEAR, SpatialGateCombiner, save_spatial_gate
from euclid_polish.eval.spatial_gate_fit import init_params
from euclid_polish.web.app import create_app
from euclid_polish.web.helpers import real_field
from euclid_polish.web.helpers.real_field import (
    _accumulate_diagnostics,
    _diagnostic_accumulators,
    _diagnostic_payload,
    _preserve_matching_member_cubes,
    _restore_matching_member_cubes,
)
from euclid_polish.web.routes import model as model_routes


def test_real_field_diagnostics_use_model_power_cross_correlation():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(64, 64, 4)).astype(np.float32)
    members = np.stack([base, base + rng.normal(0, 0.05, base.shape)], axis=0)
    acc = _diagnostic_accumulators({}, n_members=2)

    _accumulate_diagnostics(acc, members, {})
    payload = _diagnostic_payload(acc, ["m0", "m1"], {})

    json.dumps(payload)
    assert payload["version"] == 2
    assert "correlation" not in payload
    power = payload["model_power"]
    assert power["samples"] == 4
    assert power["pair_indices"] == [[0, 1]]
    assert len(power["r_pairs"]) == 1
    assert len(power["r_pairs"][0]) == len(power["k"])
    assert any(value is not None for value in power["r_cross"])


def test_matching_member_cubes_are_remapped_without_copying(tmp_path):
    cubes = tmp_path / "cubes"
    cubes.mkdir()
    for old_index, value in enumerate((10.0, 20.0, 30.0)):
        for tile in range(2):
            np.save(cubes / f"member{old_index}_{tile:03d}.npy", [value + tile])

    staging, preserved = _preserve_matching_member_cubes(
        cubes, ["a", "b", "c"], ["c", "a", "new"], count=2)
    for path in cubes.glob("*.npy"):
        path.unlink()
    _restore_matching_member_cubes(cubes, staging)

    assert preserved == 4
    np.testing.assert_allclose(np.load(cubes / "member0_000.npy"), [30.0])
    np.testing.assert_allclose(np.load(cubes / "member1_001.npy"), [11.0])
    assert not (cubes / "member2_000.npy").exists()


# ── production-only member runs (the gate's read members by default) ────────

LABELS = ["170·psnr", "171·psnr", "172·psnr"]


class _Member:
    def __init__(self, label: str) -> None:
        self.value = float(label.split("·")[0])

    def upsample_array(self, lr):
        base = np.kron(np.asarray(lr, np.float32), np.ones((2, 2, 1), np.float32))
        return base * self.value


class _Ensemble:
    built: list[dict] = []

    def __init__(self, _base, **kwargs) -> None:
        _Ensemble.built.append(kwargs)
        self.member_labels = list(kwargs["labels"])
        self.members = [_Member(label) for label in self.member_labels]


def _pruned_gate(active=(0, 2)) -> SpatialGateCombiner:
    params = init_params(len(active), 4, 8, False, [0, 0, 0, 0], seed=1)
    return SpatialGateCombiner(list(LABELS), params, width=8, use_lr=False,
                               active_members=tuple(active), mix_space=MIX_LINEAR)


@pytest.fixture
def field_env(tmp_path, monkeypatch):
    """A 2×2-tile real field under ``tmp_path`` with a pruned production gate."""
    monkeypatch.setattr(Config, "EUCLID_INFERENCE_DIR", str(tmp_path / "inference"))
    monkeypatch.setattr(Config, "VIS_DIR", str(tmp_path / "vis"))
    monkeypatch.setattr(real_field, "TILE_SIZE", 8)
    monkeypatch.setattr(real_field, "GRID_SIDE", 2)
    rng = np.random.default_rng(5)
    lr = rng.uniform(1.0, 20.0, (16, 16, 4)).astype(np.float32)
    monkeypatch.setattr(real_field, "_load_or_download_lr", lambda *_args: lr)
    monkeypatch.setattr(real_field.ensemble_registry, "regime_labels",
                        lambda *_args, **_kwargs: list(LABELS))
    monkeypatch.setattr(real_field, "EnsembleModel", _Ensemble)
    # No checkpoints here: every member's fingerprint is None (never the
    # live ckpt/ensemble's).
    monkeypatch.setattr(real_field, "member_fingerprints",
                        lambda _base, labels: dict.fromkeys(labels))
    _Ensemble.built.clear()
    gate = _pruned_gate()
    save_spatial_gate(gate, str(tmp_path / "vis" / "ensemble" / "starfull"
                               / "spatial_gate_combiner"))
    return {"gate": gate, "lr": lr}


def _cubes() -> Path:
    return real_field.field_dir(real_field.field_id(1.0, 2.0)) / "cubes"


def _tick(*_args) -> None:
    pass


def test_cache_runs_only_the_members_the_production_gate_reads(field_env):
    manifest = real_field.cache_real_field(1.0, 2.0, progress=_tick)

    assert _Ensemble.built == [{"starless": False, "labels": ["170·psnr", "172·psnr"]}]
    assert manifest["member_labels"] == LABELS
    assert manifest["run_members"] == [0, 2]
    assert manifest["run_member_labels"] == ["170·psnr", "172·psnr"]
    assert manifest["member_scope"] == "gate"
    assert manifest["pca_n"] == 1
    cubes = _cubes()
    assert not (cubes / "member1_000.npy").exists()
    stack = np.stack([np.load(cubes / f"member{i}_000.npy") for i in (0, 2)])
    np.testing.assert_allclose(np.load(cubes / "sr_000.npy"), stack.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(np.load(cubes / "comb_spatial_gate_000.npy"),
                               field_env["gate"].apply_field(stack), rtol=1e-5)
    diagnostics = json.loads((cubes.parent / "diagnostics.json").read_text())
    assert diagnostics["member_labels"] == ["170·psnr", "172·psnr"]
    assert diagnostics["member_scope"] == "gate"
    assert diagnostics["n_ensemble_members"] == 3


def test_all_members_option_adds_the_rest_and_reuses_cached_members(field_env):
    real_field.cache_real_field(1.0, 2.0, progress=_tick)
    _Ensemble.built.clear()

    manifest = real_field.cache_real_field(1.0, 2.0, progress=_tick, all_members=True)

    # Only the member the gate skipped runs; the two cached ones are reused.
    assert _Ensemble.built == [{"starless": False, "labels": ["171·psnr"]}]
    assert manifest["run_members"] == [0, 1, 2]
    assert manifest["member_scope"] == "all"
    assert manifest["pca_n"] == 2
    cubes = _cubes()
    stack = np.stack([np.load(cubes / f"member{i}_003.npy") for i in range(3)])
    np.testing.assert_allclose(np.load(cubes / "sr_003.npy"), stack.mean(axis=0), rtol=1e-6)


def test_a_gate_only_cache_keeps_member_cubes_it_did_not_run(field_env):
    real_field.cache_real_field(1.0, 2.0, progress=_tick, all_members=True)
    before = np.load(_cubes() / "member1_002.npy")
    _Ensemble.built.clear()

    manifest = real_field.cache_real_field(1.0, 2.0, progress=_tick)

    assert _Ensemble.built == []
    assert manifest["run_members"] == [0, 2]
    np.testing.assert_array_equal(np.load(_cubes() / "member1_002.npy"), before)


def test_combiners_that_read_unrun_members_are_not_applied():
    class _Reads:
        def __init__(self, labels, needed):
            self.member_labels = list(labels)
            self.needed = needed

        def needed_member_indices(self):
            return list(self.needed)

    gate, rbf = _Reads(LABELS, [0, 2]), _Reads(LABELS, [0, 1, 2])
    combiners = {"spatial_gate": gate, "raw_incremental_minmeanmax_rbf": rbf}

    assert real_field._run_member_indices(combiners, LABELS, all_members=False) == ([0, 2], "gate")
    assert real_field._run_member_indices(combiners, LABELS, all_members=True) == ([0, 1, 2], "all")
    assert real_field._run_member_indices({}, LABELS, all_members=False) == ([0, 1, 2], "all")
    assert real_field._applicable_combiners(combiners, ["170·psnr", "172·psnr"]) == {"spatial_gate": gate}
    assert set(real_field._applicable_combiners(combiners, LABELS)) == set(combiners)


def test_a_gate_fitted_before_members_joined_still_picks_the_run(field_env, monkeypatch):
    # 173 joined after the fit: the gate still reads 170 and 172, now at the
    # same positions of the four-member ensemble.
    joined = [*LABELS, "173·psnr"]
    monkeypatch.setattr(real_field.ensemble_registry, "regime_labels",
                        lambda *_args, **_kwargs: list(joined))

    manifest = real_field.cache_real_field(1.0, 2.0, progress=_tick)

    assert manifest["member_labels"] == joined
    assert manifest["run_member_labels"] == ["170·psnr", "172·psnr"]
    assert manifest["combiner_kinds"] == ["spatial_gate"]


def test_refresh_reuses_a_gate_only_cache_and_asks_for_missing_members(field_env):
    real_field.cache_real_field(1.0, 2.0, progress=_tick)
    identifier = real_field.field_id(1.0, 2.0)

    manifest = real_field.refresh_real_field_combiners(identifier, progress=_tick)

    assert manifest["run_members"] == [0, 2]
    assert manifest["combiner_kinds"] == ["spatial_gate"]
    with pytest.raises(RuntimeError, match="member cache is stale"):
        real_field.refresh_real_field_combiners(
            identifier, progress=_tick, all_members=True)


def test_refresh_route_passes_the_all_members_option(monkeypatch):
    seen: list[tuple[str, bool]] = []

    def refresh(_identifier, *, progress, all_members=False):
        seen.append(("refresh", all_members))
        raise RuntimeError("real-field member cache is stale")

    def cache(_ra, _dec, *, progress, all_members=False):
        seen.append(("cache", all_members))
        return {}

    def spawn(*, label, target):
        target(_Cap())
        return "job"

    monkeypatch.setattr(model_routes, "latest_field",
                        lambda: {"field_id": "f", "ra": 1.0, "dec": 2.0})
    monkeypatch.setattr(model_routes, "refresh_real_field_combiners", refresh)
    monkeypatch.setattr(model_routes, "cache_real_field", cache)
    monkeypatch.setattr(model_routes.REGISTRY, "spawn", spawn)
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        assert client.post("/inference/refresh-combiners").status_code == 200
        assert client.post("/inference/refresh-combiners",
                           data={"all_members": "1"}).status_code == 200

    assert seen == [("refresh", False), ("cache", False), ("refresh", True), ("cache", True)]


class _Cap:
    def tick(self, *_args, **_kwargs):
        pass


def test_the_cache_records_member_fingerprints_and_skips_a_continued_member(field_env, monkeypatch):
    fps = dict.fromkeys(LABELS, "ckpt-1")
    monkeypatch.setattr(real_field, "member_fingerprints",
                        lambda _base, labels: {lb: fps[lb] for lb in labels})
    manifest = real_field.cache_real_field(1.0, 2.0, progress=_tick)
    assert manifest["member_fps"] == fps
    _Ensemble.built.clear()
    fps["172·psnr"] = "ckpt-2"

    real_field.cache_real_field(1.0, 2.0, progress=_tick)

    assert _Ensemble.built == [{"starless": False, "labels": ["172·psnr"]}]


def test_refresh_calls_a_continued_member_stale(field_env, monkeypatch):
    fps = dict.fromkeys(LABELS, "ckpt-1")
    monkeypatch.setattr(real_field, "member_fingerprints",
                        lambda _base, labels: {lb: fps[lb] for lb in labels})
    real_field.cache_real_field(1.0, 2.0, progress=_tick)
    fps["170·psnr"] = "ckpt-2"

    with pytest.raises(RuntimeError, match="member cache is stale"):
        real_field.refresh_real_field_combiners(real_field.field_id(1.0, 2.0), progress=_tick)


def test_purge_drops_stale_members_and_renumbers_the_rest(field_env):
    real_field.cache_real_field(1.0, 2.0, progress=_tick, all_members=True)
    identifier = real_field.field_id(1.0, 2.0)
    kept = np.load(_cubes() / "member2_001.npy")

    out = real_field.purge_stale_real_fields({"170·psnr": None, "172·psnr": None})

    assert [r["dropped"] for r in out] == [["171·psnr"]] and out[0]["bytes_freed"] > 0
    manifest = real_field._read_manifest(identifier)
    assert manifest["member_labels"] == ["170·psnr", "172·psnr"]
    assert manifest["run_members"] == [0, 1] and manifest["combiner_kinds"] == []
    np.testing.assert_array_equal(np.load(_cubes() / "member1_001.npy"), kept)
    names = {p.name for p in _cubes().glob("*.npy")}
    assert not any(n.startswith(("sr_", "std_", "pca", "member2_")) for n in names)
    assert {f"lr_{t:03d}.npy" for t in range(4)} <= names
    assert real_field.purge_stale_real_fields({"170·psnr": None, "172·psnr": None}) == []


def test_purge_drops_every_member_of_a_cache_without_fingerprints(field_env):
    real_field.cache_real_field(1.0, 2.0, progress=_tick)
    path = real_field.manifest_path(real_field.field_id(1.0, 2.0))
    manifest = json.loads(path.read_text())
    del manifest["member_fps"]
    path.write_text(json.dumps(manifest))

    out = real_field.purge_stale_real_fields(dict.fromkeys(LABELS))

    assert out[0]["dropped"] == LABELS
    assert sorted(p.name[:3] for p in _cubes().glob("*.npy")) == ["lr_"] * 4
