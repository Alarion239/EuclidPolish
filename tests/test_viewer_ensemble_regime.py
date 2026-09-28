"""Regime-aware goal selection in the ensemble disagreement viewer."""

from __future__ import annotations

import numpy as np

from euclid_polish.eval.spatial_gate import MIX_LINEAR, SpatialGateCombiner, save_spatial_gate
from euclid_polish.eval.spatial_gate_fit import init_params
from euclid_polish.web.helpers import viewer_data as vd


def _manifest(_starless: bool) -> dict:
    return {"subset": "test", "indices": [3], "member_labels": []}


def test_starless_viewer_advertises_clean_goal(tmp_path, monkeypatch):
    monkeypatch.setattr(vd, "_ensemble_manifest", _manifest)
    monkeypatch.setattr(vd, "_sky_records_local_dir", lambda: str(tmp_path))
    (tmp_path / "clean_test.tfrecord").touch()

    meta = vd._ensemble_meta({"mode": "starless"})

    goal = next(tier for tier in meta["tiers"] if tier["key"] == "hr")
    blurred_goal = next(tier for tier in meta["tiers"] if tier["key"] == "bhr")
    assert goal["label"] == "Clean (starless goal)"
    assert blurred_goal["label"] == "BHR (blurred Clean)"
    assert {"hr", "bhr"}.isdisjoint({
        tier["key"] for tier in vd._ensemble_meta({"mode": "starfull"})["tiers"]
    })


def test_ensemble_goal_cubes_use_raw_and_blurred_regime_target(monkeypatch):
    seen: list[tuple[str, int, str, int, bool]] = []
    monkeypatch.setattr(vd, "_ensemble_manifest", _manifest)

    def record_cube(sub: str, n_read: int, kind: str, rec_index: int,
                    *, blurred_fwhm_arcsec: float | None = None):
        seen.append((sub, n_read, kind, rec_index, blurred_fwhm_arcsec))
        return np.zeros((2, 2, 4), np.float32), 0.05

    monkeypatch.setattr(vd, "_ensemble_record_cube", record_cube)

    _clean, clean_info = vd._ensemble_cube(
        0, "hr", {"mode": "starless"})
    _blurred_clean, blurred_clean_info = vd._ensemble_cube(
        0, "bhr", {"mode": "starless"})
    _hr, hr_info = vd._ensemble_cube(0, "hr", {"mode": "starfull"})
    _bhr, bhr_info = vd._ensemble_cube(0, "bhr", {"mode": "starfull"})

    assert seen == [
        ("test", 4, "clean", 3, None),
        ("test", 4, "clean", 3, 0.066),
        ("test", 4, "hr", 3, None),
        ("test", 4, "hr", 3, 0.066),
    ]
    assert clean_info["label"].startswith("Clean (starless goal)")
    assert blurred_clean_info["label"].startswith("BHR (blurred Clean)")
    assert hr_info["label"].startswith("HR")
    assert bhr_info["label"].startswith("BHR (blurred HR)")


def test_ensemble_sr_tier_serves_the_production_combiner(tmp_path, monkeypatch):
    manifest = {
        "subset": "test", "indices": [3], "member_labels": ["00·x", "01·y"]}
    monkeypatch.setattr(vd, "_ensemble_manifest", lambda _starless: manifest)
    monkeypatch.setattr(vd, "_ensemble_cubes_dir", lambda _starless: str(tmp_path))
    expected = np.full((6, 6, 4), 7.0, np.float32)
    seen: list[str] = []

    def combined(_starless, rec_index, labels, model_kind):
        assert rec_index == 3
        assert labels == ["00·x", "01·y"]
        seen.append(model_kind)
        return expected

    monkeypatch.setattr(vd, "_combiner_field_cube", combined)

    cube, info = vd._ensemble_cube(0, "sr", {"mode": "starfull"})

    assert np.array_equal(cube, expected)
    # Contract C6: ``sr`` is ACTIVE_COMBINER_KINDS[0], the spatial gate.
    assert seen == [vd.PRODUCTION_COMBINER_KIND] == [vd.SPATIAL_GATE_KIND]
    assert info["label"].startswith("SR · production gate")


def test_ensemble_meta_does_not_duplicate_the_production_combiner(monkeypatch):
    manifest = {
        "subset": "test", "indices": [3], "member_labels": ["00·x", "01·y"]}
    monkeypatch.setattr(vd, "_ensemble_manifest", lambda _starless: manifest)
    monkeypatch.setattr(vd, "_sky_records_local_dir", lambda: "")
    monkeypatch.setattr(vd, "_load_field_combiner", lambda *_args: object())

    tiers = vd._ensemble_meta({"mode": "starfull"})["tiers"]
    keys = [tier["key"] for tier in tiers]

    assert keys.count("sr") == 1
    assert vd.COMBINER_MODELS[vd.SPATIAL_GATE_KIND].cube_prefix not in keys
    # The legacy RBF gets no tier: Models › Images shows one SR (production).
    assert vd.COMBINER_MODELS[
        vd.RAW_INCREMENTAL_MINMEANMAX_RBF_KIND].cube_prefix not in keys
    assert next(tier["label"] for tier in tiers
                if tier["key"] == "sr") == "SR · production gate"


def test_ensemble_meta_hides_sr_when_primary_combiner_is_unavailable(monkeypatch):
    manifest = {
        "subset": "test", "indices": [3], "member_labels": ["00·x", "01·y"]}
    monkeypatch.setattr(vd, "_ensemble_manifest", lambda _starless: manifest)
    monkeypatch.setattr(vd, "_sky_records_local_dir", lambda: "")
    monkeypatch.setattr(vd, "_load_field_combiner", lambda *_args: None)

    meta = vd._ensemble_meta({"mode": "starfull"})

    assert "sr" not in {tier["key"] for tier in meta["tiers"]}
    assert meta["default_tier"] == "lr"


def test_sr_tier_reads_only_the_members_a_pruned_gate_needs(tmp_path, monkeypatch):
    labels = ["00·x", "01·y", "02·z"]
    params = init_params(2, 4, 8, False, [0, 0, 0, 0], seed=1)
    gate = SpatialGateCombiner(labels, params, width=8, use_lr=False,
                               active_members=(0, 2), mix_space=MIX_LINEAR)
    save_spatial_gate(gate, str(tmp_path / "spatial_gate_combiner"))
    cubes = tmp_path / "cubes"
    cubes.mkdir()
    rng = np.random.default_rng(3)
    stack = rng.uniform(0.0, 50.0, (3, 8, 8, 4)).astype(np.float32)
    # The pruned member's cube is absent: the bucket must still serve ``sr``.
    for index in gate.needed_member_indices():
        np.save(cubes / f"member{index}_00003.npy", stack[index])
    monkeypatch.setattr(vd, "_ensemble_cubes_dir", lambda _starless: str(cubes))
    vd._COMB_CUBE_CACHE.clear()

    out = vd._combiner_field_cube(False, 3, labels)

    np.testing.assert_allclose(out, gate.apply_field(stack), rtol=1e-5, atol=1e-5)


def test_sr_tier_maps_a_gate_onto_a_cube_stack_with_joined_members(tmp_path, monkeypatch):
    labels = ["00·x", "01·y", "02·z"]
    params = init_params(2, 4, 8, False, [0, 0, 0, 0], seed=1)
    gate = SpatialGateCombiner(labels, params, width=8, use_lr=False,
                               active_members=(1, 2), mix_space=MIX_LINEAR)
    save_spatial_gate(gate, str(tmp_path / "spatial_gate_combiner"))
    cubes = tmp_path / "cubes"
    cubes.mkdir()
    # The cubes were evaluated after "03·new" joined in front of the others.
    cube_labels = ["03·new", *labels]
    rng = np.random.default_rng(4)
    stack = rng.uniform(0.0, 50.0, (4, 8, 8, 4)).astype(np.float32)
    for index in range(4):
        np.save(cubes / f"member{index}_00001.npy", stack[index])
    monkeypatch.setattr(vd, "_ensemble_cubes_dir", lambda _starless: str(cubes))
    vd._COMB_CUBE_CACHE.clear()

    out = vd._combiner_field_cube(False, 1, cube_labels)

    np.testing.assert_allclose(out, gate.apply_field(stack[[2, 3]]), rtol=1e-5, atol=1e-5)
