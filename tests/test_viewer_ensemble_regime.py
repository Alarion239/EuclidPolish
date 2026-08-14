"""Regime-aware goal selection in the ensemble disagreement viewer."""

from __future__ import annotations

import numpy as np

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


def test_ensemble_sr_tier_serves_primary_combiner(tmp_path, monkeypatch):
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
    assert seen == [vd.RAW_INCREMENTAL_MINMEANMAX_RBF_KIND]
    assert "minibatched convex all-asinh RBF" in info["label"]


def test_ensemble_meta_does_not_duplicate_primary_combiner(monkeypatch):
    manifest = {
        "subset": "test", "indices": [3], "member_labels": ["00·x", "01·y"]}
    monkeypatch.setattr(vd, "_ensemble_manifest", lambda _starless: manifest)
    monkeypatch.setattr(vd, "_sky_records_local_dir", lambda: "")
    monkeypatch.setattr(vd, "_load_field_combiner", lambda *_args: object())

    tiers = vd._ensemble_meta({"mode": "starfull"})["tiers"]
    keys = [tier["key"] for tier in tiers]

    assert keys.count("sr") == 1
    assert vd.COMBINER_MODELS[
        vd.RAW_INCREMENTAL_MINMEANMAX_RBF_KIND].cube_prefix not in keys
    assert "minibatched convex all-asinh RBF" in next(
        tier["label"] for tier in tiers if tier["key"] == "sr")


def test_ensemble_meta_hides_sr_when_primary_combiner_is_unavailable(monkeypatch):
    manifest = {
        "subset": "test", "indices": [3], "member_labels": ["00·x", "01·y"]}
    monkeypatch.setattr(vd, "_ensemble_manifest", lambda _starless: manifest)
    monkeypatch.setattr(vd, "_sky_records_local_dir", lambda: "")
    monkeypatch.setattr(vd, "_load_field_combiner", lambda *_args: None)

    meta = vd._ensemble_meta({"mode": "starfull"})

    assert "sr" not in {tier["key"] for tier in meta["tiers"]}
    assert meta["default_tier"] == "lr"
