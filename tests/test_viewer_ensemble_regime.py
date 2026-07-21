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
    assert goal["label"] == "Clean (starless goal)"
    assert "hr" not in {
        tier["key"] for tier in vd._ensemble_meta({"mode": "starfull"})["tiers"]
    }


def test_ensemble_goal_cube_uses_regime_target(monkeypatch):
    seen: list[tuple[str, int, str, int]] = []
    monkeypatch.setattr(vd, "_ensemble_manifest", _manifest)

    def record_cube(sub: str, n_read: int, kind: str, rec_index: int):
        seen.append((sub, n_read, kind, rec_index))
        return np.zeros((2, 2, 4), np.float32), 0.05

    monkeypatch.setattr(vd, "_ensemble_record_cube", record_cube)

    _clean, clean_info = vd._ensemble_cube(
        0, "hr", {"mode": "starless"})
    _hr, hr_info = vd._ensemble_cube(0, "hr", {"mode": "starfull"})

    assert seen == [
        ("test", 4, "clean", 3),
        ("test", 4, "hr", 3),
    ]
    assert clean_info["label"].startswith("Clean (starless goal)")
    assert hr_info["label"].startswith("HR")
