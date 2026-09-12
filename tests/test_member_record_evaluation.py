"""Per-member record scores use the member's own star-regime target."""

from __future__ import annotations

import json
import os

import pytest

import euclid_polish.ensemble as ensemble_module


@pytest.mark.parametrize(("starless", "target"), [(True, "clean_test"),
                                                 (False, "hr_test")])
def test_member_record_evaluation_reads_regime_target(
    tmp_path, monkeypatch, starless, target,
):
    member = tmp_path / "member_00"
    member.mkdir()
    (member / "origin.json").write_text(json.dumps({"starless": starless}))
    read_paths: list[str] = []

    def fake_read(path, num_images=None):
        del num_images
        read_paths.append(os.path.basename(path))
        return []

    monkeypatch.setattr(ensemble_module, "Model", lambda *a, **k: object())
    monkeypatch.setattr(ensemble_module.ImageSet, "read", fake_read)

    out = ensemble_module.evaluate_member_on_records(
        str(member), str(tmp_path), subset="test",
    )

    assert out["n_scored"] == 0
    assert [p.split(".")[0] for p in read_paths] == ["dirty_test", target]
