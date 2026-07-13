from __future__ import annotations

import json

from scripts import lens_isolation_evaluate as cli


def test_evaluate_dry_run_exposes_random_crop_controls(tmp_path, capsys):
    rc = cli.main(
        [
            "--records-dir",
            str(tmp_path / "records"),
            "--ensemble-dir",
            str(tmp_path / "ensemble"),
            "--out-dir",
            str(tmp_path / "evaluation"),
            "--seed",
            "19",
            "--crop-size",
            "96",
            "--limit",
            "8",
            "--dry-run",
        ]
    )
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["seed"] == 19
    assert plan["crop_size"] == 96
    assert plan["limit"] == 8
    assert "source_baselines" not in plan
