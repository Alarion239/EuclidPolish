from __future__ import annotations

import json
import os

from scripts import lens_isolation_generate as cli


def test_dry_run_reports_isolated_generation_plan(tmp_path, capsys):
    out = str(tmp_path / "lens-experiment" / "records")
    rc = cli.main(
        [
            "--out-dir",
            out,
            "--ntrain",
            "4",
            "--nvalid",
            "2",
            "--ntest",
            "2",
            "--workers",
            "3",
            "--seed",
            "17",
            "--dry-run",
        ]
    )
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["out_dir"] == os.path.abspath(out)
    assert plan["counts"] == {"train": 4, "validate": 2, "test": 2}
    assert plan["workers"] == 3
    assert plan["positive_fraction"] == 0.5
    assert "records_v2" not in json.dumps(plan)


def test_cli_rejects_odd_split_counts(tmp_path):
    try:
        cli.main(["--out-dir", str(tmp_path / "records"), "--ntrain", "3", "--dry-run"])
    except ValueError as exc:
        assert "even" in str(exc)
    else:
        raise AssertionError("odd split count was accepted")
