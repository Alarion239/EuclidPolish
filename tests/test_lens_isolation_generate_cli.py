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
    assert plan["population"] == {
        "sersic_density_arcmin2": 0.0,
        "tng_density_arcmin2": 60.0,
        "tng_redshift_mode": True,
        "lens_density_arcmin2": 20.0,
    }
    assert plan["schema_version"] == 2
    assert "records_v2" not in json.dumps(plan)


def test_cli_accepts_odd_normal_split_counts(tmp_path, capsys):
    rc = cli.main(["--out-dir", str(tmp_path / "records"), "--ntrain", "3", "--dry-run"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["counts"]["train"] == 3
