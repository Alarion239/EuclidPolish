from __future__ import annotations

import json
import os
from pathlib import Path

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


def test_generation_dispatches_process_local_shards_not_shared_threads():
    source = Path("scripts/lens_isolation_generate.py").read_text(encoding="utf-8")
    records = Path("euclid_polish/experiments/lens_isolation/records.py").read_text(encoding="utf-8")

    assert "ProcessPoolExecutor" in source
    assert "ThreadPoolExecutor" not in records


def test_unseeded_generation_reuses_its_persisted_master_seed(tmp_path, monkeypatch):
    records = str(tmp_path / "records")
    counts = {"train": 4, "validate": 2, "test": 2}
    monkeypatch.setattr(cli.secrets, "randbits", lambda _bits: 17)

    first = cli._master_seed_for_run(
        records,
        config_fingerprint="cfg",
        counts=counts,
        requested_seed=-1,
    )
    monkeypatch.setattr(cli.secrets, "randbits", lambda _bits: 29)
    resumed = cli._master_seed_for_run(
        records,
        config_fingerprint="cfg",
        counts=counts,
        requested_seed=-1,
    )

    assert first == resumed == 17
