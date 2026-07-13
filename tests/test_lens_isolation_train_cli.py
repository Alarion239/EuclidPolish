from __future__ import annotations

import json

from scripts import lens_isolation_train as cli


def test_train_dry_run_maps_sources_to_isolated_members(tmp_path, capsys):
    source_base = tmp_path / "sources"
    for name in ("member_01", "member_04"):
        path = source_base / name
        path.mkdir(parents=True)
        (path / "checkpoint").write_text("x")
    records = tmp_path / "records"
    records.mkdir()
    out = tmp_path / "experiment-ensemble"
    rc = cli.main(
        [
            "--sources",
            "member_01,member_04",
            "--source-base",
            str(source_base),
            "--records-dir",
            str(records),
            "--out-dir",
            str(out),
            "--dry-run",
        ]
    )
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert [item["target"] for item in plan["members"]] == [
        str(out / "member_00"),
        str(out / "member_01"),
    ]
    assert plan["lr_peak"] == 1e-5
    assert plan["lr_final"] == 1e-6
