from __future__ import annotations

import json

import pytest

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
    assert plan["loss_norm"] == "l1"
    assert plan["force"] is False
    assert "lens_weight" not in plan
    assert "flux_weight" not in plan
    assert plan["forward_onthefly"] is False


def test_train_force_is_explicit_in_dry_run(tmp_path, capsys):
    source = tmp_path / "sources" / "member_01"
    source.mkdir(parents=True)
    (source / "checkpoint").write_text("x")
    rc = cli.main(
        [
            "--sources", "member_01",
            "--source-base", str(source.parent),
            "--out-dir", str(tmp_path / "experiment-ensemble"),
            "--force", "--dry-run",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["force"] is True


def test_existing_member_detection_ignores_empty_crash_husks(tmp_path):
    out = tmp_path / "ensemble"
    (out / "member_00").mkdir(parents=True)
    assert cli._existing_members(str(out)) == []
    (out / "member_00" / "checkpoint").write_text("x")
    assert cli._existing_members(str(out)) == [str(out / "member_00")]


def test_train_rerun_fails_early_with_force_instruction(tmp_path):
    source = tmp_path / "sources" / "member_01"
    source.mkdir(parents=True)
    (source / "checkpoint").write_text("source")
    out = tmp_path / "ensemble"
    member = out / "member_00"
    member.mkdir(parents=True)
    (member / "checkpoint").write_text("old")

    with pytest.raises(ValueError, match="rerun with --force"):
        cli.main(
            [
                "--sources", "member_01",
                "--source-base", str(source.parent),
                "--records-dir", str(tmp_path / "missing-records"),
                "--out-dir", str(out),
            ]
        )
