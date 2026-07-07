"""train_ensemble.py: CLI args → MemberTrainSpec lists for the three modes."""
from __future__ import annotations

import os

import pytest

from scripts.train_ensemble import build_specs, parse_args


def _mk_member(base, i):
    d = os.path.join(base, f"member_{i:02d}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "checkpoint"), "w") as f:
        f.write("x")
    return d


def test_add_mode_uses_passed_names_and_seeds(tmp_path):
    base = str(tmp_path / "ens")
    args = parse_args(["--mode", "add", "--count", "2", "--steps", "1000",
                       "--base-seed", "50",
                       "--member-names", "member_09,member_10"])
    specs = build_specs(args, base)
    assert [(s.name, s.seed, s.target_steps, s.op, s.run_steps)
            for s in specs] == [
        ("member_09", 50, 1000, "add", 1000),
        ("member_10", 51, 1000, "add", 1000)]
    assert all(s.init_from is None for s in specs)


def test_add_mode_allocates_from_disk_when_names_absent(tmp_path):
    base = str(tmp_path / "ens")
    _mk_member(base, 4)
    args = parse_args(["--mode", "add", "--count", "1", "--steps", "10"])
    assert build_specs(args, base)[0].name == "member_05"


def test_add_collision_shifts_past_existing_dir(tmp_path):
    base = str(tmp_path / "ens")
    _mk_member(base, 9)                       # queued twin already created it
    args = parse_args(["--mode", "add", "--count", "1", "--steps", "10",
                       "--member-names", "member_09"])
    assert build_specs(args, base)[0].name == "member_10"


def test_continue_mode_reads_current_step(tmp_path, monkeypatch):
    base = str(tmp_path / "ens")
    _mk_member(base, 3)
    monkeypatch.setattr("scripts.train_ensemble.checkpoint_step",
                        lambda d: 2000)
    args = parse_args(["--mode", "continue", "--members", "member_03",
                       "--extra-steps", "500"])
    (s,) = build_specs(args, base)
    assert (s.name, s.target_steps, s.run_steps, s.op) == \
        ("member_03", 2500, 500, "continue")
    assert s.init_from is None


def test_continue_requires_existing_checkpoint(tmp_path):
    args = parse_args(["--mode", "continue", "--members", "member_08",
                       "--extra-steps", "500"])
    with pytest.raises(SystemExit):
        build_specs(args, str(tmp_path / "ens"))


def test_fork_mode_builds_init_from(tmp_path):
    base = str(tmp_path / "ens")
    d = _mk_member(base, 2)
    lb = os.path.join(d, "loss_best")
    os.makedirs(lb)
    with open(os.path.join(lb, "checkpoint"), "w") as f:
        f.write("x")
    args = parse_args(["--mode", "fork", "--fork-from", "member_02",
                       "--fork-track", "loss", "--count", "2",
                       "--steps", "1000", "--base-seed", "7",
                       "--member-names", "member_09,member_10"])
    specs = build_specs(args, base)
    assert [s.name for s in specs] == ["member_09", "member_10"]
    assert all(s.init_from == os.path.join(base, "member_02", "loss_best")
               for s in specs)
    assert all(s.forked_from == "member_02·loss" for s in specs)
    assert [s.seed for s in specs] == [7, 8]
    assert all(s.op == "fork" for s in specs)


def test_fork_requires_source_checkpoint(tmp_path):
    args = parse_args(["--mode", "fork", "--fork-from", "member_02",
                       "--steps", "10"])
    with pytest.raises(SystemExit):
        build_specs(args, str(tmp_path / "ens"))


def test_n_members_is_add_count_alias(tmp_path):
    args = parse_args(["--n-members", "3", "--steps", "10"])
    specs = build_specs(args, str(tmp_path / "ens"))
    assert len(specs) == 3 and all(s.op == "add" for s in specs)


def test_required_record_names_drops_dirty_when_all_onthefly(tmp_path):
    from scripts.train_ensemble import required_record_names

    args = parse_args(["--count", "2", "--steps", "10",
                       "--forward-onthefly", "1"])
    specs = build_specs(args, str(tmp_path / "a"))
    names = required_record_names(specs)
    assert "dirty_train" not in names          # skip-dirty generation is fine
    assert {"clean_train", "dirty_validate", "clean_validate"} <= set(names)

    # ONE record-mode member in the batch → dirty_train required again.
    args = parse_args(["--count", "2", "--steps", "10",
                       "--forward-onthefly", "1",
                       "--member-spec", '[{"forward_onthefly": false}]'])
    specs = build_specs(args, str(tmp_path / "b"))
    assert "dirty_train" in required_record_names(specs)
