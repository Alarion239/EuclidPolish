"""train_ensemble.py: CLI args → MemberTrainSpec lists for the three modes."""
from __future__ import annotations

import json
import os

import pytest

from scripts.train_ensemble import build_specs, parse_args


def _mk_member(base, i, *, starless=None):
    d = os.path.join(base, f"member_{i:02d}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "checkpoint"), "w") as f:
        f.write("x")
    if starless is not None:
        with open(os.path.join(d, "origin.json"), "w") as f:
            json.dump({"starless": starless}, f)
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


def test_array_add_selects_only_its_positional_member(tmp_path, monkeypatch):
    base = str(tmp_path / "ens")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    args = parse_args(["--mode", "add", "--count", "3", "--steps", "1000",
                       "--base-seed", "50", "--array-task",
                       "--member-names", "member_09,member_10,member_11"])
    (spec,) = build_specs(args, base)
    assert (spec.name, spec.seed) == ("member_10", 51)
    assert not os.path.exists(os.path.join(base, "member_09"))
    assert os.path.isdir(os.path.join(base, "member_10"))


def test_array_add_collision_claims_name_after_sibling_reservations(
        tmp_path, monkeypatch):
    base = str(tmp_path / "ens")
    _mk_member(base, 10)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    args = parse_args(["--mode", "add", "--count", "2", "--steps", "10",
                       "--base-seed", "50",
                       "--array-task",
                       "--member-names", "member_09,member_10"])
    (spec,) = build_specs(args, base)
    assert (spec.name, spec.seed) == ("member_11", 51)
    assert os.path.isdir(os.path.join(base, "member_11"))


def test_array_collision_does_not_claim_a_sibling_target(
        tmp_path, monkeypatch):
    base = str(tmp_path / "ens")
    _mk_member(base, 9)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    args = parse_args(["--mode", "add", "--count", "3", "--steps", "10",
                       "--array-task",
                       "--member-names", "member_09,member_10,member_11"])
    (spec,) = build_specs(args, base)
    assert spec.name == "member_12"
    assert not os.path.exists(os.path.join(base, "member_10"))
    assert not os.path.exists(os.path.join(base, "member_11"))


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


def test_continue_mode_can_target_one_absolute_step(tmp_path, monkeypatch):
    base = str(tmp_path / "ens")
    _mk_member(base, 3)
    monkeypatch.setattr("scripts.train_ensemble.checkpoint_step",
                        lambda d: 2000)
    args = parse_args(["--mode", "continue", "--members", "member_03",
                       "--target-steps", "5000"])
    (s,) = build_specs(args, base)
    assert (s.name, s.target_steps, s.run_steps, s.op) == \
        ("member_03", 5000, 3000, "continue")


def test_continue_target_skips_member_already_there(tmp_path, monkeypatch):
    base = str(tmp_path / "ens")
    _mk_member(base, 3)
    monkeypatch.setattr("scripts.train_ensemble.checkpoint_step",
                        lambda d: 5000)
    args = parse_args(["--mode", "continue", "--members", "member_03",
                       "--target-steps", "5000"])
    assert build_specs(args, base) == []


def test_array_continue_selects_one_member_and_keeps_original_seed_offset(
        tmp_path, monkeypatch):
    base = str(tmp_path / "ens")
    _mk_member(base, 3)
    _mk_member(base, 5)
    monkeypatch.setattr("scripts.train_ensemble.checkpoint_step", lambda d: 2000)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    args = parse_args(["--mode", "continue",
                       "--members", "member_03,member_05",
                       "--extra-steps", "500", "--base-seed", "20",
                       "--array-task"])
    (spec,) = build_specs(args, base)
    assert (spec.name, spec.seed, spec.target_steps) == ("member_05", 21, 2500)


def test_array_continue_target_uses_selected_members_checkpoint(
        tmp_path, monkeypatch):
    base = str(tmp_path / "ens")
    _mk_member(base, 3)
    _mk_member(base, 5)
    monkeypatch.setattr(
        "scripts.train_ensemble.checkpoint_step",
        lambda d: 2000 if d.endswith("member_03") else 4500,
    )
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    args = parse_args(["--mode", "continue",
                       "--members", "member_03,member_05",
                       "--target-steps", "5000", "--base-seed", "20",
                       "--array-task"])
    (spec,) = build_specs(args, base)
    assert (spec.name, spec.seed, spec.target_steps, spec.run_steps) == \
        ("member_05", 21, 5000, 500)


def test_continue_step_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["--mode", "continue", "--members", "member_03",
                    "--extra-steps", "500", "--target-steps", "5000"])


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


@pytest.mark.parametrize("source_starless", [False, True])
def test_fork_inherits_source_regime(tmp_path, source_starless):
    base = str(tmp_path / "ens")
    _mk_member(base, 2, starless=source_starless)
    args = parse_args([
        "--mode", "fork", "--fork-from", "member_02", "--count", "1",
        "--steps", "1000", "--member-names", "member_09",
        # Deliberately request the opposite regime: fork inheritance is
        # authoritative over run-wide and per-member form defaults.
        "--starless", str(int(not source_starless)),
        "--member-spec",
        json.dumps([{"starless": not source_starless}]),
    ])
    (spec,) = build_specs(args, base)
    assert spec.starless is source_starless


def test_fork_legacy_source_without_regime_is_starfull(tmp_path):
    base = str(tmp_path / "ens")
    _mk_member(base, 2)
    args = parse_args([
        "--mode", "fork", "--fork-from", "member_02", "--count", "1",
        "--steps", "1000", "--member-names", "member_09",
    ])
    (spec,) = build_specs(args, base)
    assert spec.starless is False


def test_fork_requires_source_checkpoint(tmp_path):
    args = parse_args(["--mode", "fork", "--fork-from", "member_02",
                       "--steps", "10"])
    with pytest.raises(SystemExit):
        build_specs(args, str(tmp_path / "ens"))


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
