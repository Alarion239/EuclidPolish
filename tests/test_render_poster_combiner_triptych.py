"""The poster triptych runs only the members the production gate reads."""

from __future__ import annotations

import importlib

import numpy as np

from euclid_polish.eval.combiner import save_combiner
from euclid_polish.eval.spatial_gate import MIX_LINEAR, SpatialGateCombiner
from euclid_polish.eval.spatial_gate_fit import init_params

triptych = importlib.import_module("scripts.render_poster_combiner_triptych")

LABELS = ["170·psnr", "171·psnr", "172·psnr"]


def _pruned_gate() -> SpatialGateCombiner:
    params = init_params(2, 4, 8, False, [0, 0, 0, 0], seed=1)
    return SpatialGateCombiner(list(LABELS), params, width=8, use_lr=False,
                               active_members=(0, 2), mix_space=MIX_LINEAR)


class _Member:
    def __init__(self, value: float) -> None:
        self.value = value

    def upsample_array(self, lr):
        return np.full((2 * lr.shape[0], 2 * lr.shape[1], 4), self.value, np.float32)


class _Ensemble:
    built: list[dict] = []

    def __init__(self, ckpt_root, **kwargs) -> None:
        _Ensemble.built.append(kwargs)
        self.member_labels = list(kwargs["labels"])
        self.members = [_Member(float(label.split("·")[0]))
                        for label in self.member_labels]


def test_gate_run_labels_are_the_members_it_reads_unless_all_are_asked():
    gate = _pruned_gate()

    assert triptych._gate_run_labels(gate, all_members=False) == ["170·psnr", "172·psnr"]
    assert triptych._gate_run_labels(gate, all_members=True) == LABELS


def test_run_members_restores_only_the_requested_checkpoints(monkeypatch):
    _Ensemble.built.clear()
    monkeypatch.setattr(triptych, "EnsembleModel", _Ensemble)
    lr = np.zeros((4, 4, 4), np.float32)

    stack = triptych._run_members(lr, ckpt_root="ckpt", labels=["170·psnr", "172·psnr"])

    assert _Ensemble.built == [{"starless": False, "labels": ["170·psnr", "172·psnr"]}]
    assert stack.shape == (2, 8, 8, 4)
    assert stack[1, 0, 0, 0] == 172.0


def test_the_active_only_stack_gives_the_full_stack_gate_output():
    gate = _pruned_gate()
    rng = np.random.default_rng(2)
    full = rng.uniform(0.0, 40.0, (3, 8, 8, 4)).astype(np.float32)

    np.testing.assert_allclose(gate.apply_field(full[[0, 2]]), gate.apply_field(full),
                               rtol=1e-6, atol=1e-6)


def test_all_members_flag_is_off_by_default():
    assert triptych._parse_args([]).all_members is False
    assert triptych._parse_args(["--all-members"]).all_members is True


def test_active_combiner_finds_the_gate_by_the_members_it_reads(tmp_path, monkeypatch):
    """Like production: members that joined after the fit (199-202) do not
    hide the gate, an unread member that left is dropped in memory, and a
    read member that left makes it unavailable."""
    save_combiner(_pruned_gate(), str(tmp_path))
    active = {"labels": [*LABELS, "199·psnr", "200·psnr"]}
    monkeypatch.setattr(triptych, "regime_labels",
                        lambda ckpt_root, starless: list(active["labels"]))
    gate = triptych._active_combiner(str(tmp_path), "ckpt")
    assert gate is not None and gate.member_labels == LABELS
    active["labels"] = ["170·psnr", "172·psnr", "199·psnr"]          # 171 (unread) left
    gate = triptych._active_combiner(str(tmp_path), "ckpt")
    assert gate.member_labels == ["170·psnr", "172·psnr"]
    assert triptych._gate_run_labels(gate, all_members=True) == ["170·psnr", "172·psnr"]
    active["labels"] = ["171·psnr", "172·psnr"]                      # 170 (read) left
    assert triptych._active_combiner(str(tmp_path), "ckpt") is None
