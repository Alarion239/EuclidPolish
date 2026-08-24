"""EnsembleModel.train_members: spec-driven sequential training."""
from __future__ import annotations

import json
import os

import pytest

from euclid_polish import ensemble as ens_mod
from euclid_polish.ensemble import EnsembleModel, MemberTrainSpec


class _FakeModel:
    calls: list = []

    def __init__(self, checkpoint_dir, *, scale=2, num_res_blocks=32,
                 seed=None, init_weights_from=None, icnr=False):
        self.checkpoint_dir = checkpoint_dir
        self._num_res_blocks = num_res_blocks
        self.kwargs = {"seed": seed, "init_weights_from": init_weights_from,
                       "icnr": icnr}

    def train(self, lr, hr, steps=0, batch_size=16, resume_track="latest",
              **kw):
        _FakeModel.calls.append(
            {"dir": self.checkpoint_dir, "steps": steps,
             "resume_track": resume_track, **self.kwargs})


@pytest.fixture(autouse=True)
def _patch_model(monkeypatch):
    _FakeModel.calls = []
    monkeypatch.setattr(ens_mod, "Model", _FakeModel)


def test_train_members_runs_specs_in_order(tmp_path):
    base = str(tmp_path / "ensemble")
    specs = [
        MemberTrainSpec(name="member_09", seed=7, target_steps=1000,
                        op="add", run_steps=1000),
        MemberTrainSpec(name="member_03", seed=3, target_steps=1500,
                        op="continue", run_steps=500),
        MemberTrainSpec(name="member_10", seed=8, target_steps=1000,
                        op="fork", run_steps=1000,
                        init_from=os.path.join(base, "member_03"),
                        forked_from="member_03·psnr"),
    ]
    ens = EnsembleModel(base, _models=[])
    ens.train_members("lr.tfrecord", "hr.tfrecord", specs)
    assert [os.path.basename(c["dir"]) for c in _FakeModel.calls] == \
        ["member_09", "member_03", "member_10"]
    assert _FakeModel.calls[0] == {
        "dir": os.path.join(base, "member_09"), "steps": 1000,
        "seed": 7, "init_weights_from": None, "resume_track": "latest",
        "icnr": False}
    assert _FakeModel.calls[2]["init_weights_from"] == \
        os.path.join(base, "member_03")
    # continue resumes from the PSNR-best track (the model eval uses);
    # add/fork keep max-step resume for crash recovery.
    assert [c["resume_track"] for c in _FakeModel.calls] == \
        ["latest", "psnr", "latest"]


def test_train_members_spec_depth_overrides_run_default(tmp_path):
    base = str(tmp_path / "ensemble")
    calls = []

    class _DepthModel(_FakeModel):
        def __init__(self, checkpoint_dir, *, num_res_blocks=32, **kw):
            super().__init__(checkpoint_dir, num_res_blocks=num_res_blocks, **kw)
            calls.append(num_res_blocks)

    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    mp.setattr(ens_mod, "Model", _DepthModel)
    try:
        specs = [
            MemberTrainSpec(name="member_11", seed=1, target_steps=10,
                            op="add", run_steps=10, num_res_blocks=64),
            MemberTrainSpec(name="member_12", seed=2, target_steps=10,
                            op="add", run_steps=10),           # run default
        ]
        EnsembleModel(base, num_res_blocks=32,
                      _models=[]).train_members("lr", "hr", specs)
    finally:
        mp.undo()
    assert calls == [64, 32]


def test_train_members_writes_origin_for_created_members(tmp_path):
    base = str(tmp_path / "ensemble")
    specs = [MemberTrainSpec(name="member_09", seed=7, target_steps=100,
                             op="fork", run_steps=100, init_from="x",
                             forked_from="member_03·loss",
                             crops_per_field=1, hr_crop_size=510)]
    EnsembleModel(base, _models=[]).train_members(
        "lr", "hr", specs, batch_size=2)
    with open(os.path.join(base, "member_09", "origin.json")) as f:
        o = json.load(f)
    assert o["op"] == "fork" and o["forked_from"] == "member_03·loss"
    assert o["seed"] == 7 and o["target_steps"] == 100
    assert o["batch_size"] == 2
    assert o["crops_per_field"] == 1 and o["hr_crop_size"] == 510
    assert "created_at" in o
    assert "num_res_blocks" in o
    # continue never writes/overwrites origin
    specs2 = [MemberTrainSpec(name="member_09", seed=7, target_steps=200,
                              op="continue", run_steps=100)]
    EnsembleModel(base, _models=[]).train_members("lr", "hr", specs2)
    with open(os.path.join(base, "member_09", "origin.json")) as f:
        assert json.load(f)["op"] == "fork"        # untouched


def test_train_members_empty_specs_raises(tmp_path):
    with pytest.raises(ValueError, match="no member specs"):
        EnsembleModel(str(tmp_path / "e"), _models=[]).train_members(
            "lr", "hr", [])
