"""Loss knees: a single-image multi-knee member reads every knee but is
scored at a subset (e.g. 8 input knees, loss at 0.1–10⁴ e⁻ only)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish import ensemble as ens_mod
from euclid_polish.ensemble import EnsembleModel, MemberTrainSpec
from euclid_polish.model import Model
from euclid_polish.training.augmentation import asinh_stretch_multi_knee
from euclid_polish.training.losses import (
    channel_balanced_loss,
    knee_channel_indices,
    knee_stretched_loss,
)
from scripts.train_ensemble import _continue_identity, build_specs, parse_args

KNEES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
SIX = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def test_knee_channel_indices_pick_whole_knee_blocks():
    assert knee_channel_indices(KNEES, (0.1, 10.0), 4) == [8, 9, 10, 11, 16, 17, 18, 19]
    with pytest.raises(ValueError):
        knee_channel_indices(KNEES, (0.5,), 4)


def _member(tmp_path):
    return Model(str(tmp_path / "m"), scale=2, num_res_blocks=1, asinh_knees=KNEES,
                 output_knee=0.1, learn_output_knee=True)


def test_the_member_loss_on_loss_knees_equals_the_loss_on_those_knees_alone(tmp_path):
    m = _member(tmp_path)
    rng = np.random.default_rng(1)
    hr = tf.constant(rng.uniform(0.0, 50.0, (1, 8, 8, 4)).astype(np.float32))
    sr = hr * tf.constant(rng.uniform(0.8, 1.2, (1, 8, 8, 4)).astype(np.float32))
    target = asinh_stretch_multi_knee(hr, KNEES)            # the member's target: all 8
    got = m._member_loss("l2", "balanced", SIX)(sr, target)
    expected = knee_stretched_loss(channel_balanced_loss("l2"), SIX)(
        sr, asinh_stretch_multi_knee(hr, SIX))
    np.testing.assert_allclose(float(got), float(expected), rtol=1e-6)
    full = m._member_loss("l2", "balanced")(sr, target)
    assert not np.isclose(float(full), float(got))          # the low knees weigh in
    same = m._member_loss("l2", "balanced", KNEES)(sr, target)  # all knees = default
    np.testing.assert_allclose(float(same), float(full), rtol=1e-6)


def test_loss_knees_must_be_input_knees_of_a_single_image_member(tmp_path):
    with pytest.raises(ValueError):
        _member(tmp_path)._member_loss("l2", "balanced", (0.1, 0.5))
    heads = Model(str(tmp_path / "h"), scale=2, num_res_blocks=1, asinh_knees=KNEES)
    with pytest.raises(ValueError):
        heads._member_loss("l2", "balanced", SIX)


def test_member_spec_and_cli_set_loss_knees(tmp_path):
    member = {"asinh_knees": list(KNEES), "learn_output_knee": True, "knee_loss": "balanced",
              "loss_knees": list(SIX)}
    args = parse_args(["--count", "1", "--steps", "10", "--member-spec", json.dumps([member])])
    assert build_specs(args, str(tmp_path / "e1"))[0].loss_knees == SIX
    run_wide = parse_args(["--count", "1", "--steps", "10", "--asinh-knees",
                           ",".join(map(str, KNEES)), "--learn-output-knee",
                           "--loss-knees", "0.1,1,10,100,1000,10000"])
    assert build_specs(run_wide, str(tmp_path / "e2"))[0].loss_knees == SIX
    plain = parse_args(["--count", "1", "--steps", "10"])
    assert build_specs(plain, str(tmp_path / "e3"))[0].loss_knees is None
    bad = {**member, "loss_knees": [0.1, 0.5]}
    args = parse_args(["--count", "1", "--steps", "10", "--member-spec", json.dumps([bad])])
    with pytest.raises(SystemExit):
        build_specs(args, str(tmp_path / "e4"))


def test_continue_keeps_the_recorded_loss_knees(tmp_path):
    d = tmp_path / "member_300"
    d.mkdir()
    (d / "origin.json").write_text(json.dumps({
        "asinh_knees": list(KNEES), "knee_loss": "balanced", "loss_knees": list(SIX)}))
    identity = _continue_identity("member_300", str(d), {"loss_knees": None}, {})
    assert identity["loss_knees"] == SIX
    assert "loss_knees" not in _continue_identity("member_300", str(d), {}, {"loss_knees": None})


class _Recorder:
    """Model stand-in for train_members that records the train kwargs."""

    def __init__(self, checkpoint_dir, *, scale=2, num_res_blocks=32, seed=None,
                 init_weights_from=None, icnr=False, asinh_knee=None, asinh_knees=None,
                 output_knee=None, learn_output_knee=False):
        self._num_res_blocks = num_res_blocks
        self._asinh_knee = asinh_knee
        self._asinh_knees = tuple(asinh_knees) if asinh_knees else None
        self._learn_output_knee = bool(learn_output_knee)
        self._output_knee = None if learn_output_knee else output_knee
        self.trained: dict = {}

    def train(self, lr, hr, **kwargs):
        self.trained = kwargs


def test_train_members_records_and_passes_loss_knees(tmp_path, monkeypatch):
    monkeypatch.setattr(ens_mod, "Model", _Recorder)
    base = tmp_path / "ensemble"
    spec = MemberTrainSpec(name="member_301", seed=1, target_steps=10, run_steps=10,
                           loss_norm="l2", asinh_knees=KNEES, knee_loss="balanced",
                           output_knee=0.1, learn_output_knee=True, loss_knees=SIX)
    ens = EnsembleModel(str(base), _models=[])
    ens.train_members("lr", "hr", [spec])
    origin = json.loads((base / "member_301" / "origin.json").read_text())
    assert origin["loss_knees"] == list(SIX)
    assert ens._models[0].trained["loss_knees"] == SIX
