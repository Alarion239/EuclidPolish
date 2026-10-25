"""The Lupton loss (``lupton_loss``): one asinh magnitude per band, softened at
its per-pixel sky scatter, instead of the knee grid; and the member knobs for it
and for brightness consistency (member spec, CLI, continue, origin.json)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish import ensemble as ens_mod
from euclid_polish.ensemble import EnsembleModel, MemberTrainSpec
from euclid_polish.model import Model
from euclid_polish.training.augmentation import asinh_stretch_multi_knee
from euclid_polish.training.losses import build_loss, lupton_loss
from scripts.train_ensemble import _continue_identity, build_specs, parse_args

KNEES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
SOFT = (5.0, 0.5, 0.5, 0.5)


def test_it_is_the_rms_of_asinh_magnitudes_softened_at_the_noise():
    rng = np.random.default_rng(0)
    hr = np.abs(rng.normal(size=(2, 16, 16, 4))) * np.array([50.0, 5.0, 5.0, 5.0])
    hr[:, 8, 8, :] = 3e4                                    # a bright core
    sr = hr * (1.0 + 0.1 * rng.normal(size=hr.shape))
    target = asinh_stretch_multi_knee(tf.constant(hr, tf.float32), KNEES)   # as the trainer gives it
    loss = lupton_loss(build_loss("l2"), KNEES, SOFT, 4)
    s = np.array(SOFT)
    want = np.sqrt(np.mean((np.arcsinh(sr / s) - np.arcsinh(hr / s)) ** 2))
    assert float(loss(tf.constant(sr, tf.float32), target)) == pytest.approx(want, rel=1e-4)
    assert float(loss(tf.constant(hr, tf.float32), target)) == pytest.approx(0.0, abs=1e-5)


def test_it_is_linear_below_the_noise_and_logarithmic_above():
    loss = lupton_loss(build_loss("l2"), KNEES, (1.0,), 1)

    def cost(truth, error):
        target = asinh_stretch_multi_knee(tf.constant([[[[truth]]]], tf.float32), KNEES)
        return float(loss(tf.constant([[[[truth + error]]]], tf.float32), target))

    assert cost(0.0, 0.01) == pytest.approx(0.01, rel=1e-3)     # noise units below the noise
    assert cost(1e3, 1e2) == pytest.approx(cost(1e5, 1e4), rel=1e-3)   # relative error above it


def test_softenings_are_checked():
    with pytest.raises(ValueError, match="one value per band"):
        lupton_loss(build_loss("l2"), KNEES, (1.0, 1.0), 4)
    with pytest.raises(ValueError, match="positive"):
        lupton_loss(build_loss("l2"), KNEES, (1.0, 0.0), 2)


def test_the_model_takes_it_only_for_a_learned_knee_member_without_the_knee_grid_knobs(tmp_path):
    fixed = Model(str(tmp_path / "f"), scale=2, num_res_blocks=1, asinh_knees=KNEES, output_knee=10.0)
    with pytest.raises(ValueError, match="learned-output-knee"):
        fixed._member_loss("l2", "balanced", lupton=True)
    learned = Model(str(tmp_path / "l"), scale=2, num_res_blocks=1, asinh_knees=KNEES,
                    output_knee=10.0, learn_output_knee=True)
    for knob in ({"loss_floor": 1.0}, {"knee_tilt": -0.25}, {"loss_knees": (1.0, 10.0)}):
        with pytest.raises(ValueError, match="knee grid"):
            learned._member_loss("l2", "balanced", lupton=True, **knob)
    assert callable(learned._member_loss("l2", "balanced", lupton=True))


def _specs(tmp_path, name, argv):
    return build_specs(parse_args(["--count", "1", "--steps", "10", *argv]), str(tmp_path / name))


def test_member_spec_cli_and_continue_carry_both_knobs(tmp_path):
    member = {"asinh_knees": list(KNEES), "learn_output_knee": True, "knee_loss": "balanced",
              "forward_onthefly": True, "lupton_loss": True, "brightness_consistency": True}
    spec = _specs(tmp_path, "e1", ["--member-spec", json.dumps([member])])[0]
    assert spec.lupton is True and spec.brightness_consistency is True
    cli = _specs(tmp_path, "e2", ["--asinh-knees", ",".join(map(str, KNEES)), "--learn-output-knee",
                                  "--forward-onthefly", "1", "--lupton-loss",
                                  "--brightness-consistency"])[0]
    assert cli.lupton is True and cli.brightness_consistency is True
    plain = _specs(tmp_path, "e3", [])[0]
    assert plain.lupton is False and plain.brightness_consistency is False
    for bad in ({**member, "lupton_loss": "yes"}, {**member, "loss_floor": 1.0},
                {**member, "knee_tilt": 0.25}, {**member, "learn_output_knee": False},
                {**member, "forward_onthefly": False}):
        with pytest.raises(SystemExit):
            _specs(tmp_path, "e4", ["--member-spec", json.dumps([bad])])
    d = tmp_path / "member_312"
    d.mkdir()
    (d / "origin.json").write_text(json.dumps({"asinh_knees": list(KNEES), "lupton_loss": True,
                                               "brightness_consistency": True}))
    identity = _continue_identity("member_312", str(d), {}, {})
    assert identity["lupton"] is True and identity["brightness_consistency"] is True
    assert "lupton" not in _continue_identity("member_312", str(d), {}, {"lupton_loss": False})


class _Recorder:
    def __init__(self, checkpoint_dir, *, scale=2, num_res_blocks=32, seed=None, init_weights_from=None,
                 icnr=False, asinh_knee=None, asinh_knees=None, output_knee=None, learn_output_knee=False):
        self._num_res_blocks = num_res_blocks
        self._asinh_knee = asinh_knee
        self._asinh_knees = tuple(asinh_knees) if asinh_knees else None
        self._learn_output_knee = bool(learn_output_knee)
        self._output_knee = None if learn_output_knee else output_knee
        self.trained: dict = {}

    def train(self, lr, hr, **kwargs):
        self.trained = kwargs


def test_train_members_records_and_passes_both_knobs(tmp_path, monkeypatch):
    monkeypatch.setattr(ens_mod, "Model", _Recorder)
    base = tmp_path / "ensemble"
    common = {"target_steps": 10, "run_steps": 10, "loss_norm": "l2", "asinh_knees": KNEES,
              "knee_loss": "balanced", "output_knee": 10.0, "learn_output_knee": True}
    on = MemberTrainSpec(name="member_313", seed=1, lupton=True, brightness_consistency=True, **common)
    off = MemberTrainSpec(name="member_314", seed=2, **common)
    ens = EnsembleModel(str(base), _models=[])
    ens.train_members("lr", "hr", [on, off])
    origin = json.loads((base / "member_313" / "origin.json").read_text())
    assert origin["lupton_loss"] is True and origin["brightness_consistency"] is True
    plain_origin = json.loads((base / "member_314" / "origin.json").read_text())
    assert "lupton_loss" not in plain_origin and "brightness_consistency" not in plain_origin
    kw_on, kw_off = (m.trained for m in ens._models)
    assert kw_on["lupton"] is True and kw_on["brightness_consistency"] is True
    assert "lupton" not in kw_off and "brightness_consistency" not in kw_off
