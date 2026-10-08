"""Learned output knee: a single-image multi-knee member whose last layer
learns one asinh output knee per band and outputs electrons."""
from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf
from tf_keras.layers import Input, Lambda, UpSampling2D
from tf_keras.models import Model as KerasModel

from euclid_polish import ensemble as ens_mod
from euclid_polish.ensemble import EnsembleModel, MemberTrainSpec
from euclid_polish.model import Model
from euclid_polish.training.augmentation import asinh_stretch_multi_knee, stretch_pair
from euclid_polish.training.inference import (
    infer_checkpoint_learned_output_knee,
    infer_checkpoint_nchan_in,
    infer_checkpoint_nchan_out,
    infer_checkpoint_num_res_blocks,
    infer_checkpoint_output_knee,
    load_model_from_checkpoint,
    reconstruct,
)
from euclid_polish.training.losses import (
    build_loss,
    channel_balanced_loss,
    knee_expanded_loss,
    knee_stretched_loss,
)
from euclid_polish.training.models.common import evaluate
from euclid_polish.training.models.output_knee import (
    KNEE_MAX_E,
    KNEE_MIN_E,
    LearnedOutputKnee,
    knee_logit_for,
    knees_from_logits,
    learned_output_knees,
)
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.training.trainer import Trainer
from scripts.train_ensemble import build_specs, parse_args

KNEES = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def test_head_starts_at_its_initial_knee_in_every_band():
    head = LearnedOutputKnee(init_knee_e=10.0)
    head.build((None, None, None, 4))
    np.testing.assert_allclose(head.knees().numpy(), [10.0] * 4, rtol=1e-5)


def test_head_maps_asinh_values_to_electrons_with_its_knee():
    head = LearnedOutputKnee(init_knee_e=10.0)
    y = tf.constant(np.linspace(-3.0, 25.0, 32).reshape(1, 2, 4, 4).astype(np.float32))
    expected = 10.0 * np.sinh(np.clip(y.numpy(), -20.0, 20.0))
    np.testing.assert_allclose(head(y).numpy(), expected, rtol=1e-5)


@pytest.mark.parametrize("logit", [-1.0e4, 1.0e4])
def test_head_knee_stays_inside_its_bounds(logit):
    head = LearnedOutputKnee()
    head.build((None, None, None, 4))
    head.output_knee_logit.assign([logit] * 4)
    k = head.knees().numpy()
    assert np.all(k >= KNEE_MIN_E * (1 - 1e-5)) and np.all(k <= KNEE_MAX_E * (1 + 1e-5))


def test_head_knee_gets_a_gradient():
    head = LearnedOutputKnee()
    y = tf.constant(np.full((1, 2, 2, 4), 3.0, np.float32))
    with tf.GradientTape() as tape:
        total = tf.reduce_sum(head(y))
    grad = tape.gradient(total, head.output_knee_logit)
    assert grad is not None and np.all(np.abs(grad.numpy()) > 0)


def test_knee_logit_round_trips_and_rejects_knees_outside_the_bounds():
    for knee in (0.5, 10.0, 3000.0):
        assert knees_from_logits([knee_logit_for(knee)])[0] == pytest.approx(knee, rel=1e-6)
    for knee in (KNEE_MIN_E, KNEE_MAX_E, 0.0, -1.0):
        with pytest.raises(ValueError):
            knee_logit_for(knee)


def test_learned_output_knees_finds_the_head_of_a_model():
    inp = Input(shape=(None, None, 4))
    model = KerasModel(inp, LearnedOutputKnee(init_knee_e=30.0)(inp))
    np.testing.assert_allclose(learned_output_knees(model), [30.0] * 4, rtol=1e-5)
    plain = KerasModel(inp, Lambda(lambda t: t)(inp))
    assert learned_output_knees(plain) is None


def _learned_wdsr(blocks: int = 1, knee: float = 10.0):
    return wdsr(scale=2, num_res_blocks=blocks, nchan_in=24, nchan_out=4, input_knees=6,
                learned_output_knee=knee)


def _x24(seed: int):
    return tf.constant(np.random.default_rng(seed).normal(size=(1, 6, 6, 24)).astype(np.float32))


def test_wdsr_learned_head_outputs_electrons_from_the_stretched_image():
    model = _learned_wdsr()
    assert isinstance(model.layers[-1], LearnedOutputKnee)
    pre = KerasModel(model.inputs, model.layers[-2].output)
    x = _x24(1)
    expected = 10.0 * np.sinh(np.clip(pre(x).numpy(), -20.0, 20.0))
    np.testing.assert_allclose(model(x).numpy(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("blocks", [1, 2])
def test_checkpoint_with_a_learned_head_keeps_depth_channels_and_knees(tmp_path, blocks):
    learned = (3.0, 7.0, 20.0, 400.0)
    model = _learned_wdsr(blocks=blocks)
    model.layers[-1].output_knee_logit.assign([knee_logit_for(q) for q in learned])
    d = str(tmp_path / "ckpt")
    tf.train.Checkpoint(model=model).save(d + "/ckpt")
    assert infer_checkpoint_num_res_blocks(d) == blocks
    assert infer_checkpoint_nchan_in(d) == 24
    assert infer_checkpoint_nchan_out(d, scale=2, nchan_in=24) == 4
    assert infer_checkpoint_learned_output_knee(d) == pytest.approx(learned, rel=1e-5)
    loaded = load_model_from_checkpoint(d, scale=2, num_res_blocks=32)
    np.testing.assert_allclose(learned_output_knees(loaded), learned, rtol=1e-5)
    x = _x24(2)
    np.testing.assert_allclose(loaded(x).numpy(), model(x).numpy(), rtol=1e-5, atol=1e-5)


def test_checkpoint_without_the_head_reports_no_learned_knee(tmp_path):
    model = wdsr(scale=2, num_res_blocks=1, nchan_in=24, nchan_out=4, input_knees=6)
    d = str(tmp_path / "ckpt")
    tf.train.Checkpoint(model=model).save(d + "/ckpt")
    assert infer_checkpoint_learned_output_knee(d) is None
    assert infer_checkpoint_num_res_blocks(d) == 1
    assert learned_output_knees(load_model_from_checkpoint(d, scale=2, num_res_blocks=1)) is None


def test_learned_knee_loss_and_validation_equal_option_2_at_the_initial_knee():
    rng = np.random.default_rng(3)
    hr_e = tf.constant(rng.uniform(0.0, 3000.0, (1, 8, 8, 4)).astype(np.float32))
    target = asinh_stretch_multi_knee(hr_e, KNEES)
    y = tf.asinh(hr_e / 10.0) + 0.01          # an option-2 output, slightly off
    x = 10.0 * tf.sinh(y)                     # the same image in electrons
    for base in (build_loss("l2"), channel_balanced_loss("l2")):
        np.testing.assert_allclose(float(knee_stretched_loss(base, KNEES)(x, target)),
                                   float(knee_expanded_loss(base, 10.0, KNEES)(y, target)),
                                   rtol=1e-4)
    lr = tf.zeros((1, 4, 4, 24))
    learned = evaluate(lambda _lr: x, [(lr, target)], knees=KNEES, output_electrons=True)
    fixed = evaluate(lambda _lr: y, [(lr, target)], knees=KNEES, output_knee=10.0)
    for key in ("psnr_stretched", "psnr_raw", "mae_stretched"):
        np.testing.assert_allclose(float(learned[key]), float(fixed[key]), rtol=1e-4)


def test_trainer_steps_a_learned_knee_member_and_logs_its_knees(tmp_path, capsys):
    rng = np.random.default_rng(7)
    lr_e = tf.constant(rng.uniform(0, 200, (2, 8, 8, 4)).astype(np.float32))
    hr_e = tf.constant(rng.uniform(0, 200, (2, 16, 16, 4)).astype(np.float32))
    lr, hr = stretch_pair(lr_e, hr_e, knees=KNEES)
    model = _learned_wdsr()
    loss = knee_stretched_loss(channel_balanced_loss("l2"), KNEES)
    trainer = Trainer(model, loss=loss, learning_rate=1e-2, checkpoint_dir=str(tmp_path),
                      knees=KNEES, output_electrons=True)
    before = learned_output_knees(model).copy()
    value, gnorm = trainer.train_step(lr, hr)
    assert np.isfinite(float(value)) and np.isfinite(float(gnorm))
    assert not np.allclose(learned_output_knees(model), before)     # the knee trains
    assert np.isfinite(trainer._validate(tf.data.Dataset.from_tensors((lr, hr)), 1)["psnr_str"])
    assert "learned output knee" in capsys.readouterr().out


def test_reconstruct_returns_a_learned_knee_members_electrons_unchanged():
    inp = Input(shape=(None, None, 24))
    electrons = Lambda(lambda t: 1.0e4 * tf.sinh(t[..., 20:24]))(inp)   # the 10⁴ e⁻ block
    model = KerasModel(inp, UpSampling2D(size=2, interpolation="nearest")(electrons))
    x = np.random.default_rng(6).uniform(1.0, 500.0, (6, 6, 4)).astype(np.float32)
    _lr, sr = reconstruct(model, x, knees=KNEES, output_electrons=True)
    np.testing.assert_allclose(sr, np.kron(x, np.ones((2, 2, 1), np.float32)), rtol=1e-4)


def test_fresh_learned_knee_member_builds_the_head_and_infers_in_electrons(tmp_path):
    m = Model(str(tmp_path / "m"), scale=2, num_res_blocks=1, asinh_knees=KNEES,
              output_knee=10.0, learn_output_knee=True)
    assert m._tf_model.inputs[0].shape[-1] == 24 and m._tf_model.outputs[0].shape[-1] == 4
    np.testing.assert_allclose(learned_output_knees(m._tf_model), [10.0] * 4, rtol=1e-5)
    assert m._learn_output_knee and m._output_knee is None
    assert m._knee_kw() == {"knees": KNEES, "output_electrons": True}
    sr = m.upsample_array(np.random.default_rng(8).uniform(0, 50, (6, 6, 4)).astype(np.float32))
    assert sr.shape == (12, 12, 4) and np.all(np.isfinite(sr))
    with pytest.raises(ValueError):
        m.upsample_heads(np.zeros((4, 4, 4), np.float32))
    with pytest.raises(ValueError):          # no starting knee
        Model(str(tmp_path / "x"), num_res_blocks=1, asinh_knees=KNEES, learn_output_knee=True)


def test_learned_knee_member_resumes_with_its_head(tmp_path):
    d = tmp_path / "member"
    m = Model(str(d), scale=2, num_res_blocks=1, asinh_knees=KNEES, output_knee=10.0,
              learn_output_knee=True)
    m._tf_model.layers[-1].output_knee_logit.assign(
        [knee_logit_for(q) for q in (2.0, 5.0, 8.0, 9.0)])
    tf.train.Checkpoint(model=m._tf_model).save(str(d / "ckpt"))
    (d / "origin.json").write_text(json.dumps({
        "asinh_knees": list(KNEES), "knee_loss": "balanced",
        "learned_output_knee": {"init_e": 10.0, "min_e": 0.1, "max_e": 10000.0}}))
    r = Model(str(d), scale=2, num_res_blocks=32)
    assert r._learn_output_knee and r._output_knee is None and r._num_res_blocks == 1
    assert r._knee_kw() == {"knees": KNEES, "output_electrons": True}
    np.testing.assert_allclose(learned_output_knees(r._tf_model), (2.0, 5.0, 8.0, 9.0), rtol=1e-5)
    x = np.random.default_rng(9).uniform(0, 50, (6, 6, 4)).astype(np.float32)
    np.testing.assert_allclose(r.upsample_array(x), m.upsample_array(x), rtol=1e-5, atol=1e-4)


def test_a_fork_of_a_learned_knee_member_keeps_the_head(tmp_path):
    src = tmp_path / "src"
    m = Model(str(src), scale=2, num_res_blocks=1, asinh_knees=KNEES, output_knee=10.0,
              learn_output_knee=True)
    m._tf_model.layers[-1].output_knee_logit.assign(
        [knee_logit_for(q) for q in (4.0, 4.0, 6.0, 6.0)])
    tf.train.Checkpoint(model=m._tf_model).save(str(src / "ckpt"))
    (src / "origin.json").write_text(json.dumps({
        "asinh_knees": list(KNEES), "knee_loss": "balanced",
        "learned_output_knee": {"init_e": 10.0, "min_e": 0.1, "max_e": 10000.0}}))
    fork = Model(str(tmp_path / "fork"), scale=2, num_res_blocks=32, init_weights_from=str(src))
    assert fork._learn_output_knee and fork._output_knee is None
    np.testing.assert_allclose(learned_output_knees(fork._tf_model), (4.0, 4.0, 6.0, 6.0), rtol=1e-5)


def test_learned_knee_member_trains_under_the_stretched_loss(tmp_path):
    m = Model(str(tmp_path / "m"), scale=2, num_res_blocks=1, asinh_knees=KNEES,
              output_knee=10.0, learn_output_knee=True)
    hr_e = tf.constant(np.random.default_rng(10).uniform(0, 3000, (1, 8, 8, 4)).astype(np.float32))
    target = asinh_stretch_multi_knee(hr_e, KNEES)
    expected = knee_stretched_loss(channel_balanced_loss("l2"), KNEES)(hr_e * 1.1, target)
    got = m._member_loss("l2", "balanced")(hr_e * 1.1, target)
    np.testing.assert_allclose(float(got), float(expected), rtol=1e-6)
    fixed = Model(str(tmp_path / "f"), scale=2, num_res_blocks=1, asinh_knees=KNEES,
                  output_knee=10.0)
    y = tf.asinh(hr_e * 1.1 / 10.0)
    np.testing.assert_allclose(float(fixed._member_loss("l2", "balanced")(y, target)),
                               float(expected), rtol=1e-4)


class _LearnedKneeModel:
    """Model stand-in for train_members that resolves the knobs like Model."""

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


def test_train_members_records_a_learned_knee_without_an_output_knee(tmp_path, monkeypatch):
    monkeypatch.setattr(ens_mod, "Model", _LearnedKneeModel)
    base = tmp_path / "ensemble"
    spec = MemberTrainSpec(name="member_206", seed=1, target_steps=10, run_steps=10,
                           loss_norm="l2", asinh_knees=KNEES, knee_loss="balanced",
                           output_knee=10.0, learn_output_knee=True)
    EnsembleModel(str(base), _models=[]).train_members("lr", "hr", [spec])
    origin = json.loads((base / "member_206" / "origin.json").read_text())
    assert origin["learned_output_knee"] == {"init_e": 10.0, "min_e": 0.1, "max_e": 10000.0}
    assert "output_knee" not in origin
    assert infer_checkpoint_output_knee(str(base / "member_206")) is None


def test_member_spec_makes_a_learned_knee_member(tmp_path):
    member = {"asinh_knees": list(KNEES), "output_knee": 10, "knee_loss": "balanced",
              "learn_output_knee": True}
    args = parse_args(["--count", "1", "--steps", "10", "--member-spec", json.dumps([member])])
    spec = build_specs(args, str(tmp_path / "ens"))[0]
    assert spec.learn_output_knee and spec.output_knee == 10.0
    run_wide = parse_args(["--count", "1", "--steps", "10", "--asinh-knees", "0.1,1,10",
                           "--output-knee", "10", "--learn-output-knee"])
    assert build_specs(run_wide, str(tmp_path / "ens2"))[0].learn_output_knee
    plain = parse_args(["--count", "1", "--steps", "10"])
    assert not build_specs(plain, str(tmp_path / "ens3"))[0].learn_output_knee


@pytest.mark.parametrize("member", [
    {"asinh_knees": list(KNEES), "learn_output_knee": True},                      # no starting knee
    {"output_knee": 10, "learn_output_knee": True},                               # not multi-knee
    {"asinh_knees": list(KNEES), "output_knee": 1e5, "learn_output_knee": True},  # outside the bounds
    {"asinh_knees": list(KNEES), "output_knee": 10, "learn_output_knee": 1},      # not a bool
])
def test_member_spec_rejects_bad_learned_knee_settings(member, tmp_path):
    args = parse_args(["--count", "1", "--steps", "10", "--member-spec", json.dumps([member])])
    with pytest.raises(SystemExit):
        build_specs(args, str(tmp_path / "ens"))
