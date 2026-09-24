"""Multi-knee members: input and target stretched at every knee (knee-major
channel blocks), per-knee validation, the balanced loss, inference, the CLI
spec and the origin.json record."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import tensorflow as tf
from tf_keras.layers import Input, UpSampling2D
from tf_keras.models import Model as KerasModel

from euclid_polish import ensemble as ens_mod
from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel, MemberTrainSpec
from euclid_polish.model import Model
from euclid_polish.training.augmentation import (
    asinh_stretch_hr,
    asinh_stretch_lr,
    asinh_stretch_multi_knee,
    inverse_asinh_stretch_multi_knee,
    stretch_pair,
)
from euclid_polish.training.inference import (
    default_head_knee,
    infer_checkpoint_asinh_knees,
    reconstruct,
    reconstruct_heads,
)
from euclid_polish.training.losses import build_loss, knee_balanced_loss
from euclid_polish.training.models.common import evaluate
from euclid_polish.training.models.wdsr import wdsr
from euclid_polish.training.trainer import Trainer
from scripts.train_ensemble import build_specs, parse_args

KNEES = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def test_multi_knee_stretch_is_knee_major_and_invertible():
    x = np.random.default_rng(0).normal(0.0, 300.0, (2, 5, 5, 4)).astype(np.float32)
    y = asinh_stretch_multi_knee(tf.constant(x), KNEES)
    assert y.shape == (2, 5, 5, 24)
    for k, q in enumerate(KNEES):
        np.testing.assert_allclose(y[..., 4 * k:4 * k + 4], np.arcsinh(x / q),
                                   rtol=1e-5, atol=1e-5)
    back = inverse_asinh_stretch_multi_knee(y, KNEES).numpy()
    assert back.shape == (6, 2, 5, 5, 4)
    for head in back:
        np.testing.assert_allclose(head, x, rtol=1e-4, atol=1e-2)


def test_stretch_pair_keeps_the_single_knee_path():
    rng = np.random.default_rng(1)
    lr = tf.constant(rng.uniform(0, 500, (3, 3, 4)).astype(np.float32))
    hr = tf.constant(rng.uniform(0, 500, (6, 6, 4)).astype(np.float32))
    a, b = stretch_pair(lr, hr, knee=10.0)
    np.testing.assert_array_equal(a, asinh_stretch_lr(lr, knee=10.0))
    np.testing.assert_array_equal(b, asinh_stretch_hr(hr, knee=10.0))
    a, b = stretch_pair(lr, hr, knees=(1.0, 100.0))
    assert a.shape[-1] == 8 and b.shape[-1] == 8


def test_fresh_multi_knee_model_reads_and_writes_every_knee(tmp_path):
    m = Model(str(tmp_path / "m"), scale=2, num_res_blocks=1, asinh_knees=KNEES)
    assert m._tf_model.inputs[0].shape[-1] == 24
    assert m._tf_model.outputs[0].shape[-1] == 24
    assert m._knee_kw() == {"knees": KNEES}
    with pytest.raises(ValueError):
        Model(str(tmp_path / "x"), num_res_blocks=1, asinh_knee=10.0, asinh_knees=KNEES)


def test_checkpoint_knees_come_from_origin_json(tmp_path):
    d = tmp_path / "member_195"
    (d / "loss_best").mkdir(parents=True)
    (d / "origin.json").write_text(json.dumps({"asinh_knee": None, "asinh_knees": list(KNEES)}))
    assert infer_checkpoint_asinh_knees(str(d)) == KNEES
    assert infer_checkpoint_asinh_knees(str(d / "loss_best")) == KNEES
    single = tmp_path / "member_170"
    single.mkdir()
    (single / "origin.json").write_text('{"asinh_knee": 10}')
    assert infer_checkpoint_asinh_knees(str(single)) is None


def test_validation_scores_each_knee_against_its_own_peak():
    offsets = [0.05 * (k + 1) for k in range(len(KNEES))]
    hr = tf.zeros((1, 4, 4, 24))
    sr = tf.concat([tf.fill((1, 4, 4, 4), d) for d in offsets], axis=-1)
    metrics = evaluate(lambda _lr: sr, [(tf.zeros((1, 2, 2, 24)), hr)], knees=KNEES)
    expected = [10.0 * math.log10(math.asinh(Config.PSNR_PEAK_E / q) ** 2 / d ** 2)
                for q, d in zip(KNEES, offsets, strict=True)]
    np.testing.assert_allclose(metrics["psnr_knee"].numpy(), expected, rtol=1e-5)
    assert float(metrics["psnr_stretched"]) == pytest.approx(np.mean(expected), rel=1e-5)
    np.testing.assert_allclose(metrics["psnr_band_stretched"].numpy(),
                               [np.mean(expected)] * 4, rtol=1e-5)
    assert np.isfinite(float(metrics["psnr_raw"]))


def test_balanced_loss_is_the_geometric_mean_of_the_knee_losses():
    residuals = [10.0 ** -k for k in range(len(KNEES))]
    a = tf.zeros((1, 3, 3, 24))
    b = tf.concat([tf.fill((1, 3, 3, 4), r) for r in residuals], axis=-1)
    balanced = knee_balanced_loss("l2", len(KNEES))
    assert float(balanced(a, b)) == pytest.approx(10.0 ** -2.5, rel=1e-4)
    # The plain loss is set almost entirely by the lowest knee's residual.
    assert float(build_loss("l2")(a, b)) == pytest.approx(
        math.sqrt(sum(r * r for r in residuals) / len(KNEES)), rel=1e-4)
    # Balanced: scaling any one knee's error by 10 moves the loss equally.
    for k in (0, 5):
        scaled = [r * (10.0 if i == k else 1.0) for i, r in enumerate(residuals)]
        bk = tf.concat([tf.fill((1, 3, 3, 4), r) for r in scaled], axis=-1)
        assert float(balanced(a, bk)) == pytest.approx(10.0 ** -2.5 * 10.0 ** (1 / 6), rel=1e-4)


def test_reconstruct_returns_each_knees_image():
    inp = Input(shape=(None, None, 24))
    upsample = KerasModel(inp, UpSampling2D(size=2, interpolation="nearest")(inp))
    x = np.random.default_rng(2).uniform(1.0, 500.0, (6, 6, 4)).astype(np.float32)
    expected = np.kron(x, np.ones((2, 2, 1), np.float32))
    heads = reconstruct_heads(upsample, x, knees=KNEES)
    assert heads.shape == (6, 12, 12, 4)
    for head in heads:
        np.testing.assert_allclose(head, expected, rtol=1e-4)
    assert default_head_knee(KNEES) == 100.0
    _lr, sr = reconstruct(upsample, x, knees=KNEES)
    np.testing.assert_allclose(sr, expected, rtol=1e-4)
    with pytest.raises(ValueError):
        reconstruct(upsample, x, knees=KNEES, head_knee=3.0)


def test_trainer_steps_and_validates_a_multi_knee_member(tmp_path, capsys):
    rng = np.random.default_rng(3)
    lr_e = tf.constant(rng.uniform(0, 200, (2, 8, 8, 4)).astype(np.float32))
    hr_e = tf.constant(rng.uniform(0, 200, (2, 16, 16, 4)).astype(np.float32))
    lr, hr = stretch_pair(lr_e, hr_e, knees=KNEES)
    model = wdsr(scale=2, num_res_blocks=1, nchan_in=24, nchan_out=24)
    trainer = Trainer(model, loss=build_loss("l2"), learning_rate=1e-3,
                      checkpoint_dir=str(tmp_path), knees=KNEES)
    loss, gnorm = trainer.train_step(lr, hr)
    assert np.isfinite(float(loss)) and np.isfinite(float(gnorm))
    result = trainer._validate(tf.data.Dataset.from_tensors((lr, hr)), 1)
    assert np.isfinite(result["psnr_str"])
    assert "per-knee PSNR" in capsys.readouterr().out


def test_member_spec_makes_a_multi_knee_member(tmp_path):
    args = parse_args(["--count", "2", "--steps", "10", "--member-spec",
                       json.dumps([{"loss": "l2", "asinh_knees": list(KNEES)}, {}])])
    first, second = build_specs(args, str(tmp_path / "ens"))
    assert first.asinh_knees == KNEES and first.asinh_knee is None
    assert first.knee_loss == "plain"
    assert second.asinh_knees is None
    run_wide = parse_args(["--count", "1", "--steps", "10",
                           "--asinh-knees", "0.1,1,10", "--knee-loss", "balanced"])
    spec = build_specs(run_wide, str(tmp_path / "ens2"))[0]
    assert spec.asinh_knees == (0.1, 1.0, 10.0) and spec.knee_loss == "balanced"


@pytest.mark.parametrize("member", [
    {"asinh_knee": 10, "asinh_knees": [1, 10]},
    {"asinh_knees": [10]},
    {"asinh_knees": [1, -1]},
    {"asinh_knees": [1, 10], "knee_loss": "median"},
])
def test_member_spec_rejects_bad_multi_knee_settings(member, tmp_path):
    args = parse_args(["--count", "1", "--steps", "10",
                       "--member-spec", json.dumps([member])])
    with pytest.raises(SystemExit):
        build_specs(args, str(tmp_path / "ens"))


class _KneeModel:
    """Model stand-in that accepts the knee knobs and records train()."""

    def __init__(self, checkpoint_dir, *, scale=2, num_res_blocks=32, seed=None,
                 init_weights_from=None, icnr=False, asinh_knee=None, asinh_knees=None):
        self._num_res_blocks = num_res_blocks
        self._asinh_knee = asinh_knee
        self._asinh_knees = tuple(asinh_knees) if asinh_knees else None
        self.trained: dict = {}

    def train(self, lr, hr, **kwargs):
        self.trained = kwargs


def test_train_members_records_the_knees_in_origin_json(tmp_path, monkeypatch):
    monkeypatch.setattr(ens_mod, "Model", _KneeModel)
    base = tmp_path / "ensemble"
    spec = MemberTrainSpec(name="member_195", seed=1, target_steps=10, run_steps=10,
                           loss_norm="l2", asinh_knees=KNEES)
    ens = EnsembleModel(str(base), _models=[])
    ens.train_members("lr.tfrecord", "hr.tfrecord", [spec])
    origin = json.loads((base / "member_195" / "origin.json").read_text())
    assert origin["asinh_knees"] == list(KNEES)
    assert origin["knee_loss"] == "plain" and origin["asinh_knee"] is None
    assert ens._models[0].trained["knee_loss"] == "plain"
