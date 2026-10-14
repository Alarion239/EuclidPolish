"""Faster training steps: an XLA-compiled gradient step and bfloat16 residual
blocks, wired from the member spec through ``Model.train`` to the trainer."""
from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish import ensemble as ens_mod
from euclid_polish.config import Config
from euclid_polish.ensemble import EnsembleModel, MemberTrainSpec
from euclid_polish.image import Image
from euclid_polish.image.tfio import tfrecord_path, write_images
from euclid_polish.model import Model
from euclid_polish.training.augmentation import stretch_pair
from euclid_polish.training.losses import channel_balanced_loss, knee_stretched_loss
from euclid_polish.training.models.wdsr import PrecisionWeightNorm, wdsr
from euclid_polish.training.trainer import Trainer, seed_everything
from scripts.train_ensemble import build_specs, parse_args

KNEES = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def _learned_net():
    return wdsr(scale=2, num_res_blocks=1, nchan_in=24, nchan_out=4, input_knees=6,
                learned_output_knee=10.0, eager_weight_norm_init=True)


def _batch(seed=3):
    rng = np.random.default_rng(seed)
    lr_e = tf.constant(rng.uniform(0, 200, (2, 8, 8, 4)).astype(np.float32))
    hr_e = tf.constant(rng.uniform(0, 200, (2, 16, 16, 4)).astype(np.float32))
    return stretch_pair(lr_e, hr_e, knees=KNEES)


def test_the_xla_step_takes_the_same_step_as_the_graph_step(tmp_path):
    lr, hr = _batch()
    loss = knee_stretched_loss(channel_balanced_loss("l2"), KNEES)
    results = []
    for jit in (False, True):
        seed_everything(5)
        net = _learned_net()
        trainer = Trainer(net, loss=loss, learning_rate=1e-3, checkpoint_dir=str(tmp_path / str(jit)),
                          knees=KNEES, output_electrons=True, jit_compile=jit)
        value, gnorm = trainer.train_step(lr, hr)
        results.append((float(value), float(gnorm), [w.copy() for w in net.get_weights()],
                        int(trainer.checkpoint.step.numpy())))
    (l0, g0, w0, s0), (l1, g1, w1, s1) = results
    np.testing.assert_allclose(l1, l0, rtol=1e-5)
    np.testing.assert_allclose(g1, g0, rtol=1e-4)
    for a, b in zip(w1, w0, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)
    assert s0 == s1 == 1                             # the counter runs outside XLA too


def _write_records(tmp_path, n=3):
    """Tiny 4-band train+validate record pairs for ``Model.train``."""
    size = Config.DEFAULT_HR_CROP_SIZE
    rng = np.random.default_rng(1)
    for split in ("train", "validate"):
        lrs, hrs = [], []
        for i in range(n):
            hr = rng.uniform(0, 300, (size, size, 4)).astype(np.float32)
            lrs.append(Image(data=hr.reshape(size // 2, 2, size // 2, 2, 4).sum((1, 3)),
                             pixel_scale_arcsec=0.10, band_names=Config.LR_INPUT_BAND_NAMES,
                             is_clean=False, index=i))
            hrs.append(Image(data=hr, pixel_scale_arcsec=0.05, band_names=Config.HR_TARGET_BAND_NAMES,
                             is_clean=True, index=i))
        write_images(lrs, f"dirty_{split}", records_dir=str(tmp_path))
        write_images(hrs, f"clean_{split}", records_dir=str(tmp_path))
        write_images(hrs, f"hr_{split}", records_dir=str(tmp_path))      # starfull target
    return tfrecord_path(str(tmp_path), "dirty_train"), tfrecord_path(str(tmp_path), "clean_train")


def test_a_member_trains_with_xla_and_bf16_and_infers_in_float32(tmp_path, capsys):
    lr, hr = _write_records(tmp_path)
    m = Model(str(tmp_path / "member"), scale=2, num_res_blocks=1, asinh_knees=KNEES,
              output_knee=10.0, learn_output_knee=True, seed=11)
    before = [w.copy() for w in m._tf_model.get_weights()]
    m.train(lr, hr, steps=2, batch_size=1, loss_norm="l2", knee_loss="balanced",
            evaluate_every=1, jit_compile=True, mixed_precision=True)
    assert "training step: XLA + bf16 blocks" in capsys.readouterr().out
    # Inference reloads the checkpoint into the standard float32 build.
    assert not any(isinstance(layer, PrecisionWeightNorm) for layer in m._tf_model.layers)
    after = m._tf_model.get_weights()
    assert any(not np.allclose(a, b) for a, b in zip(after, before, strict=True))
    sr = m.upsample_array(np.full((8, 8, 4), 50.0, np.float32))
    assert sr.shape == (16, 16, 4) and np.isfinite(sr).all()


def test_the_training_copy_holds_the_models_weights_and_initial_weight_norms(tmp_path):
    m = Model(str(tmp_path / "member"), scale=2, num_res_blocks=1, asinh_knees=KNEES,
              output_knee=10.0, learn_output_knee=True, seed=4)
    lr, _hr = _batch()
    twin = m._training_network(mixed_precision=False)
    # The stock build sets g = |v| on its first call; the copy starts there.
    np.testing.assert_array_equal(twin(lr).numpy(), m._tf_model(lr).numpy())
    assert all(isinstance(layer, PrecisionWeightNorm) for layer in twin.layers if hasattr(layer, "v"))


class _RecordingModel:
    """Model stand-in for train_members that records what train() received."""

    def __init__(self, checkpoint_dir, **kwargs):
        self._num_res_blocks = kwargs.get("num_res_blocks", 32)
        self._asinh_knee = None
        self.trained: dict = {}

    def train(self, lr, hr, **kwargs):
        self.trained = kwargs


def test_train_members_passes_the_step_speedups_and_records_them(tmp_path, monkeypatch):
    monkeypatch.setattr(ens_mod, "Model", _RecordingModel)
    base = tmp_path / "ensemble"
    spec = MemberTrainSpec(name="member_901", seed=1, target_steps=10, run_steps=10,
                           jit_compile=True, mixed_precision=True)
    ens = EnsembleModel(str(base), _models=[]).train_members("lr", "hr", [spec])
    assert ens._models[0].trained["jit_compile"] is True
    assert ens._models[0].trained["mixed_precision"] is True
    origin = json.loads((base / "member_901" / "origin.json").read_text())
    assert origin["jit_compile"] is True and origin["mixed_precision"] is True


def test_the_training_script_reads_the_speedups_run_wide_and_per_member(tmp_path):
    run_wide = build_specs(parse_args(["--count", "1", "--steps", "10", "--xla", "--bf16"]),
                           str(tmp_path / "a"))[0]
    assert run_wide.jit_compile and run_wide.mixed_precision
    plain = build_specs(parse_args(["--count", "1", "--steps", "10"]), str(tmp_path / "b"))[0]
    assert not plain.jit_compile and not plain.mixed_precision
    per_member = parse_args(["--count", "2", "--steps", "10", "--member-spec",
                             json.dumps([{"xla": True}, {"bf16": True}])])
    first, second = build_specs(per_member, str(tmp_path / "c"))
    assert (first.jit_compile, first.mixed_precision) == (True, False)
    assert (second.jit_compile, second.mixed_precision) == (False, True)
    with pytest.raises(SystemExit):
        build_specs(parse_args(["--count", "1", "--steps", "10", "--member-spec",
                                json.dumps([{"xla": 1}])]), str(tmp_path / "d"))


def test_a_continued_member_keeps_the_speedups_it_was_created_with(tmp_path, monkeypatch):
    base = tmp_path / "ens"
    member = base / "member_902"
    member.mkdir(parents=True)
    (member / "origin.json").write_text(json.dumps({"jit_compile": True, "mixed_precision": True}))
    monkeypatch.setattr("scripts.train_ensemble._has_ckpt", lambda d: True)
    monkeypatch.setattr("scripts.train_ensemble.checkpoint_step", lambda d: 5)
    kept = build_specs(parse_args(["--mode", "continue", "--members", "member_902",
                                   "--target-steps", "10"]), str(base))[0]
    assert kept.jit_compile and kept.mixed_precision
    off = build_specs(parse_args(["--mode", "continue", "--members", "member_902",
                                  "--target-steps", "10", "--member-spec",
                                  json.dumps([{"xla": False}])]), str(base))[0]
    assert not off.jit_compile and off.mixed_precision


@pytest.mark.parametrize("jit", [False, True])
def test_a_device_stretch_takes_the_same_step_as_a_stretched_batch(tmp_path, jit):
    """Raw electron batches stretched inside the (compiled) step = the same
    step as batches the CPU stretched."""
    rng = np.random.default_rng(3)
    lr_e = tf.constant(rng.uniform(0, 200, (2, 8, 8, 4)).astype(np.float32))
    hr_e = tf.constant(rng.uniform(0, 200, (2, 16, 16, 4)).astype(np.float32))
    stretch = Model(str(tmp_path / "m"), scale=2, num_res_blocks=1, asinh_knees=KNEES,
                    output_knee=10.0, learn_output_knee=True).batch_stretch()
    loss = knee_stretched_loss(channel_balanced_loss("l2"), KNEES)
    results = []
    for on_device in (False, True):
        seed_everything(5)
        net = _learned_net()
        trainer = Trainer(net, loss=loss, learning_rate=1e-3,
                          checkpoint_dir=str(tmp_path / f"{jit}{on_device}"),
                          knees=KNEES, output_electrons=True, jit_compile=jit,
                          batch_stretch=stretch if on_device else None)
        batch = (lr_e, hr_e) if on_device else stretch_pair(lr_e, hr_e, knees=KNEES)
        value, gnorm = trainer.train_step(*batch)
        results.append((float(value), float(gnorm), [w.copy() for w in net.get_weights()]))
    (l0, g0, w0), (l1, g1, w1) = results
    np.testing.assert_allclose(l1, l0, rtol=1e-6)
    np.testing.assert_allclose(g1, g0, rtol=1e-5)
    for a, b in zip(w1, w0, strict=True):              # float rounding of the op order
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)


def test_batch_stretch_is_the_members_stretch(tmp_path):
    rng = np.random.default_rng(1)
    lr_e = tf.constant(rng.uniform(0, 500, (1, 4, 4, 4)).astype(np.float32))
    hr_e = tf.constant(rng.uniform(0, 500, (1, 8, 8, 4)).astype(np.float32))
    multi = Model(str(tmp_path / "a"), scale=2, num_res_blocks=1, asinh_knees=(1.0, 100.0))
    for got, want in zip(multi.batch_stretch()(lr_e, hr_e),
                         stretch_pair(lr_e, hr_e, knees=(1.0, 100.0)), strict=True):
        np.testing.assert_array_equal(got.numpy(), want.numpy())
    single = Model(str(tmp_path / "b"), scale=2, num_res_blocks=1, asinh_knee=30.0)
    for got, want in zip(single.batch_stretch()(lr_e, hr_e),
                         stretch_pair(lr_e, hr_e, knee=30.0), strict=True):
        np.testing.assert_array_equal(got.numpy(), want.numpy())
