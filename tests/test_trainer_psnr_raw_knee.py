"""Validation ``psnr_raw`` un-stretches with the member's own asinh knee.

The loader stretches a single-knee member's data with its knee
(``asinh(x / q)``), so electrons are ``sinh(y) · q`` — not ``sinh(y) · 100``.
The save-best metric ``psnr_stretched`` is unchanged by the knee.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.training.models.common import evaluate
from euclid_polish.training.trainer import Trainer

_KNEE = 10.0


def _pairs():
    """``(sr, hr)`` stretched at ``_KNEE``, plus the PSNR of their electrons."""
    rng = np.random.default_rng(3)
    hr_e = rng.uniform(0.0, 5.0e4, size=(2, 8, 8, 4)).astype(np.float32)
    sr_e = (hr_e + rng.normal(0.0, 300.0, size=hr_e.shape)).astype(np.float32)
    hr = np.arcsinh(hr_e / _KNEE).astype(np.float32)
    sr = np.arcsinh(sr_e / _KNEE).astype(np.float32)
    expected = float(tf.reduce_mean(tf.image.psnr(
        tf.constant(hr_e), tf.constant(sr_e), max_val=float(Config.PSNR_PEAK_E))))
    return sr, hr, expected


def _dataset(sr, hr):
    # The "model" below is the identity, so the LR input carries the SR.
    return tf.data.Dataset.from_tensor_slices((sr, hr)).batch(2)


def _identity_model():
    inp = tf.keras.Input(shape=(None, None, 4))
    return tf.keras.Model(inp, tf.keras.layers.Lambda(lambda t: t)(inp))


def test_evaluate_unstretches_psnr_raw_with_the_knee():
    sr, hr, expected = _pairs()
    metrics = evaluate(lambda x: x, _dataset(sr, hr), knee=_KNEE)
    assert float(metrics["psnr_raw"]) == pytest.approx(expected, abs=1e-3)


def test_evaluate_knee_leaves_psnr_stretched_alone():
    sr, hr, _ = _pairs()
    with_knee = evaluate(lambda x: x, _dataset(sr, hr), knee=_KNEE)
    default = evaluate(lambda x: x, _dataset(sr, hr))
    assert float(with_knee["psnr_stretched"]) == float(default["psnr_stretched"])
    assert float(with_knee["mae_stretched"]) == float(default["mae_stretched"])


def test_trainer_reads_the_member_knee_from_origin_json(tmp_path):
    ckpt = tmp_path / "member_001"
    ckpt.mkdir()
    (ckpt / "origin.json").write_text(json.dumps({"asinh_knee": _KNEE}))
    sr, hr, expected = _pairs()
    tr = Trainer(_identity_model(), checkpoint_dir=str(ckpt))
    metrics = tr.evaluate(_dataset(sr, hr))
    assert float(metrics["psnr_raw"]) == pytest.approx(expected, abs=1e-3)


def test_trainer_explicit_knee(tmp_path):
    sr, hr, expected = _pairs()
    tr = Trainer(_identity_model(), checkpoint_dir=str(tmp_path / "ckpt"), asinh_knee=_KNEE)
    metrics = tr.evaluate(_dataset(sr, hr))
    assert float(metrics["psnr_raw"]) == pytest.approx(expected, abs=1e-3)


def test_trainer_default_knee_member_is_unchanged(tmp_path):
    """No knee anywhere → the config default (100 e⁻), as before."""
    rng = np.random.default_rng(5)
    hr_e = rng.uniform(0.0, 5.0e4, size=(2, 8, 8, 4)).astype(np.float32)
    sr_e = (hr_e + rng.normal(0.0, 300.0, size=hr_e.shape)).astype(np.float32)
    scale = float(Config.STRETCH_SCALE_E)
    hr = np.arcsinh(hr_e / scale).astype(np.float32)
    sr = np.arcsinh(sr_e / scale).astype(np.float32)
    expected = float(tf.reduce_mean(tf.image.psnr(
        tf.constant(hr_e), tf.constant(sr_e), max_val=float(Config.PSNR_PEAK_E))))
    tr = Trainer(_identity_model(), checkpoint_dir=str(tmp_path / "ckpt"))
    metrics = tr.evaluate(_dataset(sr, hr))
    assert float(metrics["psnr_raw"]) == pytest.approx(expected, abs=1e-3)
