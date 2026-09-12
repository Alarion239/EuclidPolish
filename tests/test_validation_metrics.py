"""Validation metrics must summarize every image in a batch."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from euclid_polish.config import Config
from euclid_polish.training.models.common import evaluate


def _psnr_db(mse: float, peak: float) -> float:
    return float(10.0 * np.log10(peak * peak / mse))


def test_evaluate_averages_psnr_over_every_image_in_each_batch():
    hr = tf.zeros((2, 4, 4, 2), dtype=tf.float32)
    offsets = tf.constant([0.1, 0.5], dtype=tf.float32)[:, None, None, None]

    def model(lr):
        return lr + offsets

    metrics = evaluate(model, [(hr, hr)])

    peak = float(Config.PSNR_PEAK_STRETCHED)
    expected = np.mean([_psnr_db(0.1 ** 2, peak), _psnr_db(0.5 ** 2, peak)])
    assert float(metrics["psnr_stretched"]) == pytest.approx(expected, rel=1e-4)
    np.testing.assert_allclose(
        metrics["psnr_band_stretched"].numpy(), [expected, expected], rtol=1e-4,
    )
    single_image = _psnr_db(0.1 ** 2, peak)
    assert float(metrics["psnr_stretched"]) < single_image - 1.0
