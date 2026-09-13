"""The source-masked background residual behind the report's MER measurement."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_polish.noise_assessment.measurement import source_masked_residual


def test_residual_preserves_low_frequency_power_and_masks_source():
    side = 128
    yy, xx = np.indices((side, side))
    wave = np.sin(2.0 * np.pi * xx / 40.0)
    image = (
        100.0
        + 10.0 * wave
        + 0.03 * xx
        - 0.02 * yy
        + np.random.default_rng(3).normal(size=(side, side))
    )
    image[60:65, 60:65] += 200.0

    residual, mask = source_masked_residual(image)
    usable = ~mask
    recovered_amplitude = float(
        np.sum(residual[usable] * wave[usable]) / np.sum(np.square(wave[usable]))
    )

    assert mask[62, 62]
    assert recovered_amplitude == pytest.approx(10.0, rel=0.08)
